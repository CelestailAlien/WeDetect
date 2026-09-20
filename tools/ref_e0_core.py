"""E0 tensor checks. No model loading, training, or decoder replacement."""
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def forward_algorithm(hidden: torch.Tensor, norm_weight: torch.Tensor,
                      epsilon: float, head_weight: torch.Tensor,
                      head_bias: torch.Tensor) -> torch.Tensor:
    """Frozen Qwen3-VL final RMSNorm + shared binary head; return [..., 1].

    Match Transformers 4.57.1: accumulate RMS in FP32, cast BEFORE multiplying
    the norm weight. Do not normalize a second time or cast the head to FP32.
    """
    assert hidden.ndim in (2, 3) and hidden.shape[-1] > 0
    dim = hidden.shape[-1]
    assert norm_weight.shape == (dim,)
    assert head_weight.shape == (1, dim) and head_bias.shape == (1,)
    assert epsilon > 0
    assert all(t.dtype == hidden.dtype and t.device == hidden.device
               for t in (norm_weight, head_weight, head_bias))
    assert all(torch.isfinite(t).all() for t in (hidden, norm_weight, head_weight, head_bias))
    working = hidden.float()
    normalized = working * torch.rsqrt(working.square().mean(-1, keepdim=True) + epsilon)
    normalized = norm_weight * normalized.to(hidden.dtype)
    logits = F.linear(normalized, head_weight, head_bias)
    assert logits.shape == hidden.shape[:-1] + (1,)
    assert torch.isfinite(logits).all()
    return logits


def compare_tensors(reference: torch.Tensor, actual: torch.Tensor,
                    atol: float, rtol: float) -> dict:
    assert reference.shape == actual.shape and reference.numel() > 0
    assert atol >= 0 and rtol >= 0
    assert torch.isfinite(reference).all() and torch.isfinite(actual).all()
    reference, actual = reference.float(), actual.float()
    error = (reference - actual).abs()
    allowed = atol + rtol * reference.abs()
    return dict(passed=bool((error <= allowed).all()),
                exact=bool(torch.equal(reference, actual)),
                max_abs=float(error.max()), mean_abs=float(error.mean()),
                num_outside_tolerance=int((error > allowed).sum()),
                numel=error.numel(), atol=atol, rtol=rtol)


def object_logits(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    assert positions.dtype == torch.bool and positions.ndim == 2
    assert positions.shape[0] == 1 and positions.any()
    assert logits.shape == positions.shape + (1,)
    values = logits[positions][:, 0]
    assert values.ndim == 1 and torch.isfinite(values).all()
    return values


@contextmanager
def capture_boundaries(model, positions: torch.Tensor):
    """Read-only hooks, batch=1. Hooks are always removed, including on failure.

    h0: input of first decoder block (ROI/position projectors already ran).
    hL: input of FINAL text RMSNorm (including all DeepStack additions).
    Full hL is temporary: it allows same-shape readout to separate indexing bugs
    from BF16 GEMM differences when reading only the compact object rows.
    """
    assert positions.dtype == torch.bool and positions.shape[0] == 1
    lm = model.model.language_model
    captured, handles = {}, []

    def inspect_language_input(module, args, kwargs):
        assert not args, 'Unexpected positional language-model inputs'
        visual = kwargs['visual_pos_masks']
        deepstack = kwargs['deepstack_visual_embeds']
        assert visual.shape == positions.shape and visual.dtype == torch.bool
        assert not (visual & positions).any(), 'Image/object token masks overlap'
        assert visual.any() and len(deepstack) > 0
        assert all(t.shape == (int(visual.sum()), lm.config.hidden_size) for t in deepstack)
        assert 'num_visual_tokens' not in captured, 'Language model called twice'
        captured['num_visual_tokens'] = int(visual.sum())
        captured['deepstack_count'] = len(deepstack)
        captured['position_ids'] = kwargs['position_ids'].detach().clone()

    def capture_zero(module, args, kwargs):
        hidden = args[0] if args else kwargs['hidden_states']
        assert hidden.shape == positions.shape + (lm.config.hidden_size,)
        assert 'h0' not in captured, 'First block called twice'
        captured['h0'] = hidden[positions].detach().clone()

    def capture_final(module, args):
        hidden = args[0]
        assert hidden.shape == positions.shape + (lm.config.hidden_size,)
        assert 'hL_full' not in captured, 'Final norm called twice'
        captured['hL_full'] = hidden.detach().clone()
        captured['hL'] = hidden[positions].detach().clone()

    try:
        handles.append(lm.register_forward_pre_hook(inspect_language_input, with_kwargs=True))
        handles.append(lm.layers[0].register_forward_pre_hook(capture_zero, with_kwargs=True))
        handles.append(lm.norm.register_forward_pre_hook(capture_final))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def profile_forward(model, inputs: dict):
    """Separate CUDA-event pass: no feature capture or disk I/O.

    Event spans are elapsed stream intervals (can include CPU launch gaps), not
    sums of kernel execution time. ROI/assembly is an interval, NOT a pure module.
    """
    events, handles = {}, []

    def start_hook(name):
        def hook(module, args):
            assert name + '_start' not in events, name
            events[name + '_start'] = torch.cuda.Event(enable_timing=True)
            events[name + '_start'].record()
        return hook

    def end_hook(name):
        def hook(module, args, result):
            events[name + '_end'] = torch.cuda.Event(enable_timing=True)
            events[name + '_end'].record()
        return hook

    try:
        for name, module in (('vision', model.model.visual),
                             ('llm', model.model.language_model), ('head', model.out_proj)):
            handles.append(module.register_forward_pre_hook(start_hook(name)))
            handles.append(module.register_forward_hook(end_hook(name)))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        output = model(**inputs)
        end.record()
        torch.cuda.synchronize()
        assert len(events) == 6, 'A timed module did not run exactly once'
        spans = dict(
            before_vision_ms=start.elapsed_time(events['vision_start']),
            vision_ms=events['vision_start'].elapsed_time(events['vision_end']),
            roi_and_input_assembly_ms=events['vision_end'].elapsed_time(events['llm_start']),
            llm_ms=events['llm_start'].elapsed_time(events['llm_end']),
            before_head_ms=events['llm_end'].elapsed_time(events['head_start']),
            head_ms=events['head_start'].elapsed_time(events['head_end']),
            after_head_ms=events['head_end'].elapsed_time(end),
            total_event_ms=start.elapsed_time(end))
        assert all(value >= 0 for value in spans.values())
        return output, spans
    finally:
        for handle in handles:
            handle.remove()


def test_algorithm():
    torch.manual_seed(29)
    for dtype in (torch.float32, torch.bfloat16):
        hidden = torch.randn(1, 13, 24).to(dtype)
        norm_weight = torch.randn(24).to(dtype)
        head_weight, head_bias = torch.randn(1, 24).to(dtype), torch.randn(1).to(dtype)
        before = hidden.clone()
        actual = forward_algorithm(hidden, norm_weight, 1e-6, head_weight, head_bias)
        assert actual.shape == (1, 13, 1) and actual.dtype == dtype
        # Independent expression: explicit reduction for norm and head.
        normalized = (hidden.float() / torch.sqrt(
            hidden.float().square().mean(-1, keepdim=True) + 1e-6)).to(dtype) * norm_weight
        expected = (normalized.float() * head_weight.float()).sum(-1, keepdim=True) + head_bias.float()
        assert torch.allclose(actual.float(), expected, atol=.08 if dtype == torch.bfloat16 else 1e-5,
                              rtol=.02 if dtype == torch.bfloat16 else 1e-5)
        assert torch.equal(before, hidden), 'Readout mutated its input'
        positions = torch.zeros(1, 13, dtype=torch.bool)
        positions[0, [2, 7, 11]] = True
        assert object_logits(actual, positions).shape == (3,)
        assert compare_tensors(actual, actual.clone(), 0, 0)['exact']
        assert not compare_tensors(actual, actual + 2, 0, 0)['passed']
    print('E0 pure readout tests passed (FP32 and BF16).')


if __name__ == '__main__':
    test_algorithm()
