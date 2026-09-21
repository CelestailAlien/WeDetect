"""Two-forward accuracy validation, reusing the already audited static exit."""
import torch

from ref_e0_core import compare_tensors, forward_algorithm, object_logits
from ref_exit_core import audit_execution, decoder_prefix


def evaluate_pair(model, inputs: dict, positions: torch.Tensor, depth: int,
                  atol: float, rtol: float, control: bool = False):
    """Full and true-exit forwards on this device; no historical tensor gate.

    Hooks are read-only and every expression's pre-norm boundary is checked.
    Full-depth sham and unhooked controls run on the first expression. These
    instrumented passes MUST NOT be used as a latency benchmark.
    """
    assert not torch.is_grad_enabled(), 'Call inside torch.inference_mode()'
    assert positions.dtype == torch.bool and positions.ndim == 2 and positions.shape[0] == 1
    assert positions.any() and inputs.get('past_key_values') is None
    lm, head = model.model.language_model, model.out_proj
    original = lm.layers
    total = len(original)
    assert 0 < depth < total
    with audit_execution(lm, head, depth, total) as full_audit:
        output = model(**inputs)
    baseline = object_logits(output.logits, positions).clone()
    del output
    hidden = full_audit['boundary_full']
    assert hidden.shape == positions.shape + (lm.config.hidden_size,)
    expected = object_logits(forward_algorithm(hidden, lm.norm.weight,
        lm.norm.variance_epsilon, head.weight, head.bias), positions)
    deepstack = full_audit['deepstack_count']
    checks = {}
    if control:
        output = model(**inputs)
        checks['unhooked_full'] = compare_tensors(baseline, object_logits(output.logits, positions), atol, rtol)
        del output
        with decoder_prefix(lm, total, deepstack):
            output = model(**inputs)
        checks['full_depth_sham'] = compare_tensors(baseline, object_logits(output.logits, positions), atol, rtol)
        del output
    with audit_execution(lm, head, depth, depth) as exit_audit:
        with decoder_prefix(lm, depth, deepstack):
            output = model(**inputs)
    early = object_logits(output.logits, positions).clone()
    del output
    assert lm.layers is original
    assert exit_audit['deepstack_count'] == deepstack
    checks['exit_boundary'] = compare_tensors(hidden, exit_audit['final_full'], atol, rtol)
    checks['exit_readout'] = compare_tensors(expected, early, atol, rtol)
    # Even a tolerated rounding error must not silently change the winner.
    ranking_equal = dict(raw_logit=bool(expected.argmax() == early.argmax()),
                         bf16_sigmoid=bool(expected.sigmoid().argmax() == early.sigmoid().argmax()))
    info = dict(checks=checks, boundary_ranking_equal=ranking_equal,
        executed_full=full_audit['blocks'], executed_exit=exit_audit['blocks'],
        deepstack_count=deepstack, norm_calls_exit=exit_audit['norm_calls'],
        head_calls_exit=exit_audit['head_calls'], control_checked=control,
        passed=all(c['passed'] for c in checks.values()) and all(ranking_equal.values()))
    return dict(full=baseline, exit=early), info


if __name__ == '__main__':
    # Pure core readout test: no weights, images, or CUDA required.
    torch.manual_seed(91)
    for dtype in (torch.float32, torch.bfloat16):
        hidden = torch.randn(1, 17, 32).to(dtype)
        norm = torch.randn(32).to(dtype)
        weight, bias = torch.randn(1, 32).to(dtype), torch.randn(1).to(dtype)
        original = hidden.clone()
        result = forward_algorithm(hidden, norm, 1e-6, weight, bias)
        assert result.shape == (1, 17, 1) and result.dtype == dtype
        assert torch.equal(original, hidden)
    print('Full-eval pure readout shape/non-mutation tests passed.')
