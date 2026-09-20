"""Static prefix execution; no replacement decoder, trained head, or token removal."""
from contextlib import contextmanager

import torch

from ref_e0_core import forward_algorithm  # audited pure RMSNorm + linear readout


@contextmanager
def decoder_prefix(lm, depth: int, deepstack_count: int):
    """Temporarily expose only the prefix to the ORIGINAL HF decoder loop.

    Transformers 4.57.1 iterates self.layers, then applies the original final
    norm. Keep config/position IDs/cache policy/weights untouched. No cache may
    be reused between requests. Single-thread inference only; never save the
    model while this context is active. Tail parameters remain resident in VRAM.
    """
    original = lm.layers
    assert isinstance(original, torch.nn.ModuleList)
    assert len(original) == lm.config.num_hidden_layers
    assert not lm.training and not any(p.requires_grad for p in lm.parameters())
    assert isinstance(depth, int) and 0 < deepstack_count <= depth <= len(original)
    prefix = torch.nn.ModuleList(list(original)[:depth])
    prefix.training = original.training
    try:
        lm.layers = prefix
        yield
    finally:
        lm.layers = original


@contextmanager
def audit_execution(lm, head, boundary: int, expected_blocks: int):
    """Validation ONLY: capture full pre-norm states and prove skipped blocks.

    Enter BEFORE decoder_prefix so hooks cover the entire original layer list.
    Clones/synchronizing assertions here must never be inside timing regions.
    """
    total = len(lm.layers)
    assert 0 < boundary <= expected_blocks <= total
    result = dict(blocks=[], norm_calls=0, head_calls=0)
    handles = []

    def language_input(module, args, kwargs):
        assert not args and 'deepstack_count' not in result
        assert kwargs.get('past_key_values') is None, 'Cache reuse is not supported'
        features = kwargs['deepstack_visual_embeds']
        result['deepstack_count'] = len(features)
        assert 0 < len(features) <= boundary

    def before_block(index):
        def hook(module, args, kwargs):
            assert index == len(result['blocks'])
            result['blocks'].append(index)
            if index == boundary:
                value = args[0] if args else kwargs['hidden_states']
                result['boundary_full'] = value.detach().clone()
        return hook

    def before_norm(module, args):
        result['norm_calls'] += 1
        assert result['norm_calls'] == 1
        result['final_full'] = args[0].detach().clone()
        if boundary == expected_blocks:
            result['boundary_full'] = args[0].detach().clone()

    def before_head(module, args):
        result['head_calls'] += 1

    try:
        handles.append(lm.register_forward_pre_hook(language_input, with_kwargs=True))
        for index, block in enumerate(lm.layers):
            handles.append(block.register_forward_pre_hook(before_block(index), with_kwargs=True))
        handles.append(lm.norm.register_forward_pre_hook(before_norm))
        handles.append(head.register_forward_pre_hook(before_head))
        yield result
        assert result['blocks'] == list(range(expected_blocks)), result['blocks']
        assert result['norm_calls'] == result['head_calls'] == 1
        assert result['boundary_full'].ndim == 3
    finally:
        for handle in handles:
            handle.remove()


def test_algorithm():
    torch.manual_seed(73)
    for dtype in (torch.float32, torch.bfloat16):
        hidden = torch.randn(1, 11, 16).to(dtype)
        norm = torch.randn(16).to(dtype)
        weight, bias = torch.randn(1, 16).to(dtype), torch.randn(1).to(dtype)
        before = hidden.clone()
        output = forward_algorithm(hidden, norm, 1e-6, weight, bias)
        assert output.shape == (1, 11, 1) and output.dtype == dtype
        assert torch.equal(before, hidden)
    # The actual failure found in val500: sigmoid destroys a strict ordering.
    logits = torch.tensor([10.5, 14.875], dtype=torch.bfloat16)
    assert logits.argmax().item() == 1
    assert logits.sigmoid().tolist() == [1., 1.]
    assert logits.sigmoid().argmax().item() == 0
    print('Static-exit pure readout/shape/saturation tests passed.')


if __name__ == '__main__':
    test_algorithm()
