"""Read-only depth capture; reuse the validated E0 RMSNorm/linear algorithm."""
from contextlib import contextmanager

import torch

from ref_e0_core import capture_boundaries, forward_algorithm


@contextmanager
def capture_depths(model, positions: torch.Tensor, depths: list[int]):
    """Depth k<L is the INPUT of decoder block k (zero-based).

    It includes k completed blocks AND their DeepStack additions. Depth L is
    final RMSNorm input. Never use output_hidden_states indexing conventions.
    All blocks still execute; this is NOT early-exit inference.
    """
    lm = model.model.language_model
    total = len(lm.layers)
    assert depths == sorted(set(depths)) and depths[0] == 0 and depths[-1] == total
    assert all(0 <= k <= total for k in depths)
    result = dict(states={}, executed_blocks=[])
    handles = []

    def before_block(index):
        def hook(module, args, kwargs):
            assert index == len(result['executed_blocks']), 'Skipped/repeated/out-of-order decoder block'
            result['executed_blocks'].append(index)
            if index in depths and index != 0:
                hidden = args[0] if args else kwargs['hidden_states']
                assert hidden.shape == positions.shape + (lm.config.hidden_size,)
                result['states'][index] = hidden[positions].detach().clone()
        return hook

    try:
        with capture_boundaries(model, positions) as boundary:
            for index, block in enumerate(lm.layers):
                handles.append(block.register_forward_pre_hook(before_block(index), with_kwargs=True))
            yield result
            assert result['executed_blocks'] == list(range(total))
            result['states'][0] = boundary.pop('h0')
            result['states'][total] = boundary.pop('hL')
            result.update(boundary)
            assert sorted(result['states']) == depths
    finally:
        for handle in handles:
            handle.remove()


def read_depths(states: dict[int, torch.Tensor], norm_weight: torch.Tensor,
                epsilon: float, head_weight: torch.Tensor,
                head_bias: torch.Tensor) -> dict[int, torch.Tensor]:
    """Each depth uses exactly one frozen final Norm and the same original head.

    Keep one [N,D] linear call per depth, not a merged [K*N,D] GEMM, so the
    final-layer compact readout follows the already checked E0 numeric path.
    """
    assert states
    shape = next(iter(states.values())).shape
    assert len(shape) == 2 and shape[0] > 0
    assert all(value.shape == shape for value in states.values())
    return {depth: forward_algorithm(hidden, norm_weight, epsilon, head_weight, head_bias)[:, 0]
            for depth, hidden in sorted(states.items())}


if __name__ == '__main__':
    torch.manual_seed(19)
    for dtype in (torch.float32, torch.bfloat16):
        states = {k: torch.randn(7, 16).to(dtype) for k in (0, 2, 4)}
        scale, weight, bias = torch.randn(16).to(dtype), torch.randn(1, 16).to(dtype), torch.randn(1).to(dtype)
        logits = read_depths(states, scale, 1e-6, weight, bias)
        assert set(logits) == {0, 2, 4}
        assert all(value.shape == (7,) and value.dtype == dtype for value in logits.values())
    print('P-native random tensor shape tests passed.')
