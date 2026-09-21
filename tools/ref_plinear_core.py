"""Pure P-linear math. Frozen BF16 features, independent FP32 shared heads."""
import torch
import torch.nn.functional as F


def frozen_norm(hidden: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    assert hidden.ndim == 2 and weight.shape == (hidden.shape[-1],)
    assert hidden.dtype == weight.dtype and hidden.device == weight.device
    assert not hidden.requires_grad and not weight.requires_grad and epsilon > 0
    assert torch.isfinite(hidden).all() and torch.isfinite(weight).all()
    x = hidden.float()
    # The original final RMSNorm casts BEFORE multiplying its frozen scale.
    return weight * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)).to(hidden.dtype)


def forward_algorithm(features: torch.Tensor, weight: torch.Tensor,
                      bias: torch.Tensor) -> torch.Tensor:
    """[B,K,N,D] x [K,D] -> [B,K,N]; one shared head per depth, no position heads."""
    assert features.ndim == 4 and weight.ndim == 2
    _, depths, _, dim = features.shape
    assert weight.shape == (depths, dim) and bias.shape == (depths,)
    assert features.dtype == weight.dtype == bias.dtype == torch.float32
    assert features.device == weight.device == bias.device
    assert not features.requires_grad, 'Backbone/features must stay frozen'
    assert all(torch.isfinite(t).all() for t in (features, weight, bias))
    logits = torch.stack([F.linear(features[:, k], weight[k:k+1], bias[k:k+1])[..., 0]
                          for k in range(depths)], dim=1)
    assert logits.shape == features.shape[:3] and torch.isfinite(logits).all()
    return logits


def compute_loss(logits: torch.Tensor, targets: torch.Tensor,
                 valid: torch.Tensor) -> torch.Tensor:
    """Original soft-label sigmoid focal loss; candidate mean, then query mean.

    Return [K] losses. SUM these for backward: heads are independent, so changing
    the number of probed depths does not change a head's gradient scale.
    Padding contributes nothing; uncovered queries keep their all-zero targets.
    """
    assert logits.ndim == 3 and targets.shape == valid.shape == (logits.shape[0], logits.shape[2])
    assert valid.dtype == torch.bool and valid.any(-1).all()
    assert targets.dtype == logits.dtype == torch.float32
    assert targets.device == valid.device == logits.device
    assert torch.isfinite(logits).all() and torch.isfinite(targets).all()
    assert ((targets >= 0) & (targets <= 1)).all() and not targets.requires_grad
    y = targets[:, None, :].expand_as(logits)
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    pt = p * y + (1 - p) * (1 - y)
    loss = ce * (1 - pt).square() * (.25 * y + .75 * (1 - y))
    per_query = (loss * valid[:, None, :]).sum(-1) / valid.sum(-1)[:, None]
    out = per_query.mean(0)
    assert out.shape == (logits.shape[1],) and torch.isfinite(out).all()
    return out


if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.randn(3, 5, 7, 16)
    w = torch.randn(5, 16, requires_grad=True)
    b = torch.zeros(5, requires_grad=True)
    logits = forward_algorithm(x, w, b)
    assert logits.shape == (3, 5, 7)
    losses = compute_loss(logits, torch.rand(3, 7), torch.ones(3, 7, dtype=torch.bool))
    assert losses.shape == (5,)
    losses.sum().backward()
    assert w.grad.shape == w.shape and b.grad.shape == b.shape and x.grad is None
    print('P-linear random tensor shapes and gradients: PASS')
