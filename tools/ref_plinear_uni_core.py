"""Pure Uni output validation and checkpoint remapping. No replacement NMS."""
import torch


def forward_algorithm(boxes: torch.Tensor, scores: torch.Tensor,
                      width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert boxes.ndim == 2 and boxes.shape[1] == 4 and 0 < len(boxes) <= 100
    assert scores.shape == (len(boxes),) and boxes.dtype == scores.dtype == torch.float32
    assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
    assert width > 0 and height > 0
    assert ((scores >= 0) & (scores <= 1)).all() and (scores[:-1] >= scores[1:]).all()
    assert (boxes[:, 2:] >= boxes[:, :2]).all()
    assert (boxes >= 0).all() and (boxes[:, 0::2] <= width).all() and (boxes[:, 1::2] <= height).all()
    return boxes, scores  # preserve original prompt-aware NMS order, including ties


def remap_checkpoint(checkpoint):
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    assert isinstance(checkpoint, dict) and checkpoint
    result = {}
    for key, value in checkpoint.items():
        assert isinstance(key, str) and isinstance(value, torch.Tensor)
        if 'backbone' in key:
            key = key.replace('backbone.image_model.model.', 'backbone.')
        if 'bbox_head' in key:
            for old, new in [('bbox_head.head_module.', 'bbox_head.'), ('0.2.', '0.6.'),
                             ('1.2.', '1.6.'), ('2.2.', '2.6.'), ('1.bn', '4'),
                             ('1.conv', '3'), ('0.bn', '1'), ('0.conv', '0')]:
                key = key.replace(old, new)
        assert key not in result, f'Checkpoint remapping collision: {key}'
        result[key] = value
    return result


if __name__ == '__main__':
    torch.manual_seed(42)
    xy = torch.rand(100, 2) * 100
    boxes = torch.cat((xy, xy + torch.rand(100, 2) * 10), dim=1)
    scores = torch.rand(100).sort(descending=True).values
    b, s = forward_algorithm(boxes, scores, 640, 480)
    assert b.shape == (100, 4) and s.shape == (100,)
    assert torch.equal(b, boxes) and torch.equal(s, scores)
    print('Uni random tensor shapes / range / unchanged order: PASS')
