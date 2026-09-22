"""Pure ranking diagnostics. Geometry flags are NOT semantic error labels."""
import torch

KS = (1, 2, 3, 5, 10)
IOU_THRESHOLD = .5  # Existing P-linear evaluation uses >=, not >.
DEDUP_IOU = .9  # Conservative diagnostic only; never a new evaluation protocol.
GROUPS = ('both_correct', 'deep_only', 'shallow_only', 'both_wrong')
RANK_BINS = ('1', '2', '3', '4-5', '6-10', '11-20', '21-50', '51+', 'none')


def rank_bin(rank: int | None) -> str:
    if rank is None:
        return 'none'
    for lo, hi, label in [(1, 1, '1'), (2, 2, '2'), (3, 3, '3'), (4, 5, '4-5'),
                          (6, 10, '6-10'), (11, 20, '11-20'), (21, 50, '21-50')]:
        if lo <= rank <= hi:
            return label
    assert rank > 50
    return '51+'


def correctness_group(shallow: bool, deep: bool) -> str:
    return {(True, True): 'both_correct', (False, True): 'deep_only',
            (True, False): 'shallow_only', (False, False): 'both_wrong'}[shallow, deep]


def box_iou(boxes: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    assert boxes.ndim == other.ndim == 2 and boxes.shape[1] == other.shape[1] == 4
    assert boxes.device == other.device
    assert torch.isfinite(boxes).all() and torch.isfinite(other).all()
    assert (boxes[:, 2:] >= boxes[:, :2]).all() and (other[:, 2:] >= other[:, :2]).all()
    b, g = boxes.double(), other.double()
    inter = (torch.minimum(b[:, None, 2:], g[None, :, 2:]) -
             torch.maximum(b[:, None, :2], g[None, :, :2])).clamp_min(0).prod(-1)
    union = (b[:, 2:] - b[:, :2]).prod(-1)[:, None] + (g[:, 2:] - g[:, :2]).prod(-1)[None] - inter
    result = inter / union.clamp_min(1e-12)
    assert result.shape == (len(boxes), len(other))
    return result


def forward_algorithm(scores: torch.Tensor, overlaps: torch.Tensor,
                      boxes: torch.Tensor) -> list[dict]:
    """[A,N] scores + [N] cached IoUs + [N,4] clipped boxes -> A diagnostics.

    Stable descending score, original candidate index breaks ties, matching argmax.
    Dedup greedily suppresses boxes with pairwise IoU >= .9 to an earlier survivor.
    This does not establish that survivors correspond to distinct semantic objects.
    """
    assert scores.ndim == 2 and scores.shape[0] > 0 and scores.shape[1] > 0
    a, n = scores.shape
    assert overlaps.shape == (n,) and boxes.shape == (n, 4)
    assert scores.device == overlaps.device == boxes.device == torch.device('cpu')
    assert torch.isfinite(scores).all() and torch.isfinite(overlaps).all()
    assert ((overlaps >= 0) & (overlaps <= 1)).all()
    pairwise = box_iou(boxes, boxes).tolist()
    ious = overlaps.tolist()
    result = []
    for values in scores:
        order = torch.argsort(values, descending=True, stable=True).tolist()
        assert order[0] == int(values.argmax())
        survivors, suppressed = [], []
        for index in order:
            parent = next((j for j in survivors if pairwise[index][j] >= DEDUP_IOU), None)
            if parent is None:
                survivors.append(index)
            else:
                suppressed.append((index, parent))
        modes = {}
        for name, indices in [('raw', order), ('dedup', survivors)]:
            first = next((r + 1 for r, j in enumerate(indices) if ious[j] >= IOU_THRESHOLD), None)
            modes[name] = dict(first_qualified_rank=first, first_rank_bin=rank_bin(first),
                first_qualified_index=None if first is None else indices[first - 1],
                first_qualified_iou=None if first is None else ious[indices[first - 1]],
                first_qualified_score=None if first is None else float(values[indices[first - 1]]),
                hits={str(k): any(ious[j] >= IOU_THRESHOLD for j in indices[:k]) for k in KS},
                all_hit=first is not None, retained=len(indices), top10=indices[:10],
                top10_scores=[float(values[j]) for j in indices[:10]],
                top10_ious=[ious[j] for j in indices[:10]])
        top_iou = ious[order[0]]
        geometry = ('correct' if top_iou >= .5 else 'near_threshold_045_050' if top_iou >= .45
                    else 'overlap_010_045' if top_iou >= .1 else 'low_overlap_lt010')
        raw_rank = modes['raw']['first_qualified_rank']
        availability = ('correct' if raw_rank == 1 else 'candidate_miss' if raw_rank is None
                        else 'outside_top10' if raw_rank > 10 else 'rank6_10' if raw_rank > 5 else 'rank2_5')
        prefix = order[:5]
        result.append(dict(modes=modes, winner=order[0], top1_iou=top_iou, geometry=geometry,
            availability=availability, best_iou=max(ious), best_iou_index=int(overlaps.argmax()),
            iou_regret=max(ious) - top_iou,
            top1_ties=int((values == values[order[0]]).sum()),
            cutoff_ties={str(k): k < n and bool(values[order[k-1]] == values[order[k]]) for k in KS},
            top5_high_overlap_pairs=sum(pairwise[i][j] >= DEDUP_IOU
                                       for pos, i in enumerate(prefix) for j in prefix[pos+1:]),
            top5_survivors=sum(j in survivors for j in prefix),
            suppressed_count=len(suppressed),
            suppressed_qualified_by_unqualified=sum(ious[i] >= .5 and ious[j] < .5 for i, j in suppressed),
            dedup_lost_coverage=modes['raw']['all_hit'] and not modes['dedup']['all_hit']))
    assert len(result) == a
    return result


if __name__ == '__main__':
    torch.manual_seed(31)
    xy = torch.rand(17, 2)
    boxes = torch.cat([xy, xy + torch.rand(17, 2)], dim=1)
    scores, ious = torch.randn(5, 17), torch.rand(17)
    output = forward_algorithm(scores, ious, boxes)
    assert len(output) == 5 and all(len(r['modes']['raw']['top10']) == 10 for r in output)
    for row in output:
        hits = list(row['modes']['raw']['hits'].values()) + [row['modes']['raw']['all_hit']]
        assert hits == sorted(hits)
    print('Random [A,N] / [N] / [N,4] ranking shapes and monotonicity: PASS')
