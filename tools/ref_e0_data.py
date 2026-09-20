"""Dependency-free sample selection and output comparison for E0."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from humanref_pipeline import iou, validate_boxes


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def load_rec_annotations(annotation_path, proposal_path):
    """Read the author's processed REC JSON, not raw COCO/REFER annotations.

    Same query field and image-key lookup as eval_grounding/eval.py. Accept both
    bare box lists and [boxes, objectness] without confusing TWO bare boxes with
    a boxes/scores pair. Objectness is validated, never multiplied into Ref scores.
    """
    annotations = json.loads(Path(annotation_path).read_text(encoding='utf-8'))
    source = json.loads(Path(proposal_path).read_text(encoding='utf-8'))
    assert isinstance(annotations, list) and annotations
    assert isinstance(source, dict)
    rows, proposals = [], {}
    for ann in annotations:
        image = ann['image']
        query = ann['conversations'][1]['value']
        gt = ann['bounding_boxes']
        assert isinstance(image, str) and isinstance(query, str) and query.strip()
        assert len(gt) == 1, 'E0 REC expects exactly one GT box per expression'
        validate_boxes(gt)
        rows.append(dict(id=ann['id'], image_name=image, referring=query,
                         answer_boxes=gt, domain='rec', sub_domain=Path(annotation_path).stem))
        if image not in proposals:
            entry = source[image]  # missing proposals are an error, never substitute GT
            assert isinstance(entry, list)
            paired = (len(entry) == 2 and isinstance(entry[0], list)
                      and (not entry[0] or isinstance(entry[0][0], list)))
            boxes = entry[0] if paired else entry
            if paired:
                scores = entry[1]
                assert isinstance(scores, list) and len(scores) == len(boxes)
                assert all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores)
            validate_boxes(boxes)
            proposals[image] = dict(boxes=boxes)
    assert len({row['id'] for row in rows}) == len(rows), 'Duplicate expression IDs'
    return rows, proposals


def choose_samples(rows, count, seed):
    """Seeded expression sample, with a real same-image query pair first.

    This is an engineering subset, not a representative accuracy estimate.
    Prefer two distinct queries with distinct GT sets; labels are used only for
    sample selection. Candidate lookup must use ALL annotations, before sampling.
    """
    assert 2 <= count <= len(rows)
    by_image = defaultdict(list)
    for row in rows:
        by_image[row['image_name']].append(row)
    pairs = []
    for group in by_image.values():
        for i, first in enumerate(group):
            for second in group[i + 1:]:
                if first['referring'] != second['referring']:
                    different_gt = first['answer_boxes'] != second['answer_boxes']
                    key = canonical_hash([seed, first['id'], second['id']])
                    pairs.append((not different_gt, key, first, second))
    assert pairs, 'Need at least one real same-image, different-query pair for E0'
    pair = min(pairs, key=lambda item: item[:2])
    selected = [pair[2], pair[3]]
    chosen = {row['id'] for row in selected}
    remaining = [row for row in rows if row['id'] not in chosen]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    selected.extend(remaining[:count - 2])
    assert len({row['id'] for row in selected}) == count
    return selected


def compare_decisions(boxes, baseline_scores, actual_scores, gt):
    """REC Top-1 only. IoU=.5 is an evaluation criterion, not a score cutoff.

    Keep the existing BF16 sigmoid scoring path. Exact score ties use the first
    original proposal, explicitly fixed for both arms (torch.topk ties can differ).
    Different winning indices may still both be correct; record both facts.
    """
    assert boxes and len(boxes) == len(baseline_scores) == len(actual_scores)
    assert len(gt) == 1
    validate_boxes(boxes)
    validate_boxes(gt)
    assert all(math.isfinite(s) and 0 <= s <= 1 for s in baseline_scores + actual_scores)
    baseline_top = max(range(len(boxes)), key=baseline_scores.__getitem__)
    actual_top = max(range(len(boxes)), key=actual_scores.__getitem__)
    baseline_iou, actual_iou = iou(boxes[baseline_top], gt[0]), iou(boxes[actual_top], gt[0])
    return dict(top1_equal=baseline_top == actual_top,
                baseline_top1=baseline_top, actual_top1=actual_top,
                baseline_indices=[baseline_top], actual_indices=[actual_top],
                baseline_top1_ties=sum(s == baseline_scores[baseline_top] for s in baseline_scores),
                actual_top1_ties=sum(s == actual_scores[actual_top] for s in actual_scores),
                baseline_iou=baseline_iou, actual_iou=actual_iou,
                baseline_correct=baseline_iou >= .5, actual_correct=actual_iou >= .5,
                candidate_covers_gt=any(iou(b, gt[0]) >= .5 for b in boxes))
