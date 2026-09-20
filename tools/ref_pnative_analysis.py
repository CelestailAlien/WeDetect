"""Pure CPU statistics. Native readout is not a trained linear probe."""
from collections import defaultdict
import math

from humanref_pipeline import iou
from ref_e0_data import compare_decisions


def selected_depths(total):
    assert isinstance(total, int) and total > 0
    return sorted({0, (total + 3) // 4, (total + 1) // 2,
                   (2 * total + 2) // 3, (5 * total + 5) // 6, total})


def layer_result(boxes, gt, baseline_scores, scores, logits):
    assert len(logits) == len(scores) and all(math.isfinite(x) for x in logits)
    decision = compare_decisions(boxes, baseline_scores, scores, gt)
    raw_index = max(range(len(logits)), key=logits.__getitem__)
    raw_iou = iou(boxes[raw_index], gt[0])
    decision.update(
        raw_logit_top1=raw_index, raw_logit_top1_iou=raw_iou,
        raw_logit_correct=raw_iou >= .5,
        sigmoid_vs_logit_top1_differs=raw_index != decision['actual_top1'],
        sigmoid_zero_count=sum(x == 0 for x in scores),
        sigmoid_one_count=sum(x == 1 for x in scores),
        harmed=decision['baseline_correct'] and not decision['actual_correct'],
        recovered=not decision['baseline_correct'] and decision['actual_correct'])
    return decision


def summarize(records, depths):
    assert records and len({r['id'] for r in records}) == len(records)
    assert all(r['passed'] and sorted(map(int, r['layers'])) == depths for r in records)
    assert all(r['layers'][str(depths[-1])]['decision']['top1_equal'] for r in records)
    total = len(records)
    baseline_correct = sum(r['layers'][str(depths[-1])]['decision']['baseline_correct'] for r in records)
    covered = sum(r['layers'][str(depths[-1])]['decision']['candidate_covers_gt'] for r in records)
    curve, errors = [], {}
    for depth in depths:
        decisions = [r['layers'][str(depth)]['decision'] for r in records]
        assert all(d['baseline_correct'] == r['layers'][str(depths[-1])]['decision']['baseline_correct']
                   and d['candidate_covers_gt'] == r['layers'][str(depths[-1])]['decision']['candidate_covers_gt']
                   for r, d in zip(records, decisions))
        assert sum(d['baseline_correct'] for d in decisions) == baseline_correct
        assert sum(d['candidate_covers_gt'] for d in decisions) == covered
        correct = sum(d['actual_correct'] for d in decisions)
        harmed, recovered = sum(d['harmed'] for d in decisions), sum(d['recovered'] for d in decisions)
        assert correct - baseline_correct == recovered - harmed
        assert all(not d['actual_correct'] or d['candidate_covers_gt'] for d in decisions)
        curve.append(dict(depth=depth, samples=total, baseline_correct=baseline_correct,
            correct=correct, accuracy=correct / total,
            delta_pp=100 * (correct - baseline_correct) / total,
            accuracy_given_covered=correct / covered if covered else None,
            index_disagreement_rate=sum(not d['top1_equal'] for d in decisions) / total,
            different_index_both_correct=sum(not d['top1_equal'] and d['baseline_correct']
                                             and d['actual_correct'] for d in decisions),
            harmed_count=harmed, harm_rate_all=harmed / total,
            harm_rate_given_baseline_correct=harmed / baseline_correct if baseline_correct else None,
            recovered_count=recovered,
            recovery_rate_given_baseline_wrong=recovered / (total - baseline_correct)
                if total != baseline_correct else None,
            raw_logit_accuracy=sum(d['raw_logit_correct'] for d in decisions) / total,
            tied_top1_count=sum(d['actual_top1_ties'] > 1 for d in decisions),
            sigmoid_vs_logit_top1_differs=sum(d['sigmoid_vs_logit_top1_differs'] for d in decisions)))
        errors[str(depth)] = {name: [r['id'] for r, d in zip(records, decisions) if d[name]]
                              for name in ('harmed', 'recovered')}
    return dict(samples=total, images=len({r['image_name'] for r in records}),
                baseline_accuracy=baseline_correct / total, candidate_coverage=covered / total,
                curve=curve), errors


def query_pairs(records, depths):
    """Descriptive same-image/different-query pairs, NOT independent observations.

    Only differing GT boxes with mutual IoU<.5 enter this target-switch diagnostic.
    Candidate identity/order must match. No h0 performance is called reasoning.
    """
    groups = defaultdict(list)
    for record in records:
        groups[record['image_name']].append(record)
    pairs = []
    for group in groups.values():
        for i, first in enumerate(group):
            for second in group[i + 1:]:
                if first['query'] == second['query'] or iou(first['answer_boxes'][0], second['answer_boxes'][0]) >= .5:
                    continue
                assert first['boxes_sha256'] == second['boxes_sha256'], 'Same-image pair used different proposals'
                pairs.append(dict(ids=[first['id'], second['id']], image_name=first['image_name'],
                    layers={str(k): dict(
                        selection_changes=first['layers'][str(k)]['decision']['actual_top1']
                            != second['layers'][str(k)]['decision']['actual_top1'],
                        both_correct=first['layers'][str(k)]['decision']['actual_correct']
                            and second['layers'][str(k)]['decision']['actual_correct']) for k in depths}))
    curve = [dict(depth=k, pairs=len(pairs),
                  both_correct_rate=sum(p['layers'][str(k)]['both_correct'] for p in pairs) / len(pairs) if pairs else None,
                  selection_change_rate=sum(p['layers'][str(k)]['selection_changes'] for p in pairs) / len(pairs) if pairs else None)
             for k in depths]
    return dict(pair_count=len(pairs), curve=curve, pairs=pairs,
                note='Overlapping pairs are correlated; changed index alone does not mean correct query following.')
