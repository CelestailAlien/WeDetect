"""Paired accuracy and latency statistics, without model or GPU dependencies."""
import math
import statistics

from humanref_pipeline import iou, validate_boxes


def decisions(boxes, gt, logits, sigmoid_scores):
    assert boxes and len(gt) == 1 and len(boxes) == len(logits) == len(sigmoid_scores)
    validate_boxes(boxes)
    validate_boxes(gt)
    assert all(math.isfinite(x) for x in logits + sigmoid_scores)
    assert all(0 <= x <= 1 for x in sigmoid_scores)
    result = {}
    for name, scores in (('bf16_sigmoid', sigmoid_scores), ('raw_logit', logits)):
        winner = max(range(len(scores)), key=scores.__getitem__)
        overlap = iou(boxes[winner], gt[0])
        result[name] = dict(index=winner, iou=overlap, correct=overlap >= .5,
                            top_ties=sum(x == scores[winner] for x in scores))
    return result


def summarize(records):
    assert records and len({r['id'] for r in records}) == len(records)
    assert all(r['passed'] for r in records)
    assert records[0]['checks']
    assert all(set(r['checks']) == set(records[0]['checks']) for r in records)
    accuracy, errors = {}, {}
    for protocol in ('bf16_sigmoid', 'raw_logit'):
        base = [r['decisions']['full'][protocol] for r in records]
        early = [r['decisions']['exit'][protocol] for r in records]
        harms = [r['id'] for r, b, e in zip(records, base, early) if b['correct'] and not e['correct']]
        recovers = [r['id'] for r, b, e in zip(records, base, early) if not b['correct'] and e['correct']]
        bcount, ecount = sum(b['correct'] for b in base), sum(e['correct'] for e in early)
        assert ecount - bcount == len(recovers) - len(harms)
        accuracy[protocol] = dict(full_correct=bcount, exit_correct=ecount,
            full_accuracy=bcount / len(records), exit_accuracy=ecount / len(records),
            delta_pp=100 * (ecount - bcount) / len(records),
            harmed=len(harms), recovered=len(recovers),
            index_disagreement=sum(b['index'] != e['index'] for b, e in zip(base, early)),
            full_top_ties=sum(b['top_ties'] > 1 for b in base),
            exit_top_ties=sum(e['top_ties'] > 1 for e in early),
            full_mean_iou=statistics.mean(b['iou'] for b in base),
            exit_mean_iou=statistics.mean(e['iou'] for e in early))
        errors[protocol] = dict(harmed=harms, recovered=recovers)
    timing = {}
    for scope in ('forward', 'request'):
        medians = {arm: [] for arm in ('full', 'exit')}
        for record in records:
            rows = record['timing'][scope]
            assert rows and len(rows) % 2 == 0
            assert all(r['arm'] in medians for r in rows)
            by_arm = {arm: [r['wall_ms'] for r in rows if r['arm'] == arm] for arm in medians}
            assert len(by_arm['full']) == len(by_arm['exit']) > 0
            assert all(math.isfinite(v) and v > 0 for values in by_arm.values() for v in values)
            for arm in medians:
                medians[arm].append(statistics.median(by_arm[arm]))
        full, early = statistics.mean(medians['full']), statistics.mean(medians['exit'])
        timing[scope] = dict(full_mean_sample_median_ms=full, exit_mean_sample_median_ms=early,
            speedup_ratio=full / early, latency_reduction_percent=100 * (1 - early / full),
            median_paired_speedup=statistics.median(b / e for b, e in zip(medians['full'], medians['exit'])),
            note='Mean of per-expression medians; paired, alternating-order, warm measurements. Not throughput.')
    report = dict(status='PASSED', samples=len(records), images=len({r['image_name'] for r in records}),
        accuracy=accuracy, timing=timing,
        worst_errors={key: max(r['checks'][key]['max_abs'] for r in records)
                      for key in records[0]['checks']},
        note='Fixed author proposals. No Uni timing, training, test-set claim, or memory-saving claim.')
    return report, errors


def markdown(report, depth):
    lines = ['# Static Ref exit: consistency gates PASSED', '',
        f'{report["samples"]} expressions / {report["images"]} images; exit after {depth} decoder blocks.', '',
        '| Ranking protocol | Full | Exit | Delta pp | Harmed | Recovered |',
        '|---|---:|---:|---:|---:|---:|']
    for protocol, row in report['accuracy'].items():
        lines.append(f'| {protocol} | {row["full_accuracy"]:.2%} | {row["exit_accuracy"]:.2%} '
                     f'| {row["delta_pp"]:+.2f} | {row["harmed"]} | {row["recovered"]} |')
    lines += ['', 'Original BF16-sigmoid protocol and raw-logit diagnostic are BOTH retained.', '',
        '| Warm latency scope | Full ms | Exit ms | Speedup | Reduction |',
        '|---|---:|---:|---:|---:|']
    for scope, row in report['timing'].items():
        lines.append(f'| {scope} | {row["full_mean_sample_median_ms"]:.3f} '
                     f'| {row["exit_mean_sample_median_ms"]:.3f} | {row["speedup_ratio"]:.3f}x '
                     f'| {row["latency_reduction_percent"]:.2f}% |')
    lines += ['', 'forward: prepared GPU inputs -> Ref output; no capture/check hooks.',
        'request: image decode + preprocessing/H2D + Ref + object-score D2H + both Top-1 selections.',
        'Warm filesystem/allocator; fixed proposals already loaded. No Uni, evaluation IoU, hashing, or report I/O.',
        'Layer-list switching is outside timing (static deployment setup); tail weights remain resident.',
        'PASS means implementation consistency, NOT an accuracy gain or acceptable speed/accuracy tradeoff.', '']
    return '\n'.join(lines)
