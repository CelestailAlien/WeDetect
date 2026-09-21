"""Full-split coverage, paired metrics, and non-blocking historical diagnostics."""
import math
import statistics

from humanref_pipeline import iou
from ref_exit_analysis import decisions

PROTOCOLS = ('bf16_sigmoid', 'raw_logit')


def select_rows(rows, limit):
    assert rows and isinstance(limit, int) and 0 <= limit <= len(rows)
    assert len({r['id'] for r in rows}) == len(rows)
    # Preserve annotation order; no GT-based pairing or accuracy-based filtering.
    return list(rows if limit == 0 else rows[:limit])


def history_diagnostic(current, previous):
    for key in ('id', 'query', 'image_name', 'boxes', 'answer_boxes'):
        assert current[key] == previous[key], f'Historical input changed: {key}'
    result = {}
    for arm in ('full', 'exit'):
        old, new = previous['logits'][arm], current['logits'][arm]
        assert len(old) == len(new) > 0
        assert all(math.isfinite(v) for v in old + new)
        errors = [abs(a - b) for a, b in zip(old, new)]
        old_choice = decisions(previous['boxes'], previous['answer_boxes'], old, previous['scores'][arm])
        result[arm] = dict(max_abs=max(errors), mean_abs=statistics.mean(errors),
            logits_exact=old == new, scores_exact=previous['scores'][arm] == current['scores'][arm],
            protocols={p: dict(previous_index=old_choice[p]['index'],
                current_index=current['decisions'][arm][p]['index'],
                index_changed=old_choice[p]['index'] != current['decisions'][arm][p]['index'],
                previous_correct=old_choice[p]['correct'],
                current_correct=current['decisions'][arm][p]['correct']) for p in PROTOCOLS})
    return result


def metrics(records):
    if not records:
        return dict(samples=0, images=0, candidate_coverage=None, accuracy=None)
    covered = sum(r['candidate_covers_gt'] for r in records)
    result = dict(samples=len(records), images=len({r['image_name'] for r in records}),
                  candidate_covered=covered, candidate_coverage=covered / len(records), accuracy={})
    for protocol in PROTOCOLS:
        base = [r['decisions']['full'][protocol] for r in records]
        early = [r['decisions']['exit'][protocol] for r in records]
        bcount, ecount = sum(d['correct'] for d in base), sum(d['correct'] for d in early)
        harmed = sum(b['correct'] and not e['correct'] for b, e in zip(base, early))
        recovered = sum(not b['correct'] and e['correct'] for b, e in zip(base, early))
        assert ecount - bcount == recovered - harmed
        result['accuracy'][protocol] = dict(full_correct=bcount, exit_correct=ecount,
            full_accuracy=bcount / len(records), exit_accuracy=ecount / len(records),
            delta_pp=100 * (ecount - bcount) / len(records), harmed=harmed, recovered=recovered,
            full_accuracy_given_covered=bcount / covered if covered else None,
            exit_accuracy_given_covered=ecount / covered if covered else None,
            index_disagreement=sum(b['index'] != e['index'] for b, e in zip(base, early)),
            full_top_ties=sum(d['top_ties'] > 1 for d in base),
            exit_top_ties=sum(d['top_ties'] > 1 for d in early),
            full_mean_iou=statistics.mean(d['iou'] for d in base),
            exit_mean_iou=statistics.mean(d['iou'] for d in early),
            full_accuracy_iou75=sum(d['iou'] >= .75 for d in base) / len(records),
            exit_accuracy_iou75=sum(d['iou'] >= .75 for d in early) / len(records))
    return result


def summarize(records, selected_ids, all_ids, prior_ids, prior_images):
    assert records and len(set(all_ids)) == len(all_ids)
    assert len(set(selected_ids)) == len(selected_ids) and set(selected_ids) <= set(all_ids)
    assert set(prior_ids) <= set(all_ids), 'Reference samples must belong to this validation split'
    assert [r['id'] for r in records] == selected_ids, 'Missing, duplicate, extra, or reordered sample'
    assert all(r['passed'] for r in records)
    assert records[0]['control_checked'] and all(not r['control_checked'] for r in records[1:])
    assert {'unhooked_full', 'full_depth_sham'} <= set(records[0]['checks'])
    for r in records:
        assert r['checks'] and all(c['passed'] for c in r['checks'].values())
        assert {'exit_boundary', 'exit_readout'} <= set(r['checks'])
        assert all(r['boundary_ranking_equal'][p] for p in PROTOCOLS)
        assert r['executed_full'] == list(range(36)) and r['executed_exit'] == list(range(30))
        assert r['norm_calls_exit'] == r['head_calls_exit'] == 1
        covered = any(iou(b, r['answer_boxes'][0]) >= .5 for b in r['boxes'])
        assert r['candidate_covers_gt'] == covered
        for arm in ('full', 'exit'):
            assert r['decisions'][arm] == decisions(r['boxes'], r['answer_boxes'], r['logits'][arm], r['scores'][arm])
    prior_ids, prior_images = set(prior_ids), set(prior_images)
    summary = metrics(records)
    full_split = selected_ids == all_ids
    summary.update(status='PASSED', stage='ref-full-validation', full_split=full_split,
        evaluation_scope='full_validation' if full_split else 'smoke_only',
        annotation_samples=len(all_ids), exit_depth=30, total_depth=36,
        worst_errors={key: max(r['checks'][key]['max_abs'] for r in records if key in r['checks'])
                      for key in sorted({k for r in records for k in r['checks']})},
        control_samples=sum(r['control_checked'] for r in records),
        groups={
            'previous_expressions': metrics([r for r in records if r['id'] in prior_ids]),
            'new_expressions': metrics([r for r in records if r['id'] not in prior_ids]),
            'new_images': metrics([r for r in records if r['image_name'] not in prior_images])},
        timing_measured=False,
        note='Validation, not an independent test. New expressions may share old images; new_images is a subset of new_expressions. No timing or training.')
    history = [r['history'] for r in records if r['id'] in prior_ids]
    assert all(h is not None for h in history)
    summary['history'] = dict(overlap=len(history), blocking=False, arms={})
    for arm in ('full', 'exit'):
        summary['history']['arms'][arm] = dict(
            max_abs=max(h[arm]['max_abs'] for h in history) if history else None,
            nonexact_logits=sum(not h[arm]['logits_exact'] for h in history),
            nonexact_scores=sum(not h[arm]['scores_exact'] for h in history),
            index_changes={p: sum(h[arm]['protocols'][p]['index_changed'] for h in history) for p in PROTOCOLS})
    errors = {p: dict(
        harmed=[r['id'] for r in records if r['decisions']['full'][p]['correct'] and not r['decisions']['exit'][p]['correct']],
        recovered=[r['id'] for r in records if not r['decisions']['full'][p]['correct'] and r['decisions']['exit'][p]['correct']])
        for p in PROTOCOLS}
    return summary, errors


def markdown(report):
    lines = [f'# RefCOCOg {report["evaluation_scope"]}: PASSED', '',
        f'{report["samples"]}/{report["annotation_samples"]} expressions, {report["images"]} images; full 36 vs exit 30.',
        f'Candidate coverage: {report["candidate_coverage"]:.2%}. No score threshold/NMS. IoU >= 0.5.', '',
        '| Group | N | Ranking | Full | Exit | Delta pp | Harmed | Recovered |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for name, group in [('all_evaluated', report), *report['groups'].items()]:
        if not group['samples']:
            lines.append(f'| {name} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |')
            continue
        for protocol, row in group['accuracy'].items():
            lines.append(f'| {name} | {group["samples"]} | {protocol} | {row["full_accuracy"]:.2%} '
                f'| {row["exit_accuracy"]:.2%} | {row["delta_pp"]:+.2f} | {row["harmed"]} | {row["recovered"]} |')
    lines += ['', 'Both ranking rules are fixed and applied to BOTH arms. Raw logits are BF16 forward outputs, not FP32 inference.',
        'new_images is nested inside new_expressions. Neither group is an official independent test.',
        f'Historical overlap: {report["history"]["overlap"]}; numeric drift is diagnostic, not a failure gate.',
        'Every expression passed same-run boundary/readout and execution checks. First-expression controls also passed.',
        'No latency measurement in this run. Do not infer speedup from total script runtime.',
        'PASS means engineering consistency and coverage, not accuracy improvement.', '']
    return '\n'.join(lines)
