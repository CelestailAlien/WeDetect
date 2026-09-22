"""Read-only P-linear cache diagnostics; see REF_TOPK_REPORT.md. No model load."""
import csv
import hashlib
import json
import os
from pathlib import Path

import torch

from humanref_pipeline import digest, save_json
from ref_plinear_core import forward_algorithm as readout
from ref_plinear_data import batches, open_cache, read_json
from ref_topk_core import (DEDUP_IOU, GROUPS, IOU_THRESHOLD, KS, RANK_BINS,
                           box_iou, correctness_group, forward_algorithm)

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get('PL_DEVICE', 'cuda')
BATCH_SIZE = 16  # Same linear readout kernel shape as the original evaluation.
DEEP = 'original_full36_raw_bf16'
NATIVE30 = 'native_d30_raw_bf16'
REVIEW_PER_STRATUM = 3


def aggregate(rows: list[dict], arm: str, mode: str) -> dict:
    items = [r['arms'][arm] for r in rows]
    n = len(items)
    hits = {str(k): sum(r['modes'][mode]['hits'][str(k)] for r in items) for k in KS}
    hits['all'] = sum(r['modes'][mode]['all_hit'] for r in items)
    return dict(n=n, hits=hits, rates={k: v / n if n else None for k, v in hits.items()},
        first_rank_counts={b: sum(r['modes'][mode]['first_rank_bin'] == b for r in items) for b in RANK_BINS},
        geometry_counts={g: sum(r['geometry'] == g for r in items) for g in
                         ('correct', 'near_threshold_045_050', 'overlap_010_045', 'low_overlap_lt010')},
        availability_counts={b: sum(r['availability'] == b for r in items) for b in
                             ('correct', 'rank2_5', 'rank6_10', 'outside_top10', 'candidate_miss')},
        top5_high_overlap_queries=sum(r['top5_high_overlap_pairs'] > 0 for r in items),
        top5_mean_survivors=sum(r['top5_survivors'] for r in items) / n if n else None,
        dedup_lost_coverage=sum(r['dedup_lost_coverage'] for r in items),
        suppressed_qualified_by_unqualified_queries=sum(r['suppressed_qualified_by_unqualified'] > 0 for r in items),
        top1_tied_queries=sum(r['top1_ties'] > 1 for r in items),
        cutoff_tied_queries={str(k): sum(r['cutoff_ties'][str(k)] for r in items) for k in KS})


def summarize(rows: list[dict], shallow_arms: list[str]) -> dict:
    assert rows and len({r['id'] for r in rows}) == len(rows)
    arms = shallow_arms + [DEEP]
    overall = {a: {m: aggregate(rows, a, m) for m in ('raw', 'dedup')} for a in arms}
    pairs = {}
    for shallow in shallow_arms:
        groups = {}
        for group in GROUPS:
            selected = [r for r in rows if r['groups'][shallow] == group]
            groups[group] = {a: {m: aggregate(selected, a, m) for m in ('raw', 'dedup')}
                             for a in (shallow, DEEP)}
        assert sum(groups[g][shallow]['raw']['n'] for g in GROUPS) == len(rows)
        pairs[shallow] = groups
    return dict(n=len(rows), images=len({r['image_key'] for r in rows}), overall=overall, pairs=pairs)


def markdown(summary: dict) -> str:
    lines = ['# RefCOCOg Top-k / paired correctness / box-quality diagnostics', '',
        f'N={summary["n"]} expressions, {summary["images"]} images. Each seed reported separately.', '',
        'IoU >= 0.5. Raw logits; stable candidate-index tie break. No GT insertion or score cutoff.',
        'Dedup: greedy pairwise IoU >= 0.9; geometric diagnostic, NOT semantic object identities or a tuned protocol.',
        'Four groups always use ORIGINAL raw Top-1 decisions; dedup never changes group membership.',
        'Percentages below are conditional on the displayed n; empty groups are NA, not zero.', '',
        '## Overall', '', '| Arm / mode | n | @1 | @2 | @3 | @5 | @10 | @all |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def line(label, metric):
        values = ['NA' if metric['rates'][k] is None else f'{100*metric["rates"][k]:.2f}'
                  for k in ('1', '2', '3', '5', '10', 'all')]
        return '| ' + ' | '.join([label, str(metric['n'])] + values) + ' |'
    for arm, modes in summary['overall'].items():
        for mode, metric in modes.items():
            lines.append(line(f'{arm} / {mode}', metric))
    for shallow, groups in summary['pairs'].items():
        lines += ['', f'## {shallow} vs original36', '',
                  '| Group / arm / mode | n | @1 | @2 | @3 | @5 | @10 | @all |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for group, arms in groups.items():
            for arm, modes in arms.items():
                for mode, metric in modes.items():
                    lines.append(line(f'{group} / {"deep" if arm == DEEP else "shallow"} / {mode}', metric))
        lines += ['', 'Both-wrong FIRST qualifying rank counts (raw; not cumulative):', '',
                  '| Arm | ' + ' | '.join(RANK_BINS) + ' |', '|---|' + '---:|' * len(RANK_BINS)]
        for arm in (shallow, DEEP):
            counts = groups['both_wrong'][arm]['raw']['first_rank_counts']
            lines.append('| ' + arm + ' | ' + ' | '.join(str(counts[b]) for b in RANK_BINS) + ' |')
    lines += ['', '## Geometry-only error flags and duplicate risks', '',
              '| Arm | wrong | IoU [0.45,0.5) | [0.1,0.45) | <0.1 | no qualifying candidate | qualified outside top10 | top5 high overlap | dedup lost all coverage |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, modes in summary['overall'].items():
        r = modes['raw']; g = r['geometry_counts']; a = r['availability_counts']
        values = [r['n']-g['correct'], g['near_threshold_045_050'], g['overlap_010_045'],
                  g['low_overlap_lt010'], a['candidate_miss'], a['outside_top10'],
                  r['top5_high_overlap_queries'], r['dedup_lost_coverage']]
        lines.append('| ' + arm + ' | ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', 'Geometry and availability are separate axes; columns must NOT be summed as one error taxonomy.',
              'Near-threshold IoU does not prove same object; low IoU does not prove semantic error.',
              'Semantic-object mistakes / same-object localization / ambiguous expression or annotation require manual review.',
              'review.csv has BLANK manual labels. review_cases.jsonl contains GT and relevant clipped candidate boxes.',
              'Review sampling is deterministic and stratified; do not estimate population prevalence from this enriched subset.',
              'High Top-k is oracle shortlist headroom, not evidence that a learned verifier can exploit it.',
              'Validation is exploratory; this report does not tune thresholds, fit heads, or claim actual early-exit speedup.', '']
    return '\n'.join(lines)


def select_review(rows: list[dict], shallow_arms: list[str]) -> dict:
    strata = {}
    for row in rows:
        for arm in shallow_arms:
            d = row['arms'][arm]
            tags = [f'{arm}/{row["groups"][arm]}/rank_{d["modes"]["raw"]["first_rank_bin"]}']
            if d['geometry'] == 'near_threshold_045_050':
                tags.append(f'{arm}/near_threshold')
            if d['top5_high_overlap_pairs']:
                tags.append(f'{arm}/top5_high_overlap')
            if d['dedup_lost_coverage']:
                tags.append(f'{arm}/dedup_lost_coverage')
            for tag in tags:
                strata.setdefault(tag, []).append(row['id'])
    selected = {}
    for tag, ids in sorted(strata.items()):
        for sample_id in sorted(ids, key=lambda x: hashlib.sha256((tag+'|'+x).encode()).hexdigest())[:REVIEW_PER_STRATUM]:
            selected.setdefault(sample_id, []).append(tag)
    return selected


def run() -> None:
    assert __debug__, 'Do not disable assertion gates with python -O'
    root = Path(os.environ['PL_OUT']).resolve()
    out = Path(os.environ.get('DIAG_OUT', root / 'topk_diagnostics_v1')).resolve()
    assert root != out and not out.exists(), f'Refusing overwrite: {out}'
    assert DEVICE in ('cuda', 'cpu') and (DEVICE == 'cpu' or torch.cuda.is_available())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    plan = read_json(root / 'plan.json')
    cache = root / 'cache_validation'
    manifest, entries = open_cache(cache, 'validation')
    expected = [r for r in plan['rows'] if r['split'] == 'validation']
    assert manifest['plan_sha256'] == digest(root / 'plan.json')
    assert [(e['id'], e['image_key'], e['split']) for e in entries] == [
        (r['id'], r['image_key'], r['split']) for r in expected]
    assert len(entries) == plan['counts']['validation']
    for entry in entries:
        assert (cache / entry['file']).is_file(), f'Missing feature cache {entry["file"]}; run on server; no fallback'
    trained = read_json(root / 'train/COMPLETE.json')
    assert trained['status'] == 'PASSED' and trained['plan_sha256'] == digest(root / 'plan.json')
    assert trained['protocol_sha256'] == digest(root / 'train/protocol.json')
    assert manifest['frozen_train_complete_sha256'] == digest(root / 'train/COMPLETE.json')
    protocol = read_json(root / 'train/protocol.json')
    depths = plan['depths']
    assert manifest['depths'] == protocol['depths'] == depths and 24 in depths and 30 in depths and 36 in depths
    for name in ('ref_plinear_core.py', 'ref_plinear_data.py', 'humanref_pipeline.py'):
        assert digest(ROOT / 'tools' / name) == protocol['source_sha256'][name], f'Changed input/readout code: {name}'
    heads = []
    for item in trained['checkpoints']:
        path = root / 'train' / item['file']
        assert digest(path) == item['sha256']
        h = torch.load(path, map_location='cpu', weights_only=True)
        assert h['seed'] == item['seed'] and h['depths'] == depths
        assert h['plan_sha256'] == digest(root / 'plan.json') and h['protocol_sha256'] == trained['protocol_sha256']
        assert h['model_signature'] == manifest['model_signature']
        assert h['selected_epochs'] == item['selected_epochs']
        h['weight'], h['bias'] = h['weight'].to(DEVICE), h['bias'].to(DEVICE)
        heads.append(h)
    assert [h['seed'] for h in heads] == protocol['seeds']
    previous = read_json(root / 'evaluation/predictions.json')
    eval_summary = read_json(root / 'evaluation/summary.json')
    assert eval_summary['status'] == 'PASSED' and eval_summary['n'] == len(entries)
    for key, path in [('plan_sha256', root / 'plan.json'), ('cache_complete_sha256', cache / 'COMPLETE.json'),
                      ('train_complete_sha256', root / 'train/COMPLETE.json')]:
        assert eval_summary[key] == digest(path)
    assert [r['id'] for r in previous] == [e['id'] for e in entries]
    shallow = [f'linear_d24_seed{h["seed"]}_raw_fp32' for h in heads] + [NATIVE30]
    arms = shallow + [DEEP]
    out.mkdir(parents=True)
    rows = []
    with torch.inference_mode():
        for x, _, _, _, samples in batches(cache, entries, BATCH_SIZE, DEVICE):
            logits = [readout(x, h['weight'], h['bias']).cpu() for h in heads]
            for i, sample in enumerate(samples):
                old, planned = previous[len(rows)], expected[len(rows)]
                assert sample['query'] == old['query'] == planned['referring'] and sample['image_key'] == old['image_key']
                assert sample['depths'] == depths and sample['id'] == old['id']
                n = sample['features'].shape[1]
                assert sample['native_bf16'].shape == (len(depths), n) and sample['baseline'].shape == (n,)
                assert len(sample['gt']) == 1
                boxes = torch.tensor(sample['boxes'], dtype=torch.float64)
                overlaps = sample['overlaps']
                assert boxes.shape == (n, 4) and n == len(planned['candidate_boxes'])
                recomputed = box_iou(boxes, torch.tensor(sample['gt'], dtype=torch.float64))[:, 0].float()
                assert torch.allclose(recomputed, overlaps, atol=1e-6, rtol=0), 'GT/box/IoU mismatch'
                values = [v[i, depths.index(24), :n] for v in logits] + [
                    sample['native_bf16'][depths.index(30)].float(), sample['baseline'].float()]
                result = forward_algorithm(torch.stack(values), overlaps, boxes)
                for arm, diag in zip(arms, result):
                    before = old['arms'][arm]
                    actual = dict(index=diag['winner'], correct=diag['top1_iou'] >= IOU_THRESHOLD,
                                  iou=diag['top1_iou'], ties=diag['top1_ties'])
                    if actual != before:
                        save_json(out / 'CONSISTENCY_FAILURE.json', dict(id=sample['id'], arm=arm,
                            previous=before, recomputed=actual, device=DEVICE,
                            note='Do not merge changed Top-1 with old groups. Retry original evaluation device; investigate numerical drift.'))
                        raise AssertionError(f'Top-1 mismatch: {sample["id"]}, {arm}; see CONSISTENCY_FAILURE.json')
                assert old['covered'] == result[-1]['modes']['raw']['all_hit']
                mapping = dict(zip(arms, result))
                rows.append(dict(id=sample['id'], image_key=sample['image_key'], image_name=sample['image_name'],
                    image_path=planned['image_path'], query=sample['query'], gt=sample['gt'],
                    groups={a: correctness_group(mapping[a]['top1_iou'] >= .5, mapping[DEEP]['top1_iou'] >= .5) for a in shallow},
                    arms=mapping, boxes=sample['boxes']))
            print(f'Diagnostic {len(rows)}/{len(entries)} PASS', flush=True)
    report = summarize(rows, shallow)
    report.update(status='PASSED', iou_rule='>=0.5', dedup_iou=DEDUP_IOU, k=list(KS)+['all'],
        device=DEVICE, torch=str(torch.__version__), feature_gpu=manifest['gpu'],
        seeds=protocol['seeds'], smoke_only=plan['smoke_only'],
        sources={p.name: digest(p) for p in [Path(__file__), ROOT/'tools/ref_topk_core.py']},
        inputs={str(p.relative_to(root)): digest(p) for p in [root/'plan.json', cache/'COMPLETE.json',
                root/'train/COMPLETE.json', root/'evaluation/predictions.json', root/'evaluation/summary.json']})
    selected = select_review(rows, shallow)
    with (out/'per_expression.jsonl').open('x', encoding='utf-8') as all_file, \
         (out/'review_cases.jsonl').open('x', encoding='utf-8') as review_file:
        for row in rows:
            boxes = row.pop('boxes')
            all_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            if row['id'] in selected:
                indices = {d['best_iou_index'] for d in row['arms'].values()}
                for d in row['arms'].values():
                    for m in d['modes'].values():
                        indices.update(m['top10'])
                        if m['first_qualified_index'] is not None:
                            indices.add(m['first_qualified_index'])
                review_file.write(json.dumps(dict(row, review_strata=selected[row['id']],
                    candidate_boxes={str(j): boxes[j] for j in sorted(indices)}), ensure_ascii=False, allow_nan=False)+'\n')
    with (out/'review.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'image_path', 'query', 'strata', 'arm_under_review', 'semantic_object_error',
                         'same_object_localization', 'expression_or_annotation_ambiguity', 'uncertain', 'notes'])
        for row in rows:
            if row['id'] in selected:
                writer.writerow([row['id'], row['image_path'], row['query'], ';'.join(selected[row['id']])] + ['']*6)
    report['review_n'] = len(selected)
    save_json(out/'summary.json', report)
    (out/'summary.md').write_text(markdown(report), encoding='utf-8')
    save_json(out/'COMPLETE.json', dict(status='PASSED', n=len(rows),
        files={p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(f'Completed: {out / "summary.md"}; review expressions={len(selected)}', flush=True)


if __name__ == '__main__':
    run()
