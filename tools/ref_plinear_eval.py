"""Paired validation of fixed P-linear checkpoints; never chooses a winning seed/depth."""
import os
from pathlib import Path
import statistics

from humanref_pipeline import digest, save_json
from ref_plinear import DEPTHS, OUTPUT, source_hashes
from ref_plinear_data import batches, open_cache, read_json

DEVICE = os.environ.get('PL_DEVICE', 'cuda')
BATCH_SIZE = 16


def decision(values, overlaps):
    assert values.ndim == overlaps.ndim == 1 and values.shape == overlaps.shape
    assert values.numel() > 0
    assert values.isfinite().all() and overlaps.isfinite().all()
    assert ((overlaps >= 0) & (overlaps <= 1)).all()
    winner = int(values.argmax())
    return dict(index=winner, iou=float(overlaps[winner]), correct=bool(overlaps[winner] >= .5),
                ties=int((values == values[winner]).sum()))


def paired_summary(records, arm):
    actual = [r['arms'][arm] for r in records]
    base = [r['arms']['original_full36_raw_bf16'] for r in records]
    n = len(records)
    assert n > 0
    hits = sum(x['correct'] for x in actual)
    covered = sum(r['covered'] for r in records)
    return dict(arm=arm, n=n, correct=hits, accuracy=hits / n,
        delta_pp=(hits - sum(b['correct'] for b in base)) * 100 / n,
        harmed=sum(b['correct'] and not a['correct'] for b, a in zip(base, actual)),
        recovered=sum(not b['correct'] and a['correct'] for b, a in zip(base, actual)),
        changed_winner=sum(b['index'] != a['index'] for b, a in zip(base, actual)),
        top1_tied=sum(a['ties'] > 1 for a in actual), mean_iou=statistics.mean(a['iou'] for a in actual),
        covered_n=covered, covered_accuracy=hits / covered if covered else None)


def run():
    import torch
    from ref_plinear_core import forward_algorithm
    assert __debug__ and int(os.environ.get('WORLD_SIZE', '1')) == 1
    assert DEVICE == 'cpu' or (DEVICE == 'cuda' and torch.cuda.is_available())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root, train, out = OUTPUT / 'cache_validation', OUTPUT / 'train', OUTPUT / 'evaluation'
    assert not out.exists(), f'Refusing to overwrite {out}'
    plan = read_json(OUTPUT / 'plan.json')
    manifest, entries = open_cache(root, 'validation')
    assert manifest['plan_sha256'] == digest(OUTPUT / 'plan.json')
    assert manifest['model_signature']['sources'] == source_hashes() == plan['sources']
    expected = [r for r in plan['rows'] if r['split'] == 'validation']
    assert [(e['id'], e['split'], e['image_key']) for e in entries] == [
        (r['id'], r['split'], r['image_key']) for r in expected]
    complete = read_json(train / 'COMPLETE.json')
    assert complete['status'] == 'PASSED' and complete['plan_sha256'] == digest(OUTPUT / 'plan.json')
    assert manifest['frozen_train_complete_sha256'] == digest(train / 'COMPLETE.json'), 'Heads changed after validation extraction'
    assert complete['protocol_sha256'] == digest(train / 'protocol.json')
    protocol = read_json(train / 'protocol.json')
    fit, _ = open_cache(OUTPUT / 'cache_fit', 'fit')
    assert manifest['model_signature'] == fit['model_signature']
    assert protocol['cache_complete_sha256'] == digest(OUTPUT / 'cache_fit/COMPLETE.json')
    heads = []
    for entry in complete['checkpoints']:
        path = train / entry['file']
        assert digest(path) == entry['sha256']
        h = torch.load(path, map_location='cpu', weights_only=True)
        assert h['depths'] == DEPTHS and h['seed'] == entry['seed']
        assert h['protocol_sha256'] == digest(train / 'protocol.json')
        assert h['plan_sha256'] == digest(OUTPUT / 'plan.json')
        assert h['model_signature'] == manifest['model_signature']
        h['weight'], h['bias'] = h['weight'].to(DEVICE), h['bias'].to(DEVICE)
        heads.append(h)
    assert [h['seed'] for h in heads] == protocol['seeds']
    records = []
    with torch.inference_mode():
        for x, _, _, _, samples in batches(root, entries, BATCH_SIZE, DEVICE):
            outputs = {h['seed']: forward_algorithm(x, h['weight'], h['bias']).cpu() for h in heads}
            for i, sample in enumerate(samples):
                overlaps = sample['overlaps']
                n = len(overlaps)
                arms = dict(original_full36_raw_bf16=decision(sample['baseline'], overlaps),
                            original_full36_sigmoid_bf16=decision(sample['baseline'].sigmoid(), overlaps))
                for k, depth in enumerate(DEPTHS):
                    arms[f'native_d{depth}_raw_bf16'] = decision(sample['native_bf16'][k], overlaps)
                    arms[f'native_d{depth}_sigmoid_bf16'] = decision(sample['native_bf16'][k].sigmoid(), overlaps)
                    arms[f'native_d{depth}_raw_fp32'] = decision(sample['native_fp32'][k], overlaps)
                    for h in heads:
                        arms[f'linear_d{depth}_seed{h["seed"]}_raw_fp32'] = decision(outputs[h['seed']][i, k, :n], overlaps)
                records.append(dict(id=sample['id'], image_key=sample['image_key'], query=sample['query'],
                                    covered=bool((overlaps >= .5).any()), arms=arms))
    assert len(records) == len(entries)
    curve = [paired_summary(records, name) for name in records[0]['arms']]
    means = []
    for depth in DEPTHS:
        values = [next(r['accuracy'] for r in curve if r['arm'] == f'linear_d{depth}_seed{h["seed"]}_raw_fp32') for h in heads]
        means.append(dict(depth=depth, accuracy_mean=statistics.mean(values),
                          seed_sample_std=statistics.stdev(values) if len(values) > 1 else None))
    save_json(out / 'predictions.json', records)
    save_json(out / 'summary.json', dict(status='PASSED', n=len(records),
        images=len({r['image_key'] for r in records}), smoke_only=plan['smoke_only'], curve=curve,
        trained_seed_summary=means, depths=DEPTHS, seeds=[h['seed'] for h in heads],
        plan_sha256=digest(OUTPUT / 'plan.json'), cache_complete_sha256=digest(root / 'COMPLETE.json'),
        train_complete_sha256=digest(train / 'COMPLETE.json'),
        evaluation_source_sha256=digest(__file__), feature_gpu=manifest['gpu'],
        fit_feature_gpu=fit['gpu'], device=DEVICE, torch=str(torch.__version__),
        note='Linear readability only. All 36 blocks executed when extracting; no speedup claim. '
             'Validation already used in method development; not an untouched test set. Seed SD is not a confidence interval.'))
    with (out / 'summary.md').open('x', encoding='utf-8') as stream:
        stream.write(f'# P-linear: {len(records)} validation expressions\n\n')
        stream.write(f'Smoke only: {plan["smoke_only"]}. Ranking = raw logits unless explicitly sigmoid.\n\n')
        stream.write('| Arm | Correct | Acc % | Delta pp vs original raw | Harmed | Recovered |\n|---|---:|---:|---:|---:|---:|\n')
        for r in curve:
            stream.write(f'| {r["arm"]} | {r["correct"]} | {100*r["accuracy"]:.3f} | {r["delta_pp"]:+.3f} | {r["harmed"]} | {r["recovered"]} |\n')
        stream.write('\nAll seeds and depths reported; do not select the best validation seed. '
                     'Compare linear shallow vs linear36 AND original36; native FP32 is the precision-only control. '
                     'See predictions.json for image-paired analyses. No latency or actual trained-exit claim.\n')
    print(f'Completed: {out / "summary.md"}', flush=True)


if __name__ == '__main__':
    run()
