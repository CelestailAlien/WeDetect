"""CPU synthetic tests; synthetic inputs never enter the real report as fallbacks."""
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from humanref_pipeline import digest, save_json
from ref_plinear_core import forward_algorithm as readout
from ref_topk_core import (KS, box_iou, correctness_group, forward_algorithm)
import ref_topk_report as report


def must_fail(fn, kinds=(AssertionError,)):
    try:
        fn()
    except kinds:
        return
    raise AssertionError('Expected invalid input to fail')


def test_core():
    torch.manual_seed(87)
    xy = torch.rand(23, 2)
    boxes = torch.cat([xy, xy + torch.rand(23, 2)], 1)
    scores, overlaps = torch.randn(5, 23), torch.rand(23)
    result = forward_algorithm(scores, overlaps, boxes)
    assert len(result) == 5 and all(len(d['modes']['raw']['top10']) == 10 for d in result)
    for a, d in enumerate(result):
        order = sorted(range(23), key=lambda j: (-float(scores[a, j]), j))
        for k in KS:
            assert d['modes']['raw']['hits'][str(k)] == any(float(overlaps[j]) >= .5 for j in order[:k])
        assert d['winner'] == int(scores[a].argmax())
    # Exact 0.5 is positive; ties prefer the original lower candidate index.
    small = torch.tensor([[0., 0, 1, 1], [0., 0, 1, 1], [3., 3, 4, 4]])
    edge = forward_algorithm(torch.tensor([[2., 2., 1.]]), torch.tensor([.49, .5, .1]), small)[0]
    assert edge['winner'] == 0 and edge['top1_ties'] == 2 and edge['cutoff_ties']['1']
    assert edge['modes']['raw']['first_qualified_rank'] == 2
    assert not edge['modes']['raw']['hits']['1'] and edge['modes']['raw']['hits']['2']
    assert edge['modes']['raw']['hits']['10']  # k>N uses all candidates.
    assert edge['dedup_lost_coverage'] and edge['suppressed_qualified_by_unqualified'] == 1
    assert edge['geometry'] == 'near_threshold_045_050'
    # Use REAL geometry to show an unqualified box suppressing a qualified box.
    b = torch.tensor([[0., 0, .49, 1], [0., 0, .5, 1]], dtype=torch.float64)
    actual_ious = box_iou(b, torch.tensor([[0., 0, 1, 1]]))[:, 0].float()
    d = forward_algorithm(torch.tensor([[2., 1.]]), actual_ious, b)[0]
    assert d['dedup_lost_coverage'] and d['modes']['raw']['first_qualified_rank'] == 2
    # Dedup can also expose a different candidate previously crowded out.
    d = forward_algorithm(torch.tensor([[3., 2., 1.]]), torch.tensor([0., 0., .8]), small)[0]
    assert d['modes']['raw']['first_qualified_rank'] == 3
    assert d['modes']['dedup']['first_qualified_rank'] == 2
    absent = forward_algorithm(torch.ones(1, 3), torch.zeros(3), small)[0]
    assert absent['availability'] == 'candidate_miss' and not absent['modes']['raw']['all_hit']
    assert absent['modes']['raw']['first_qualified_rank'] is None
    assert correctness_group(True, True) == 'both_correct'
    assert correctness_group(False, True) == 'deep_only'
    assert correctness_group(True, False) == 'shallow_only'
    assert correctness_group(False, False) == 'both_wrong'
    must_fail(lambda: forward_algorithm(scores, overlaps[:-1], boxes))
    must_fail(lambda: forward_algorithm(scores * float('nan'), overlaps, boxes))
    must_fail(lambda: forward_algorithm(scores, overlaps + 2, boxes))
    must_fail(lambda: forward_algorithm(scores, overlaps, boxes[:, :3]))
    must_fail(lambda: forward_algorithm(scores, overlaps, boxes[:, [2, 3, 0, 1]]))
    # Empty subgroup is undefined (None), not artificially zero recall.
    empty = report.aggregate([], 'unused', 'raw')
    assert empty['n'] == 0 and empty['rates']['1'] is None
    # Exact rank buckets cover high ranks as well as 2/3 and candidate misses.
    wide = torch.cat([torch.arange(60.)[:, None], torch.zeros(60, 1)], 1)
    wide = torch.cat([wide, wide + 1], 1)
    for rank, bucket in [(1, '1'), (2, '2'), (3, '3'), (5, '4-5'), (10, '6-10'),
                         (20, '11-20'), (50, '21-50'), (60, '51+')]:
        positive = torch.zeros(60); positive[rank-1] = .5
        d = forward_algorithm(-torch.arange(60.)[None], positive, wide)[0]
        assert d['modes']['raw']['first_qualified_rank'] == rank
        assert d['modes']['raw']['first_rank_bin'] == bucket


def make_fixture(root):
    """17 rows: exercises a full batch, padding and a final singleton batch."""
    torch.manual_seed(19)
    depths, seeds = [9, 18, 24, 30, 36], [42, 43, 44]
    cache, train, evaluation = root/'cache_validation', root/'train', root/'evaluation'
    (cache/'samples').mkdir(parents=True); train.mkdir(); evaluation.mkdir()
    rows, samples, entries = [], [], []
    for i in range(17):
        n = 3 + i % 4
        boxes = [[float(j), 0., float(j+1), 1.] for j in range(n)]
        gt = [boxes[i % n]]
        row = dict(id=f'synthetic_{i}', image_key=i, image_name=f'{i}.jpg',
                   image_path=f'/synthetic/{i}.jpg', referring='SYNTHETIC TEST ONLY',
                   answer_boxes=gt, candidate_boxes=boxes, split='validation')
        rows.append(row)
        ov = box_iou(torch.tensor(boxes), torch.tensor(gt))[:, 0].float()
        samples.append(dict(id=row['id'], image_key=i, image_name=row['image_name'], query=row['referring'],
            split='validation', depths=depths, boxes=boxes, gt=gt, features=torch.randn(5, n, 8).bfloat16(),
            overlaps=ov, labels=torch.where(ov > .5, ov, 0),
            native_bf16=torch.randn(5, n).bfloat16(), baseline=torch.randn(n).bfloat16()))
    save_json(root/'plan.json', dict(rows=rows, depths=depths, counts=dict(validation=17), smoke_only=True))
    protocol = dict(depths=depths, seeds=seeds, source_sha256={name: digest(report.ROOT/'tools'/name)
        for name in ('ref_plinear_core.py', 'ref_plinear_data.py', 'humanref_pipeline.py')})
    save_json(train/'protocol.json', protocol)
    heads, checkpoints = [], []
    for seed in seeds:
        h = dict(seed=seed, depths=depths, weight=torch.randn(5, 8), bias=torch.randn(5),
                 plan_sha256=digest(root/'plan.json'), protocol_sha256=digest(train/'protocol.json'),
                 model_signature=dict(synthetic=True), selected_epochs=[1]*5)
        path = train/f'heads_seed{seed}.pt'; torch.save(h, path); heads.append(h)
        checkpoints.append(dict(file=path.name, sha256=digest(path), seed=seed, selected_epochs=[1]*5))
    save_json(train/'COMPLETE.json', dict(status='PASSED', checkpoints=checkpoints,
        plan_sha256=digest(root/'plan.json'), protocol_sha256=digest(train/'protocol.json')))
    for i, s in enumerate(samples):
        path = cache/'samples'/f'{i:05d}.pt'; torch.save(s, path)
        entries.append(dict(id=s['id'], image_key=s['image_key'], split='validation',
                            file=f'samples/{i:05d}.pt', sha256=digest(path)))
    save_json(cache/'index.json', entries)
    save_json(cache/'manifest.json', dict(sample_ids=[r['id'] for r in rows], depths=depths,
        plan_sha256=digest(root/'plan.json'), frozen_train_complete_sha256=digest(train/'COMPLETE.json'),
        model_signature=dict(synthetic=True), gpu='SYNTHETIC CPU'))
    save_json(cache/'COMPLETE.json', dict(status='PASSED', stage='validation', samples=len(rows),
        manifest_sha256=digest(cache/'manifest.json'), index_sha256=digest(cache/'index.json')))
    previous = []
    for x, _, _, _, batch in report.batches(cache, entries, 16, 'cpu'):
        outputs = [readout(x, h['weight'], h['bias']) for h in heads]
        for i, s in enumerate(batch):
            names = [f'linear_d24_seed{seed}_raw_fp32' for seed in seeds] + [report.NATIVE30, report.DEEP]
            n = len(s['overlaps'])
            vals = [v[i, 2, :n] for v in outputs] + [s['native_bf16'][3], s['baseline']]
            arms = {}
            for name, v in zip(names, vals):
                j = int(v.argmax()); ov = float(s['overlaps'][j])
                arms[name] = dict(index=j, correct=ov >= .5, iou=ov, ties=int((v == v[j]).sum()))
            previous.append(dict(id=s['id'], image_key=s['image_key'], query=s['query'],
                                 covered=bool((s['overlaps'] >= .5).any()), arms=arms))
    save_json(evaluation/'predictions.json', previous)
    save_json(evaluation/'summary.json', dict(status='PASSED', n=17, plan_sha256=digest(root/'plan.json'),
        cache_complete_sha256=digest(cache/'COMPLETE.json'), train_complete_sha256=digest(train/'COMPLETE.json')))


def test_integration():
    with tempfile.TemporaryDirectory(prefix='ref-topk-test-') as tmp, patch.object(report, 'DEVICE', 'cpu'):
        root = Path(tmp); make_fixture(root)
        with patch.dict(os.environ, PL_OUT=str(root), DIAG_OUT=str(root/'report')):
            report.run()
            summary = report.read_json(root/'report/summary.json')
            assert summary['status'] == 'PASSED' and summary['n'] == 17 and len(summary['pairs']) == 4
            previous = report.read_json(root/'evaluation/predictions.json')
            for arm, modes in summary['overall'].items():
                assert modes['raw']['hits']['1'] == sum(r['arms'][arm]['correct'] for r in previous)
                assert modes['raw']['hits']['all'] == 17
            for arm, groups in summary['pairs'].items():
                for group, metrics in groups.items():
                    actual_n = sum(correctness_group(r['arms'][arm]['correct'],
                                   r['arms'][report.DEEP]['correct']) == group for r in previous)
                    assert metrics[arm]['raw']['n'] == actual_n
                    if group == 'both_wrong':
                        assert metrics[arm]['raw']['hits']['1'] == 0
                        assert metrics[report.DEEP]['raw']['hits']['1'] == 0
            assert len((root/'report/per_expression.jsonl').read_text().splitlines()) == 17
            assert (root/'report/COMPLETE.json').is_file()
            must_fail(report.run)  # existing output must not be overwritten
        previous[0]['arms'][report.DEEP]['index'] = 999
        (root/'evaluation/predictions.json').write_text(json.dumps(previous), encoding='utf-8')
        with patch.dict(os.environ, PL_OUT=str(root), DIAG_OUT=str(root/'mismatch')):
            must_fail(report.run)
            assert (root/'mismatch/CONSISTENCY_FAILURE.json').is_file()
            assert not (root/'mismatch/COMPLETE.json').exists()
        entries = report.read_json(root/'cache_validation/index.json')
        entries[0]['file'] = 'samples/missing.pt'
        (root/'cache_validation/index.json').write_text(json.dumps(entries), encoding='utf-8')
        complete = report.read_json(root/'cache_validation/COMPLETE.json')
        complete['index_sha256'] = digest(root/'cache_validation/index.json')
        (root/'cache_validation/COMPLETE.json').write_text(json.dumps(complete), encoding='utf-8')
        with patch.dict(os.environ, PL_OUT=str(root), DIAG_OUT=str(root/'missing')):
            must_fail(report.run)
            assert not (root/'missing').exists()


if __name__ == '__main__':
    assert __debug__
    torch.set_num_threads(2)
    test_core()
    print('Random shapes / threshold equality / ties / rank bins / dedup risks / invalid inputs: PASS')
    test_integration()
    print('Real report pipeline / batched readout / four groups / empty groups / no overwrite / missing cache / drift gates: PASS')
