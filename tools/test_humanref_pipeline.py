"""CPU-only regression tests; no model weights or third-party packages needed."""
import json
import ast
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from humanref_pipeline import (aggregate, diagnose, digest, load_shards, save_json,
                              select_indices, matching_counts, check_comparable, dataset_proposals)


def main():
    from humanref_sensitivity import experiment_plan
    plan = experiment_plan()
    assert len(plan) == len({(r['score'], r['nms']) for r in plan}) == 7
    assert sum(r['axis'] == 'anchor' for r in plan) == 1
    assert all(r['nms'] == .5 for r in plan if r['axis'] == 'score')
    assert all(r['score'] == .35 for r in plan if r['axis'] == 'nms')
    a, b = [0, 0, 10, 10], [20, 20, 30, 30]
    assert select_indices([a, a, b], [.6, .9, .8], .35, .7) == [1, 2]
    assert select_indices([a, a], [.9, .9], .35, .7) == [0]
    assert select_indices([], [], .35, .7) == []
    assert matching_counts([a], [a, a, b], .5) == (1, 1, 1)
    assert matching_counts([a, b], [a, b], .5) == (2, 0, 0)
    # Execute only the official pure metric functions, avoiding GPU dependencies.
    source = Path(__file__).resolve().parents[1] / 'wedetect_ref/eval_grounding/recall_precision_densityf1.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('calculate_iou', 'calculate_metrics')]
    official = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), official)
    for predictions in ([], [a], [a, a, b], [b, a], [b, b]):
        recall, precision = official['calculate_metrics']([a, b], predictions, .5)
        tp, _, _ = matching_counts([a, b], predictions, .5)
        assert recall == tp/2
        assert precision == (tp/len(predictions) if predictions else 0)
    for bad_boxes, bad_scores in (([[0, 0, 1]], [.5]), ([a], [float('nan')])):
        try:
            select_indices(bad_boxes, bad_scores, .35, .7)
            raise RuntimeError('Invalid prediction accepted')
        except AssertionError:
            pass
    assert dataset_proposals([dict(image_name='x', candidate_boxes=[a]),
                              dict(image_name='x', candidate_boxes=[b])])['x']['boxes'] == [b]
    cases = [
        ([], [], [], 'correct_rejection'),
        ([], [a], [.9], 'false_positive_rejection'),
        ([a], [b], [.9], 'no_target_covered'),
        ([a, b], [a], [.9], 'partial_target_coverage'),
        ([a], [a], [.2], 'covered_but_ref_missed'),
        ([a], [a, b], [.9, .8], 'all_targets_recovered'),
    ]
    rows = []
    for gt, boxes, scores, expected in cases:
        d = diagnose(gt, boxes, scores, .35, .5)
        assert d['category'] == expected, d
        assert d['gt_count'] - d['recovered_targets'] == d['proposal_missed_targets'] + d['ref_missed_covered_targets']
        assert d['selected_count'] == d['one_to_one_tp'] + d['one_to_one_fp']
        assert d['one_to_one_fp'] == d['nonoverlap_fp'] + d['duplicate_or_assignment_fp']
        rows.append(d)
    assert rows[-1]['unmatched_selected_boxes'] == 1  # recovered != error-free
    s = aggregate(rows)
    assert s['rejection_accuracy'] == .5
    assert s['gt_targets'] == 5 and s['covered_targets'] == 3 and s['recovered_targets'] == 2
    assert diagnose([a], [a], [.35], .35, .5)['selected_count'] == 0
    assert diagnose([a], [], [], .35, .5)['category'] == 'no_target_covered'
    assert aggregate([])['target_coverage'] is None
    try:
        diagnose([a], [a], [], .35, .5)
        raise RuntimeError('Missing scores were accepted')
    except AssertionError:
        pass
    rng = random.Random(7)
    for _ in range(100):
        scores = [rng.random(), rng.random()]
        lo = diagnose([a, b], [a, b], scores, .2, .5)
        hi = diagnose([a, b], [a, b], scores, .8, .5)
        assert lo['recovered_targets'] >= hi['recovered_targets']
        assert lo['covered_targets'] == hi['covered_targets'] == 2
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        annotations = [dict(id=i, image_name=f'{i}.jpg', referring='person',
                            domain='rejection' if not gt else 'attribute', sub_domain='test',
                            answer_boxes=gt, candidate_boxes=[a, b])
                       for i, (gt, _, _, _) in enumerate(cases)]
        ann = root / 'annotations.jsonl'
        ann.write_text('\n'.join(json.dumps(r) for r in annotations), encoding='utf-8')
        meta = dict(annotation_sha256=digest(ann), ids=list(range(len(cases))), num_proposals=100)
        meta.update(checkpoint='fixture', checkpoint_files_sha256={'weights':'fixture'},
                    attention='test', coordinate_policy='fp32_output_bf16_model_input',
                    prompt='test', script_sha256='fixture', torch_version='test', candidate_source='uni')
        records = [dict(id=i, image_name=f'{i}.jpg', boxes=boxes, scores=scores, seconds=.1)
                   for i, (_, boxes, scores, _) in enumerate(cases)]
        shards = root / 'ref'
        for rank in range(2):
            save_json(shards / f'ref.rank{rank:03d}.json', dict(meta=meta, rank=rank, world_size=2, records=records[rank::2]))
        loaded, _, _ = load_shards(shards, 'ref')
        assert set(loaded) == set(range(len(cases)))
        out = root / 'analysis'
        subprocess.run([sys.executable, str(Path(__file__).with_name('humanref_pipeline.py')),
                        'analyze', '--annotations', str(ann), '--ref-dir', str(shards),
                        '--output', str(out), '--skip-official-metrics'], check=True, capture_output=True)
        report = json.loads((out / 'diagnostics.json').read_text(encoding='utf-8'))
        assert report['overall']['gt_targets'] == 5
        assert len((out / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()) == 6
        a_meta = dict(meta, candidate_source='dataset')
        check_comparable(a_meta, meta)
        try:
            check_comparable(dict(a_meta, num_proposals=20), meta)
            raise RuntimeError('Mixed proposal caps accepted')
        except AssertionError:
            pass
        a_dir = root / 'a'
        save_json(a_dir / 'ref.rank000.json', dict(meta=a_meta, rank=0, world_size=1, records=records))
        abc = root / 'abc'
        subprocess.run([sys.executable, str(Path(__file__).with_name('humanref_pipeline.py')),
                        'compare', '--annotations', str(ann), '--ref-dir', str(shards),
                        '--a-ref-dir', str(a_dir), '--output', str(abc), '--skip-official-metrics'],
                       check=True, capture_output=True)
        bp = json.loads((abc/'B/provenance.json').read_text(encoding='utf-8'))
        cp = json.loads((abc/'C/provenance.json').read_text(encoding='utf-8'))
        assert bp == cp
        assert (abc/'comparison.md').is_file()
        (shards / 'ref.rank001.json').unlink()
        try:
            load_shards(shards, 'ref')
            raise RuntimeError('Missing shard was accepted')
        except AssertionError:
            pass
    print('PASS: NMS, matching conservation, dataset source, A/B/C provenance, error categories, randomized thresholds, shard validation, offline CLI integration')


if __name__ == '__main__':
    main()
