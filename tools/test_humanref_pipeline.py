"""CPU-only regression tests; no model weights or third-party packages needed."""
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from humanref_pipeline import aggregate, diagnose, digest, load_shards, save_json


def main():
    a, b = [0, 0, 10, 10], [20, 20, 30, 30]
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
        (shards / 'ref.rank001.json').unlink()
        try:
            load_shards(shards, 'ref')
            raise RuntimeError('Missing shard was accepted')
        except AssertionError:
            pass
    print('PASS: six error categories, threshold boundaries, randomized monotonicity, missing-score rejection, shard validation, offline CLI integration')


if __name__ == '__main__':
    main()
