"""Small one-factor-at-a-time OFFLINE sensitivity study; never selects a winner."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from humanref_pipeline import digest, load_annotations, load_shards, save_json


def experiment_plan():
    # One common anchor, seven unique evaluations; not a Cartesian grid.
    return [
        dict(name='anchor_s035_n050', score=.35, nms=.5, axis='anchor'),
        dict(name='score_s025_n050', score=.25, nms=.5, axis='score'),
        dict(name='score_s045_n050', score=.45, nms=.5, axis='score'),
        dict(name='nms_s035_none', score=.35, nms=None, axis='nms'),
        dict(name='nms_s035_n040', score=.35, nms=.4, axis='nms'),
        dict(name='nms_s035_n060', score=.35, nms=.6, axis='nms'),
        dict(name='nms_s035_n070', score=.35, nms=.7, axis='nms'),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--ref-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--plan-only', action='store_true', help='Print plan; no files, data or dependencies required')
    args = parser.parse_args()
    plan = experiment_plan()
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    output = Path(args.output)
    assert not output.exists(), 'Choose a new output directory'
    rows = load_annotations(args.annotations, args.limit)
    refs, meta, hashes = load_shards(args.ref_dir, 'ref')
    assert meta['annotation_sha256'] == digest(args.annotations)
    assert meta['ids'] == [r['id'] for r in rows] and set(refs) == {r['id'] for r in rows}
    save_json(output/'plan.json', dict(status='exploratory_HumanRef_not_independent_validation',
              selection_rule='none; report every setting', runs=plan,
              ref_meta=meta, ref_shard_sha256=hashes))
    script = Path(__file__).with_name('humanref_pipeline.py')
    reports = []
    for setting in plan:
        target = output/setting['name']
        cmd = [sys.executable, '-B', str(script), 'analyze', '--annotations', args.annotations,
               '--ref-dir', args.ref_dir, '--output', str(target), '--limit', str(args.limit),
               '--score-threshold', str(setting['score'])]
        if setting['nms'] is not None:
            cmd += ['--nms-iou', str(setting['nms'])]
        print('Running ' + setting['name'], flush=True)
        with (output/(setting['name']+'.log')).open('x', encoding='utf-8') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        official = json.loads((target/'official_summary.json').read_text(encoding='utf-8'))
        diagnostic = json.loads((target/'diagnostics.json').read_text(encoding='utf-8'))['overall']
        reports.append(dict(**setting, **official, **{k: diagnostic[k] for k in (
            'target_coverage', 'ref_missed_covered_targets', 'one_to_one_tp', 'one_to_one_fp',
            'nonoverlap_fp', 'duplicate_or_assignment_fp', 'nms_removed')}))
    save_json(output/'summary.json', reports)
    with (output/'summary.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    columns = ['axis', 'score', 'nms', 'P50', 'R50', 'DF150', 'P5095', 'R5095',
               'DF15095', 'Rejection', 'one_to_one_fp', 'duplicate_or_assignment_fp']
    lines = ['Exploratory sensitivity only. Same cached inference; no best-setting selection.', '',
             '| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for report in reports:
        values = [report[k] for k in columns]
        lines.append('| '+' | '.join('off' if v is None and k == 'nms' else
                                    'N/A' if v is None else f'{v:.6g}' if isinstance(v, float)
                                    else str(v) for k, v in zip(columns, values))+' |')
    (output/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Completed all seven settings: {output / "summary.md"}')


if __name__ == '__main__':
    main()
