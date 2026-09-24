"""Synthetic, offline tests for SADS post-hoc inference and ledger accounting."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import ref_sads_analysis as analysis_module

from ref_sads_io import (ROOT, SOURCE_FILES, digest, object_hash, read_json, read_jsonl,
                         require_stage, source_digest, write_json, write_jsonl, write_stage)

from ref_sads_analysis import (
    _effect_summary, _interval, analyze_run, candidate_ious, focal_loss_fp32,
    image_bootstrap_weights, raw_top1, render_markdown, report_run, weighted_auc, weighted_spearman,
)


def fixture(no_op=False, one_image=False):
    """Four outcome-independent synthetic inputs; 25 logical arms per input."""
    config = dict(layers=[28, 32, 36], random_seeds=[11, 29, 47], gates=[0, .5],
                  candidate_count=2,
                  counts=dict(calibration=1, evaluation=4),
                  analysis=dict(bootstrap_repeats=80, bootstrap_seed=4))
    boxes = [[0, 0, 10, 10], [10, 10, 20, 20]]
    inputs = [dict(id='cal', image_key=99, split='calibration', query='calibration',
                   image_name='cal.jpg', image_sha256='image-99', width=20, height=20,
                   candidate_boxes=boxes, candidate_sha256=object_hash(boxes))]
    predictions, selections, targets, resources = [], [], [], []
    for i in range(4):
        sid, image = f'e{i}', 10 if one_image else 10+i
        inp = dict(id=sid, image_key=image, split='evaluation', query='target',
                   image_name=f'{image}.jpg', image_sha256=f'image-{image}', width=20, height=20,
                   candidate_boxes=boxes, candidate_sha256=object_hash(boxes))
        inputs.append(inp)
        targets.append(dict(id=sid, answer_boxes=[[0, 0, 10, 10]]))
        baseline = [-1., 1.] if i % 2 == 0 else [1., -1.]
        baseline_pid = f'{sid}-base'
        def physical(pid, collector):
            resources.append(dict(physical_forward_id=pid, phase='collect_evaluation' if collector else 'intervene',
                                  collector=collector, backend='flash_attention_2', forward_ms=2 if collector else 1,
                                  request_ms=3 if collector else 2, peak_allocated_bytes=100,
                                  peak_reserved_bytes=200))
        physical(baseline_pid, True)
        predictions.append(dict(id=sid, image_key=image, arm_id='baseline', kind='baseline', layer=None,
                                gate=1, seed=None, head=None, k=0, logits=baseline,
                                prediction_index=raw_top1(baseline), physical_forward_id=baseline_pid,
                                input_sha256=f'in-{sid}'))
        for layer in config['layers']:
            active = not no_op and i != 3
            stats = [dict(head=h, x=.01 if h == 1 else .1, H=.2, e=.2 if h == 1 else .8,
                          valid=True, category='sinkS' if h == 1 else 'vision',
                          score=1. if h == 1 else -float(h)) for h in range(32)]
            selection = dict(id=sid, image_key=image, layer=layer, eligible_heads=[1] if active else [],
                             k=int(active), sink_head=1 if active else None,
                             random_heads={str(seed): head if active else None
                                           for seed, head in zip(config['random_seeds'], [1, 2, 3])},
                             head_statistics=stats, reason='test_active' if active else 'no_sinkS')
            selections.append(selection)
            for gate in config['gates']:
                physical_rows = {}
                for kind, seed, head in [('sinkS', None, 1), ('random', 11, 1), ('random', 29, 2), ('random', 47, 3)]:
                    if not active:
                        logits, pid, actual_head = baseline, baseline_pid, None
                    else:
                        actual_head = head
                        if head == 1:
                            # A beneficial suppression, with varying continuous outcome.
                            shift = (1-gate) * (2.5+i*.1)
                        elif head == 2:
                            shift = -(1-gate)*2.5
                        else:
                            shift = 0.
                        logits = [baseline[0]+shift, baseline[1]-shift]
                        pid = f'{sid}-{layer}-{gate}-{head}'
                        if head not in physical_rows:
                            physical(pid, False)
                            physical_rows[head] = pid
                    arm = f'{kind}_l{layer}_g{gate}' + (f'_s{seed}' if seed is not None else '')
                    predictions.append(dict(id=sid, image_key=image, arm_id=arm, kind=kind, layer=layer,
                                            gate=gate, seed=seed, head=actual_head, k=int(active), logits=logits,
                                            prediction_index=raw_top1(logits), physical_forward_id=pid,
                                            input_sha256=f'in-{sid}'))
    consistency = dict(passed=True, eps=1e-6)
    return [inputs, predictions, selections, targets, config, consistency, resources]


def write_cli_fixture(run, case, reference=False):
    """Publish synthetic stage artifacts using the production exclusive I/O API.

    No checkpoint, image, model, real data or runtime class is touched. Markers
    exercise the actual dependency/hash validation used by the report CLI.
    """
    inputs, predictions, selections, targets, config, consistency, resources = copy.deepcopy(case)
    config['attention_backend'] = 'sdpa' if reference else 'flash_attention_2'
    for record in predictions:
        record['backend'] = config['attention_backend']
    for resource in resources:
        resource['backend'] = config['attention_backend']
    if reference:
        for base in list(predictions):
            if base['kind'] != 'baseline':
                continue
            ref = dict(base, kind='reference', arm_id='reference_fa2', backend='flash_attention_2',
                       physical_forward_id=base['physical_forward_id']+'-reference')
            resource = next(r for r in resources if r['physical_forward_id'] == base['physical_forward_id'])
            resources.append(dict(resource, physical_forward_id=ref['physical_forward_id'],
                                  phase='reference', collector=False, backend='flash_attention_2'))
            predictions.append(ref)
    manifest = dict(schema_version=1, synthetic_offline_test=True,
                    config_sha256=object_hash(config),
                    sources={name: dict(raw_sha256=digest(ROOT / name), lf_sha256=source_digest(ROOT / name))
                             for name in SOURCE_FILES})
    write_json(run / 'manifest.json', manifest)
    write_json(run / 'config.json', config)
    write_jsonl(run / 'inputs.jsonl', inputs)
    write_jsonl(run / 'eval_targets.jsonl', targets)
    write_stage(run, 'prepare', ['config.json', 'inputs.jsonl', 'eval_targets.jsonl'],
                metadata={'synthetic_offline_test': True, 'forward_count': 0})
    write_json(run / 'checks/consistency.json', consistency)
    write_stage(run, 'check', ['checks/consistency.json'], parents=['prepare'])
    write_json(run / 'fixtures/collect_calibration.json', {'synthetic_offline_test': True})
    write_stage(run, 'collect_calibration', ['fixtures/collect_calibration.json'], parents=['check'])
    write_json(run / 'calibration.json', {'synthetic_offline_test': True})
    write_stage(run, 'calibrate', ['calibration.json'], parents=['collect_calibration'])
    write_jsonl(run / 'baseline/evaluation.jsonl', [r for r in predictions if r['kind'] == 'baseline'])
    write_stage(run, 'collect_evaluation', ['baseline/evaluation.jsonl'], parents=['check', 'calibrate'])
    write_jsonl(run / 'selections.jsonl', selections)
    write_stage(run, 'select', ['selections.jsonl'], parents=['collect_evaluation', 'calibrate'])
    write_jsonl(run / 'predictions.jsonl', predictions)
    write_jsonl(run / 'resources.jsonl', resources)
    write_stage(run, 'intervene', ['predictions.jsonl', 'resources.jsonl'], parents=['select'])
    for index, (command, split, seconds, success) in enumerate([
            ('prepare', None, 2., True), ('collect', 'calibration', 3., True), ('check', None, .5, False)]):
        write_json(run / 'timings' / f'{command}_{index}.json',
                   dict(command=command, split=split, succeeded=success, wall_seconds=seconds,
                        created_utc='2026-09-24T00:00:00+00:00'))


def report_cli(run, resume=False):
    command = [sys.executable, '-B', str(ROOT / 'tools/ref_sads_analysis.py'), '--run', str(run)]
    if resume:
        command.append('--resume')
    return subprocess.run(command,
                          cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=30)


def interrupt_after_first_report_output(run):
    original = analysis_module._publish_frozen
    def interrupt(path, value, kind, resume):
        original(path, value, kind, resume)
        if path.name == 'eval_predictions.jsonl':
            raise RuntimeError('synthetic interruption after first output')
    with mock.patch.object(analysis_module, '_publish_frozen', side_effect=interrupt):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, 'synthetic interruption'):
            report_run(run)


class MathTests(unittest.TestCase):
    def test_raw_first_tie_and_nonfinite_rejection(self):
        self.assertEqual(raw_top1([2., 2., -1.]), 0)
        with self.assertRaises(ValueError):
            raw_top1([float('nan'), 0.])

    def test_focal_soft_labels_and_saturation(self):
        for target in (0., .5, 1.):
            expected = (.25*target+.75*(1-target))*.25*math.log(2)
            self.assertAlmostEqual(focal_loss_fp32([0.], [target]), expected, places=7)
        self.assertTrue(math.isfinite(focal_loss_fp32([-1000., 1000.], [1., 0.])))
        self.assertEqual(focal_loss_fp32([-1000., 1000.], [0., 1.]), 0.)
        with self.assertRaises(ValueError):
            focal_loss_fp32([0.], [-.1])

    def test_iou_boundary_and_degenerate_candidate(self):
        values = candidate_ious([[0, 0, 2, 2], [0, 0, 1, 2], [-2, 0, -1, 2]],
                                [[0, 0, 1, 2]], 2, 2)
        np.testing.assert_array_equal(values, [.5, 1., 0.])
        self.assertTrue(values[0] >= .5)
        self.assertEqual(np.where(values > .5, values, 0)[0], 0.)

    def test_auroc_ties_and_missing_classes(self):
        self.assertEqual(weighted_auc([0, 1, 2], [False, True, True]), 1.)
        self.assertEqual(weighted_auc([0, 1, 1, 2], [False, False, True, True]), .875)
        self.assertTrue(math.isnan(weighted_auc([0, 1], [True, True])))
        self.assertTrue(math.isnan(weighted_auc([1, 1], [False, True])))
        self.assertTrue(math.isnan(weighted_auc([], [])))

    def test_weighted_ranks_equal_expanded_cluster_draw(self):
        x = np.asarray([1., 1., 3., 4., 6.])
        y = np.asarray([4., 2., 2., 1., 0.])
        w = np.asarray([3, 1, 0, 2, 1])
        xx, yy = np.repeat(x, w), np.repeat(y, w)
        self.assertAlmostEqual(weighted_spearman(x, y, w), weighted_spearman(xx, yy), places=14)
        self.assertAlmostEqual(weighted_auc(x, y > 1, w), weighted_auc(xx, yy > 1), places=14)

    def test_bootstrap_is_image_clustered_and_reproducible(self):
        keys = [1, 1, 2, 3, 3, 3]
        a = image_bootstrap_weights(keys, 20, 7)
        b = image_bootstrap_weights(keys, 20, 7)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:, 0], a[:, 1])
        np.testing.assert_array_equal(a[:, 3], a[:, 5])
        np.testing.assert_array_equal(a[:, [0, 2, 3]].sum(axis=1), 3)

    def test_cross_layer_effect_is_expression_balanced(self):
        ds = np.asarray([[-2., 0.], [-6., -6.], [0., 0.]])
        dr = np.zeros((3, 2, 3))
        eligible = np.asarray([[True, False], [True, True], [False, False]])
        result = _effect_summary(ds, dr, eligible, image_bootstrap_weights([1, 2, 3], 30, 4), [1, 2, 3], 1e-6)
        self.assertEqual(result['eligible']['D']['value'], -4.)
        self.assertAlmostEqual(result['itt']['D']['value'], -7/3)
        self.assertEqual(result['eligible']['C']['value'], 1.)
        self.assertEqual(result['eligible']['C']['n_expressions'], 2)

    def test_inference_missing_is_not_zero(self):
        self.assertIsNone(_interval(.6, [float('nan')]*8+[.5, .8], 4)['ci95'])
        self.assertIsNone(_interval(.6, [.5, .8], 1)['ci95'])


class AnalysisTests(unittest.TestCase):
    def test_complete_matrix_and_fh_identity(self):
        report = analyze_run(*fixture())
        self.assertEqual(report['status'], 'PASSED')
        self.assertEqual(len(report['arms']), 25)
        self.assertEqual(len(report['eval_predictions']), 100)
        for arm in report['arms'].values():
            self.assertAlmostEqual(arm['delta_top1'], (arm['F']-arm['H'])/4)
            self.assertEqual(arm['N'], 4)
        self.assertEqual(report['arms']['sinkS_l28_g0']['F'], 2)
        self.assertEqual(report['arms']['sinkS_l28_g0']['H'], 0)
        self.assertEqual(report['arms']['random_l28_g0_s29']['H'], 1)
        self.assertEqual(report['counts']['eligible_sample_layers'], 9)
        self.assertEqual(report['resources']['logical_prediction_rows'], 100)
        self.assertEqual(report['resources']['unique_prediction_forwards'], 58)
        self.assertEqual(report['resources']['scopes']['collector']['physical_forwards'], 4)
        self.assertEqual(report['resources']['scopes']['plain_or_gate']['physical_forwards'], 54)
        self.assertEqual(report['resources']['scopes']['collector']['forward_ms']['total'], 8.)
        self.assertEqual(report['predictivity']['0']['layers']['28']['n_observations'], 9)
        self.assertEqual(report['predictivity']['0']['layers']['28']['auroc']['value'], 1.)
        json.dumps(report, allow_nan=False)
        self.assertIn('re-sinking', render_markdown(report))

    def test_every_layer_falls_back_without_engineering_failure(self):
        report = analyze_run(*fixture(no_op=True))
        self.assertEqual(report['status'], 'PASSED')
        self.assertEqual(report['conclusions']['category_comparison'], 'unidentifiable')
        self.assertEqual(report['conclusions']['continuous_score_prediction'], 'unidentifiable')
        self.assertEqual(report['resources']['unique_prediction_forwards'], 4)
        self.assertEqual(report['effects']['0']['aggregate']['itt']['D']['value'], 0.)
        self.assertIsNone(report['effects']['0']['aggregate']['eligible']['D']['value'])
        self.assertTrue(all(row['N'] == 4 and row['net_repairs'] == 0 for row in report['arms'].values()))
        json.dumps(report, allow_nan=False)

    def test_one_image_does_not_become_multiple_independent_samples(self):
        report = analyze_run(*fixture(one_image=True))
        self.assertEqual(report['counts']['evaluation_images'], 1)
        self.assertIsNone(report['effects']['0']['aggregate']['eligible']['D']['ci95'])
        self.assertEqual(report['conclusions']['category_comparison'], 'unidentifiable')

    def test_one_effect_class_is_na_and_not_an_engineering_failure(self):
        case = fixture()
        for row in case[1]:
            if row['k']:
                row['logits'] = [5., -5.]
                row['prediction_index'] = 0
        report = analyze_run(*case)
        self.assertEqual(report['status'], 'PASSED')
        for layer in report['predictivity']['0']['layers'].values():
            self.assertEqual(layer['non_harmful'], 0)
            self.assertIsNone(layer['auroc']['value'])
            self.assertIsNone(layer['spearman']['value'])

    def test_missing_layer_is_not_silently_removed_from_primary_macro(self):
        case = fixture()
        for selection in case[2]:
            if selection['layer'] == 36:
                for stat in selection['head_statistics']:
                    stat['score'] = None
        report = analyze_run(*case)
        metric = report['predictivity']['0']['macro']['auroc']
        self.assertEqual(metric['preregistered_layers'], 3)
        self.assertEqual(metric['available_layers'], 2)
        self.assertEqual(metric['available_layers_descriptive_mean'], 1.)
        self.assertIsNone(metric['value'])
        self.assertIsNone(metric['ci95'])

    def test_posthoc_gt_changes_outcomes_not_selections(self):
        original = fixture()
        modified = copy.deepcopy(original)
        for target in modified[3]:
            target['answer_boxes'] = [[10, 10, 20, 20]]
        a, b = analyze_run(*original), analyze_run(*modified)
        self.assertEqual(original[2], modified[2])
        self.assertNotEqual(a['arms']['sinkS_l28_g0']['F'], b['arms']['sinkS_l28_g0']['F'])

    def test_missing_or_duplicate_logical_arms_are_rejected(self):
        case = fixture()
        case[1].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete logical'):
            analyze_run(*case)
        case = fixture()
        case[1].append(copy.deepcopy(case[1][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate logical'):
            analyze_run(*case)

    def test_gt_leakage_and_wrong_target_ids_are_rejected(self):
        case = fixture()
        case[0][1]['answer_boxes'] = [[0, 0, 10, 10]]
        with self.assertRaisesRegex(ValueError, 'GT found|unapproved fields'):
            analyze_run(*case)
        case = fixture()
        case[3].append(dict(id='cal', answer_boxes=[[0, 0, 10, 10]]))
        with self.assertRaisesRegex(ValueError, 'GT IDs'):
            analyze_run(*case)

    def test_shared_head_and_reselection_are_rejected(self):
        case = fixture()
        case[2][0]['random_heads']['11'] = 0
        with self.assertRaisesRegex(ValueError, 'shared head'):
            analyze_run(*case)
        case = fixture()
        case[1][1]['head'] = 2
        with self.assertRaisesRegex(ValueError, 'head mismatch|inconsistent intervention identity'):
            analyze_run(*case)

    def test_fake_resource_duplication_is_rejected(self):
        case = fixture()
        case[6].append(copy.deepcopy(case[6][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate physical resource'):
            analyze_run(*case)
        case = fixture()
        case[6].pop()
        with self.assertRaisesRegex(ValueError, 'missing physical resource'):
            analyze_run(*case)
        case = fixture()
        del case[6][0]['peak_reserved_bytes']
        with self.assertRaisesRegex(ValueError, 'Missing/invalid actual resource measurement'):
            analyze_run(*case)
        case = fixture()
        baseline = next(r for r in case[1] if r['id'] == 'e0' and r['kind'] == 'baseline')
        no_effect_head = next(r for r in case[1] if r['id'] == 'e0' and r['head'] == 3)
        self.assertEqual(no_effect_head['logits'], baseline['logits'])
        no_effect_head['physical_forward_id'] = baseline['physical_forward_id']
        with self.assertRaisesRegex(ValueError, 'inconsistent intervention identity'):
            analyze_run(*case)

    def test_reference_backend_is_reported_but_excluded_from_main_matrix(self):
        case = fixture()
        for base in list(case[1]):
            if base['kind'] != 'baseline':
                continue
            reference = dict(base, kind='reference', arm_id='reference_fa2',
                             physical_forward_id=base['physical_forward_id']+'-reference')
            resource = next(r for r in case[6] if r['physical_forward_id'] == base['physical_forward_id'])
            case[6].append(dict(resource, physical_forward_id=reference['physical_forward_id'],
                                phase='reference', collector=False))
            case[1].append(reference)
        report = analyze_run(*case)
        self.assertEqual(len(report['arms']), 26)
        self.assertIn('reference_fa2', report['arms'])
        self.assertEqual(report['effects']['0']['aggregate']['eligible']['C']['n_expressions'], 3)


class CLITests(unittest.TestCase):
    def test_real_io_report_cli_with_reference_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='sads_analysis_cli_') as directory:
            run = Path(directory)
            write_cli_fixture(run, fixture(), reference=True)
            completed = report_cli(run)
            self.assertEqual(completed.returncode, 0, completed.stdout+'\n'+completed.stderr)
            self.assertIn('PASSED', completed.stdout)
            marker = require_stage(run, 'report')
            complete = read_json(run / 'COMPLETE.json')
            report = read_json(run / 'metrics.json')
            self.assertEqual(complete['report_stage_sha256'], digest(run / 'stages/report.json'))
            self.assertEqual(set(marker['parents']), {'intervene'})
            self.assertEqual(len(read_jsonl(run / 'eval_predictions.jsonl')), 104)
            self.assertEqual(report['counts']['logical_arms'], 26)
            self.assertIn('reference_fa2', report['arms'])
            self.assertEqual(report['resources']['unique_prediction_forwards'], 62)
            resource = report['resources']
            self.assertEqual(resource['job_stage_wall_seconds_sum'], 5.5)
            self.assertEqual(resource['recorded_command_invocations'], 3)
            self.assertEqual(resource['failed_command_invocations'], 1)
            self.assertGreater(resource['report_compute_wall_seconds'], 0)
            self.assertEqual(sum(event['wall_seconds'] for event in resource['non_forward_events']), 5.5)
            self.assertTrue(all(name in marker['files'] for name in resource['timing_snapshot_files']))
            self.assertIn('不含命令之间的用户等待', (run / 'report.md').read_text(encoding='utf-8'))
            outputs = ['eval_predictions.jsonl', 'metrics.json', 'report.md', 'COMPLETE.json', 'stages/report.json']
            before = {name: digest(run / name) for name in outputs}
            repeated = report_cli(run)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn('refusing overwrite', repeated.stderr)
            self.assertEqual(before, {name: digest(run / name) for name in outputs})

    def test_real_io_report_cli_all_fallback_is_complete(self):
        with tempfile.TemporaryDirectory(prefix='sads_analysis_cli_') as directory:
            run = Path(directory)
            write_cli_fixture(run, fixture(no_op=True))
            completed = report_cli(run)
            self.assertEqual(completed.returncode, 0, completed.stdout+'\n'+completed.stderr)
            report = read_json(run / 'metrics.json')
            self.assertEqual(report['conclusions']['category_comparison'], 'unidentifiable')
            self.assertEqual(report['resources']['total_physical_forwards'], 4)
            self.assertEqual(len(read_jsonl(run / 'eval_predictions.jsonl')), 100)
            self.assertEqual(read_json(run / 'COMPLETE.json')['status'], 'PASSED')

    def test_cli_missing_resource_is_not_marked_complete(self):
        with tempfile.TemporaryDirectory(prefix='sads_analysis_cli_') as directory:
            run = Path(directory)
            case = fixture()
            del case[6][0]['request_ms']
            write_cli_fixture(run, case)
            completed = report_cli(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('Missing/invalid actual resource measurement', completed.stderr)
            self.assertFalse((run / 'COMPLETE.json').exists())
            self.assertFalse((run / 'metrics.json').exists())

    def test_partial_report_requires_explicit_resume_and_preserves_measured_time(self):
        with tempfile.TemporaryDirectory(prefix='sads_analysis_cli_') as directory:
            run = Path(directory)
            write_cli_fixture(run, fixture())
            interrupt_after_first_report_output(run)
            self.assertTrue((run / 'eval_predictions.jsonl').exists())
            self.assertFalse((run / 'metrics.json').exists())
            frozen_path = run / 'parts/report/prepared.json'
            frozen = read_json(frozen_path)['payload']
            hashes = {str(p): digest(p) for p in [frozen_path, run / 'eval_predictions.jsonl']}
            rejected = report_cli(run)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('use --resume', rejected.stderr)
            completed = report_cli(run, resume=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(hashes, {name: digest(name) for name in hashes})
            self.assertEqual(read_json(run / 'metrics.json')['resources']['report_compute_wall_seconds'],
                             frozen['report']['resources']['report_compute_wall_seconds'])
            require_stage(run, 'report')

    def test_report_marker_without_complete_is_recoverable(self):
        with tempfile.TemporaryDirectory(prefix='sads_analysis_cli_') as directory:
            run = Path(directory)
            write_cli_fixture(run, fixture())
            with mock.patch.object(analysis_module, '_finish_complete', side_effect=RuntimeError('before COMPLETE')):
                with self.assertRaisesRegex(RuntimeError, 'before COMPLETE'):
                    report_run(run)
            require_stage(run, 'report')
            self.assertFalse((run / 'COMPLETE.json').exists())
            paths = ['metrics.json', 'report.md', 'eval_predictions.jsonl', 'stages/report.json']
            hashes = {name: digest(run / name) for name in paths}
            completed = report_cli(run, resume=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(hashes, {name: digest(run / name) for name in paths})
            self.assertEqual(read_json(run / 'COMPLETE.json')['report_stage_sha256'], hashes['stages/report.json'])

    def test_resume_refuses_conflicting_partial_output(self):
        with tempfile.TemporaryDirectory(prefix='sads_analysis_cli_') as directory:
            run = Path(directory)
            write_cli_fixture(run, fixture())
            interrupt_after_first_report_output(run)
            (run / 'eval_predictions.jsonl').write_text('[]\n', encoding='utf-8')
            completed = report_cli(run, resume=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('Report resume content mismatch', completed.stderr)
            self.assertEqual((run / 'eval_predictions.jsonl').read_text(encoding='utf-8'), '[]\n')
            self.assertFalse((run / 'COMPLETE.json').exists())

    def test_resume_rejects_changed_frozen_timing(self):
        with tempfile.TemporaryDirectory(prefix='sads_analysis_cli_') as directory:
            run = Path(directory)
            write_cli_fixture(run, fixture())
            interrupt_after_first_report_output(run)
            timing = run / 'timings/prepare_0.json'
            record = read_json(timing)
            record['wall_seconds'] += 1
            timing.write_text(json.dumps(record), encoding='utf-8')
            completed = report_cli(run, resume=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('Frozen report input changed: timings/prepare_0.json', completed.stderr)
            self.assertFalse((run / 'COMPLETE.json').exists())

    def test_resume_can_promote_complete_pending_computation_write(self):
        with tempfile.TemporaryDirectory(prefix='sads_analysis_cli_') as directory:
            run = Path(directory)
            write_cli_fixture(run, fixture())
            interrupt_after_first_report_output(run)
            frozen = run / 'parts/report/prepared.json'
            pending = frozen.with_name(frozen.name+'.writing')
            pending.write_bytes(frozen.read_bytes())
            frozen.unlink()
            completed = report_cli(run, resume=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(frozen.exists())
            self.assertFalse(pending.exists())
            self.assertTrue((run / 'COMPLETE.json').exists())
            self.assertTrue((run / 'metrics.json').exists())

    def test_cli_unknown_gt_input_field_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='sads_analysis_cli_') as directory:
            run = Path(directory)
            case = fixture()
            case[0][0]['labels'] = [1., 0.]
            write_cli_fixture(run, case)
            completed = report_cli(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('unapproved fields', completed.stderr)
            self.assertFalse((run / 'COMPLETE.json').exists())

    def test_cli_gt_tampering_is_caught_by_prepare_parent_hash(self):
        with tempfile.TemporaryDirectory(prefix='sads_analysis_cli_') as directory:
            run = Path(directory)
            write_cli_fixture(run, fixture())
            target = run / 'eval_targets.jsonl'
            target.write_text(target.read_text(encoding='utf-8')+'\n', encoding='utf-8')
            completed = report_cli(run)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('Artifact changed: eval_targets.jsonl', completed.stderr)
            self.assertFalse((run / 'COMPLETE.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
