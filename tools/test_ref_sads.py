"""Run all offline SADS tests. No real checkpoint, server, GPU or downloads.

The runner integration fixture has a deliberately synthetic runtime, passed as
a Python test dependency. The production CLI has no fake-runtime option.
"""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import ref_sads as runner
from ref_sads_io import (ROOT, SOURCE_FILES, digest, load_run, marker_path, object_hash,
    part_read, part_write, read_json, read_jsonl, require_stage, source_digest,
    split_development, stage_lock, validate_inputs, write_json, write_jsonl, write_stage)


class FakeRuntime:
    """Deterministic scalar test fixture, NOT a WeDetect performance surrogate."""
    instances = 0
    uniform_stats = False
    bad_identity = False
    interrupt_after = None
    calls = 0

    def __init__(self, loaded, backend=None):
        type(self).instances += 1
        self.loaded, self.config = loaded, loaded['config']
        self.backend = backend or self.config['attention_backend']
        self.environment = dict(test_fixture=True, backend=self.backend)

    def prepare_input(self, row):
        audit = dict(sequence_length=20 + row['image_key'] % 10, id=row['id'])
        return dict(audit=audit, input_sha256=object_hash(audit))

    def forward(self, row, mode, physical_id, phase, gate=None, expected_input=None):
        cls = type(self)
        cls.calls += 1
        if cls.interrupt_after is not None and cls.calls > cls.interrupt_after:
            raise RuntimeError('Simulated interrupted run')
        prepared = self.prepare_input(row)
        if expected_input is not None and expected_input != prepared['input_sha256']:
            raise ValueError('Input mismatch')
        logits = [-.5, .5]
        if gate:
            change = (1-gate['gate']) * (1 if gate['head'] < 8 else -.2)
            logits = [logits[0]+change, logits[1]-change]
        elif cls.bad_identity and mode == 'gate':
            logits[0] += .1
        statistics = {}
        if mode in ('stats', 'reference_stats'):
            rng = np.random.default_rng(row['image_key'])
            for layer in self.config['layers']:
                records = []
                for head in range(32):
                    x = (.02 if head < 16 else .16) + rng.normal(0, .0005)
                    e = (.15 if head < 8 else .8) + rng.normal(0, .003)
                    if cls.uniform_stats:
                        x = .1
                    records.append(dict(head=head, x=float(x), H=float(e*math.log(20)),
                        e=float(e), valid=True, x_valid=True, entropy_valid=True,
                        query_count=20, valid_query_count=20, reason=None))
                statistics[layer] = records
        return dict(id=row['id'], logits=logits, input_sha256=prepared['input_sha256'],
            input_audit=prepared['audit'], backend=self.backend, statistics=statistics,
            operation=None if gate is None else {k: gate[k] for k in ('layer', 'head', 'gate')}, diagnostics={},
            resource=dict(physical_forward_id=physical_id, id=row['id'], phase=phase,
                collector=mode in ('stats', 'reference_stats'), backend=self.backend,
                forward_ms=1., request_ms=2., peak_allocated_bytes=100., peak_reserved_bytes=200.))

    def close(self):
        pass


def synthetic_run(root, backend='flash_attention_2'):
    root.mkdir()
    config = read_json(ROOT / 'config/ref_sads_pilot_v1.json')
    config.update(counts=dict(calibration=24, evaluation=4), candidate_count=2, attention_backend=backend)
    config['gmm'].update(init_seeds=[0, 1], bootstrap_repeats=4, bootstrap_min_success=3,
                         bootstrap_iqr_fraction=.25, min_observations=80, min_images=4)
    config['analysis'].update(bootstrap_repeats=40)
    rows, targets = [], []
    for split, count in config['counts'].items():
        for i in range(count):
            key = i + (100 if split == 'calibration' else 200)
            boxes = [[0, 0, 10, 10], [10, 10, 20, 20]]
            rows.append(dict(id=f'{split}_{i}', split=split, image_key=key,
                image_name=f'COCO_train2014_{key:012}.jpg', image_sha256=f'fake-image-{key}',
                query='synthetic target', width=20, height=20, candidate_boxes=boxes,
                candidate_sha256=object_hash(boxes)))
            if split == 'evaluation':
                targets.append(dict(id=rows[-1]['id'], answer_boxes=[[0, 0, 10, 10]]))
    manifest = dict(config_sha256=object_hash(config), maximum_forwards=5392,
                    sources={name: dict(lf_sha256=source_digest(ROOT/name)) for name in SOURCE_FILES})
    write_json(root/'manifest.json', manifest)
    write_json(root/'config.json', config)
    write_jsonl(root/'inputs.jsonl', rows)
    write_jsonl(root/'eval_targets.jsonl', targets)
    write_stage(root, 'prepare', ['config.json', 'inputs.jsonl', 'eval_targets.jsonl'])
    return SimpleNamespace(run=root, n=10, resume=False)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ref_sads_test_')
        self.addCleanup(self.temp.cleanup)
        FakeRuntime.instances = FakeRuntime.calls = 0
        FakeRuntime.uniform_stats = FakeRuntime.bad_identity = False
        FakeRuntime.interrupt_after = None
        self.args = synthetic_run(Path(self.temp.name)/'run')

    def call(self, func, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*args, **kwargs)

    def through_selection(self):
        self.call(runner.check, self.args, FakeRuntime)
        self.args.split = 'calibration'
        self.call(runner.collect, self.args, FakeRuntime)
        self.call(runner.calibrate, self.args)
        self.args.split = 'evaluation'
        self.call(runner.collect, self.args, FakeRuntime)
        self.call(runner.select, self.args)

    def test_complete_stages_fake_runtime_and_posthoc_join(self):
        self.through_selection()
        self.call(runner.intervene, self.args, FakeRuntime)
        run = self.args.run
        require_stage(run, 'intervene')
        predictions = read_jsonl(run/'predictions.jsonl')
        resources = read_jsonl(run/'resources.jsonl')
        self.assertEqual(len(predictions), 4*25)
        self.assertLessEqual(len(resources), 46+28+4*24)
        self.assertEqual(len(read_jsonl(run/'eval_targets.jsonl')), 4)
        from ref_sads_analysis import analyze_run
        report = analyze_run(read_jsonl(run/'inputs.jsonl'), predictions,
            read_jsonl(run/'selections.jsonl'), read_jsonl(run/'eval_targets.jsonl'),
            read_json(run/'config.json'), read_json(run/'checks/consistency.json'), resources)
        self.assertEqual(report['status'], 'PASSED')
        self.assertEqual(len(report['arms']), 25)

    def test_all_unclassified_keeps_full_matrix_without_model_loading(self):
        FakeRuntime.uniform_stats = True
        self.through_selection()
        before = FakeRuntime.instances
        self.call(runner.intervene, self.args, FakeRuntime)
        self.assertEqual(before, FakeRuntime.instances)
        records = read_jsonl(self.args.run/'predictions.jsonl')
        self.assertEqual(len(records), 100)
        self.assertEqual(len({r['physical_forward_id'] for r in records}), 4)
        self.assertTrue(all(r['k'] == 0 for r in records))

    def test_backend_change_always_adds_original_backend_baselines(self):
        self.args = synthetic_run(Path(self.temp.name)/'eager', 'eager')
        self.through_selection()
        self.call(runner.intervene, self.args, FakeRuntime)
        records = read_jsonl(self.args.run/'predictions.jsonl')
        reference = [r for r in records if r['kind'] == 'reference']
        self.assertEqual(len(reference), 4)
        self.assertTrue(all(r['backend'] == 'flash_attention_2' for r in reference))
        self.assertTrue(all(r['backend'] == 'eager' for r in records if r['kind'] != 'reference'))

    def test_sham_failure_prevents_any_followup_stage(self):
        FakeRuntime.bad_identity = True
        with self.assertRaisesRegex(ValueError, 'does not reproduce'):
            self.call(runner.check, self.args, FakeRuntime)
        self.assertFalse(marker_path(self.args.run, 'check').exists())
        self.args.split = 'calibration'
        with self.assertRaisesRegex(ValueError, 'not complete'):
            self.call(runner.collect, self.args, FakeRuntime)

    def test_missing_dependency_and_tampering_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not complete'):
            runner.select(self.args)
        with (self.args.run/'inputs.jsonl').open('a') as stream:
            stream.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'Artifact changed'):
            load_run(self.args.run)

    def test_budget_failure_blocks_followup(self):
        from unittest.mock import patch
        original = runner.load_run
        def limited(path):
            loaded = original(path)
            loaded['config']['budget_seconds'] = 1
            return loaded
        with patch('ref_sads.load_run', side_effect=limited):
            with self.assertRaisesRegex(ValueError, 'exceeds fixed budget'):
                self.call(runner.check, self.args, FakeRuntime)
        self.assertFalse(marker_path(self.args.run, 'check').exists())
        self.args.split = 'calibration'
        with self.assertRaisesRegex(ValueError, 'not complete'):
            self.call(runner.collect, self.args, FakeRuntime)

    def test_complete_stage_refuses_before_model_load(self):
        self.call(runner.check, self.args, FakeRuntime)
        before = FakeRuntime.instances
        with self.assertRaisesRegex(ValueError, 'already complete'):
            self.call(runner.check, self.args, FakeRuntime)
        self.assertEqual(FakeRuntime.instances, before)

    def test_resume_reuses_only_verified_completed_forwards(self):
        FakeRuntime.interrupt_after = 9
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.call(runner.check, self.args, FakeRuntime)
        with self.assertRaisesRegex(ValueError, 'Partial'):
            self.call(runner.check, self.args, FakeRuntime)
        FakeRuntime.interrupt_after = None
        calls_before = FakeRuntime.calls
        self.args.resume = True
        self.call(runner.check, self.args, FakeRuntime)
        self.assertEqual(FakeRuntime.calls - calls_before, 46-9)
        self.assertEqual(len(read_jsonl(self.args.run/'checks/resources.jsonl')), 46)

    def test_source_signature_changes_rejected(self):
        original = source_digest
        from unittest.mock import patch
        with patch('ref_sads_io.source_digest', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                load_run(self.args.run)


class IOTests(unittest.TestCase):
    def test_input_schema_forbids_labels_and_targets(self):
        config = dict(counts={'calibration': 0, 'evaluation': 1}, candidate_count=1)
        row = dict(id='a', image_key=1, split='evaluation', image_name='a.jpg', query='x',
            image_sha256='x', candidate_boxes=[[0, 0, 1, 1]], width=2, height=2,
            candidate_sha256=object_hash([[0, 0, 1, 1]]))
        validate_inputs([row], config)
        for field in ('gt', 'answer_boxes', 'loss', 'correct'):
            with self.assertRaises(ValueError):
                validate_inputs([dict(row, **{field: 1})], config)

    def test_sampling_ignores_labels_and_protects_images(self):
        rows = [dict(id=f'{image}_{j}', image_key=image, split='dev', answer_boxes=[image])
                for image in range(30) for j in range(3)]
        counts = dict(calibration=10, evaluation=20)
        a = split_development(rows, {99}, counts, 42)
        b = split_development([dict(r, answer_boxes=['changed']) for r in rows], {99}, counts, 42)
        self.assertEqual([(s, r['id']) for s, r in a], [(s, r['id']) for s, r in b])
        ca = {r['image_key'] for s, r in a if s == 'calibration'}
        ev = {r['image_key'] for s, r in a if s == 'evaluation'}
        self.assertFalse(ca & ev)
        with self.assertRaises(ValueError):
            split_development(rows, {0}, counts, 42)

    def test_safe_atomic_write_and_lock_resume(self):
        with tempfile.TemporaryDirectory(prefix='ref_sads_io_test_') as folder:
            root = Path(folder)
            path = root/'value.json'
            # Same fully written temporary bytes are recoverable without discard.
            pending = path.with_name(path.name+'.writing')
            pending.write_bytes((json.dumps({'a': 1}, indent=2)+'\n').encode('utf-8'))
            write_json(path, {'a': 1})
            self.assertEqual(read_json(path), {'a': 1})
            with self.assertRaises(ValueError):
                write_json(path, {'a': 2})
            with stage_lock(root, 'test'):
                with self.assertRaisesRegex(ValueError, 'Another process'):
                    with stage_lock(root, 'test', resume=True):
                        pass
            with stage_lock(root, 'test', resume=True):
                pass

    def test_corrupt_partial_record_is_not_reused(self):
        with tempfile.TemporaryDirectory(prefix='ref_sads_io_test_') as folder:
            path = Path(folder)/'part.json'
            part_write(path, {'logits': [1., 2.]}, 'manifest')
            value = read_json(path)
            value['payload']['logits'][0] = 0
            path.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'corrupted'):
                part_read(path, 'manifest')


def main():
    # Fail, do not silently skip, if numerical test dependencies are missing.
    import torch
    import test_ref_sads_core
    test_ref_sads_core.run_all()
    import test_ref_sads_stats
    import test_ref_sads_analysis
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    for module in (__import__(__name__), test_ref_sads_stats, test_ref_sads_analysis):
        suite.addTests(loader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(f'Offline tests PASS (torch {torch.__version__}, numpy {np.__version__}). '
          'No production model, checkpoint, GPU experiment or training ran.')


if __name__ == '__main__':
    main()
