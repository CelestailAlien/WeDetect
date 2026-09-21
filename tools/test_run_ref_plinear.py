"""Stdlib-only runner tests. Toy subprocesses; never loads a model or CUDA."""
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import run_ref_plinear as runner


def fails(fn):
    try:
        fn()
    except AssertionError:
        return
    raise AssertionError('Invalid stage contract was accepted')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_contracts(root):
    output = root / 'contracts'
    save(output / 'plan.json', dict(counts=dict(train=5000, dev=1000, validation=2573),
        smoke_only=False, depths=[9, 18, 24, 30, 36], selection_sha256='fixture', uni_provenance={'fixture': True}))
    runner.stage_check('preflight', output)
    for stage, directory, kind, samples in [('cache', 'cache_fit', 'fit', 6000), ('cache_val', 'cache_validation', 'validation', 2573)]:
        save(output / directory / 'COMPLETE.json', dict(status='PASSED', stage=kind, samples=samples))
        runner.stage_check(stage, output)
        save(output / directory / 'COMPLETE.json', dict(status='PASSED', stage=kind, samples=1))
        fails(lambda: runner.stage_check(stage, output))
    checkpoints = [dict(seed=s, file=f'head{s}.pt') for s in [42, 43, 44]]
    save(output / 'train/COMPLETE.json', dict(status='PASSED', checkpoints=checkpoints))
    for c in checkpoints:
        (output / 'train' / c['file']).write_bytes(b'SYNTHETIC-PRESENCE-ONLY')
    runner.stage_check('train', output)
    summary = dict(status='PASSED', smoke_only=False, n=2573, depths=[9, 18, 24, 30, 36], seeds=[42, 43, 44], curve=[{}]*32)
    save(output / 'evaluation/summary.json', summary)
    runner.stage_check('evaluate', output)
    save(output / 'evaluation/summary.json', dict(summary, smoke_only=True))
    fails(lambda: runner.stage_check('evaluate', output))


def test_runner(root, failed_stage=None):
    output = root / ('success' if failed_stage is None else 'failure')
    commands, verified = [], []
    real_stream = runner.stream_command

    def toy_stream(command, log_path, pipeline_log, env):
        stage = log_path.stem
        commands.append(stage)
        assert command[0] == sys.executable
        assert env['PL_TRAIN_N'] == '5000' and env['PL_DEV_N'] == '1000' and env['PL_VAL_N'] == '0'
        assert env['CUDA_VISIBLE_DEVICES'] == '2' and env['PL_OUT'] == str(output)
        code = f'print("SYNTHETIC runner stage {stage}"); raise SystemExit({7 if stage == failed_stage else 0})'
        return real_stream([sys.executable, '-u', '-c', code], log_path, pipeline_log, env)

    with patch.object(runner, 'ROOT', root), \
         patch.dict(os.environ, {'PL_RUN_OUT': str(output), 'PL_TRAIN_N': '16', 'PL_DEV_N': '8',
                                'PL_VAL_N': '6', 'CUDA_VISIBLE_DEVICES': '2'}), \
         patch.object(runner, 'precheck', return_value=50.), \
         patch.object(runner, 'stream_command', side_effect=toy_stream), \
         patch.object(runner, 'stage_check', side_effect=lambda stage, output: verified.append(stage)):
        rc = runner.run()
    directory = Path(str(output) + '_runner')
    status = runner.read_json(directory / 'status.json')
    if failed_stage is None:
        assert rc == 0 and status['status'] == 'SUCCESS'
        assert commands == ['startup', 'preflight', 'cache', 'train', 'cache_val', 'evaluate']
        assert verified == [stage for stage, _ in runner.STEPS]
        assert (directory / 'SUCCESS.json').is_file()
    else:
        assert rc == 1 and status['status'] == 'FAILED' and status['child_exit_code'] == 7
        assert commands == ['startup', 'preflight', 'cache', 'train']
        assert verified == ['preflight', 'cache']
        assert not (directory / 'SUCCESS.json').exists()
        assert not (directory / 'cache_val.log').exists()
    assert 'SYNTHETIC runner stage' in (directory / 'pipeline.log').read_text(encoding='utf-8')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='plinear-runner-test-') as directory:
        root = Path(directory)
        test_contracts(root)
        test_runner(root)
        test_runner(root, failed_stage='train')
    print('PASS: stage contracts, real subprocess logging, stale-smoke reset, GPU inheritance, success and fail-fast.')
