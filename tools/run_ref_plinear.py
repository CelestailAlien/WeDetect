"""Unattended full P-linear run. Activate conda first; launch with nohup.

Only stdlib here: each stage runs in its own process, releasing GPU memory on exit.
No automatic retry, resume, GPU switching, or changes to scientific hyperparameters.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_GIB = 25
STEPS = [('preflight', 'ref_plinear.py'), ('cache', 'ref_plinear.py'),
         ('train', 'ref_plinear_train.py'), ('cache_val', 'ref_plinear.py'),
         ('evaluate', 'ref_plinear_eval.py')]


def timestamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_status(directory, status):
    # directory was exclusively created by this run; never edits a prior run.
    temporary = directory / 'status.tmp'
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(directory / 'status.json')


def stream_command(command, log_path, pipeline_log, env):
    """Tee output without a shell pipeline; preserve the actual child exit code."""
    with log_path.open('x', encoding='utf-8') as stage_log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in child.stdout:
                stage_log.write(line)
                stage_log.flush()
                pipeline_log.write(line)
                pipeline_log.flush()
                print(line, end='', flush=True)
            return child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            child.stdout.close()


def stage_check(stage, output):
    """Exit code zero alone is insufficient; verify the expected completion contract."""
    if stage == 'preflight':
        plan = read_json(output / 'plan.json')
        assert plan['counts'] == dict(train=5000, dev=1000, validation=2573)
        assert not plan['smoke_only'] and plan['depths'] == [9, 18, 24, 30, 36]
        assert plan['selection_sha256'] and plan['uni_provenance']
    elif stage in ('cache', 'cache_val'):
        directory = 'cache_fit' if stage == 'cache' else 'cache_validation'
        complete = read_json(output / directory / 'COMPLETE.json')
        assert complete['status'] == 'PASSED'
        assert complete['stage'] == ('fit' if stage == 'cache' else 'validation')
        assert complete['samples'] == (6000 if stage == 'cache' else 2573)
    elif stage == 'train':
        complete = read_json(output / 'train/COMPLETE.json')
        assert complete['status'] == 'PASSED'
        assert [c['seed'] for c in complete['checkpoints']] == [42, 43, 44]
        assert all((output / 'train' / c['file']).is_file() for c in complete['checkpoints'])
    elif stage == 'evaluate':
        result = read_json(output / 'evaluation/summary.json')
        assert result['status'] == 'PASSED' and not result['smoke_only']
        assert result['n'] == 2573 and result['depths'] == [9, 18, 24, 30, 36]
        assert result['seeds'] == [42, 43, 44] and len(result['curve']) == 32
    else:
        raise ValueError(stage)


def child_environment(output):
    env = dict(os.environ)
    # Explicitly override stale smoke settings and dataset paths from previous shell sessions.
    paths = dict(PL_TRAIN_ANN='data/refcocog_plinear_umd_v1/refcocog_train.json',
        PL_TRAIN_PROPOSALS='results/ref_plinear_uni_k100_v1/train_proposals.json',
        PL_UNI_RUN='results/ref_plinear_uni_k100_v1',
        PL_SELECTION='results/ref_plinear_data_audit_v1/planned_selection.json',
        PL_IMAGES='data/coco2014', PL_REF='checkpoints/WeDetect-Ref-4B',
        PL_REFERENCE='results/ref_full_d30_refcocog_validation',
        PL_VAL_ANN='wedetect_ref/eval_grounding/eval_refcoco/refcocog_validation.json',
        PL_VAL_PROPOSALS='wedetect_ref/eval_grounding/eval_refcoco/refcoco_proposals_all.json')
    env.update({key: str(ROOT / value) for key, value in paths.items()})
    env.update(PL_OUT=str(output), PL_TRAIN_N='5000', PL_DEV_N='1000', PL_VAL_N='0',
               PL_DEVICE='cuda', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    # CUDA_VISIBLE_DEVICES is inherited unchanged: do not choose or switch a GPU.
    env.pop('PL_STAGE', None)
    return env


def precheck(output, env):
    assert __debug__, 'Do not use python -O'
    assert int(env.get('WORLD_SIZE', '1')) == 1, 'Use python, not multi-rank torchrun'
    assert not output.exists(), f'Refusing to overwrite {output}; use a new PL_RUN_OUT'
    for key in ('PL_TRAIN_ANN', 'PL_TRAIN_PROPOSALS', 'PL_SELECTION', 'PL_VAL_ANN', 'PL_VAL_PROPOSALS'):
        assert Path(env[key]).is_file(), f'Missing {key}: {env[key]}'
    for key in ('PL_IMAGES', 'PL_REF', 'PL_REFERENCE'):
        assert Path(env[key]).is_dir(), f'Missing {key}: {env[key]}'
    assert read_json(Path(env['PL_UNI_RUN']) / 'COMPLETE.json')['status'] == 'PASSED'
    for _, script in STEPS:
        assert (ROOT / 'tools' / script).is_file()
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free / 1024**3
    assert free >= MIN_FREE_GIB, f'Only {free:.1f} GiB free; need at least {MIN_FREE_GIB} GiB for feature caches'
    return free


def run():
    default = ROOT / 'results' / ('ref_plinear_refcocog_uni_full_' + datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}')
    output = Path(os.environ.get('PL_RUN_OUT', str(default)))
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    directory = Path(str(output) + '_runner')
    directory.mkdir(parents=True, exist_ok=False)  # independent of preflight's must-not-exist output
    status = dict(status='RUNNING', current_stage='startup', pid=os.getpid(), started_at=timestamp(),
        output=str(output), runner_directory=str(directory), python=sys.executable,
        conda_environment=os.environ.get('CONDA_DEFAULT_ENV'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), finished_stages=[])
    write_status(directory, status)
    with (directory / 'pipeline.log').open('x', encoding='utf-8') as pipeline_log:
        def announce(message):
            line = f'[{timestamp()}] {message}\n'
            print(line, end='', flush=True)
            pipeline_log.write(line)
            pipeline_log.flush()
        try:
            announce(f'OUTPUT: {output}')
            announce(f'STATUS: {directory / "status.json"}')
            announce(f'PYTHON: {sys.executable}; CUDA_VISIBLE_DEVICES={status["cuda_visible_devices"]}')
            env = child_environment(output)
            free = precheck(output, env)
            announce(f'Input paths present, {free:.1f} GiB free. Frozen 5000/1000/2573 protocol.')
            # Separate process: inspecting CUDA must not leave a context resident in the runner.
            check_cuda = ('import torch; assert torch.cuda.is_available(), "CUDA unavailable"; '
                          'assert torch.cuda.is_bf16_supported(), "BF16 unsupported"; '
                          'print("GPU:", torch.cuda.get_device_name(0), "torch:", torch.__version__)')
            rc = stream_command([sys.executable, '-u', '-B', '-c', check_cuda], directory / 'startup.log', pipeline_log, env)
            if rc != 0:
                raise RuntimeError(f'CUDA startup check exited {rc}')
            for stage, script in STEPS:
                status.update(current_stage=stage, stage_started_at=timestamp())
                write_status(directory, status)
                announce(f'START {stage}')
                started = time.monotonic()
                stage_env = dict(env, PL_STAGE=stage)
                rc = stream_command([sys.executable, '-u', '-B', str(ROOT / 'tools' / script)],
                                    directory / f'{stage}.log', pipeline_log, stage_env)
                if rc != 0:
                    status['child_exit_code'] = rc
                    raise RuntimeError(f'{stage} exited {rc}; later stages were NOT started')
                stage_check(stage, output)
                status['finished_stages'].append(dict(stage=stage, seconds=round(time.monotonic()-started, 2), finished_at=timestamp()))
                write_status(directory, status)
                announce(f'PASS {stage}')
            status.update(status='SUCCESS', current_stage='complete', finished_at=timestamp())
            write_status(directory, status)
            (directory / 'SUCCESS.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
            announce(f'ALL COMPLETED: {output / "evaluation/summary.md"}')
            return 0
        except (Exception, KeyboardInterrupt) as error:
            status.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                          finished_at=timestamp(), error=f'{type(error).__name__}: {error}')
            write_status(directory, status)
            details = traceback.format_exc()
            pipeline_log.write(details)
            pipeline_log.flush()
            print(details, file=sys.stderr, flush=True)
            announce(f'{status["status"]} at {status["current_stage"]}; no automatic retry or resume. See stage log.')
            return 130 if isinstance(error, KeyboardInterrupt) else 1


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Received signal {signum}')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(run())
