"""Small, stdlib-only artifact and provenance helpers for the SADS pilot."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [f'tools/{name}' for name in (
    'ref_sads.py', 'ref_sads_io.py', 'ref_sads_core.py', 'ref_sads_stats.py',
    'ref_sads_analysis.py', 'ref_e0.py', 'ref_e0_core.py', 'ref_e0_data.py',
    'humanref_pipeline.py')]
SOURCE_FILES += ['wedetect_ref/models/qwen3vl_referring.py',
                 'wedetect_ref/models/vision_process.py']
INPUT_KEYS = {'id', 'image_key', 'split', 'image_name', 'query', 'image_sha256',
              'candidate_boxes', 'candidate_sha256', 'width', 'height'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'))


def object_hash(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write(path, data):
    """Exclusive atomic publish; failed writes never masquerade as complete files."""
    path = Path(path)
    require(not path.exists(), f'Refusing overwrite: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    if temporary.exists():
        require(temporary.read_bytes() == data,
                f'Interrupted write differs from intended data: {temporary}; preserve it and use a new run')
    else:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            import os
            os.fsync(stream.fileno())
    # link is atomic AND refuses overwrite, unlike replace/rename on POSIX.
    import os
    os.link(temporary, path)
    temporary.unlink()


def write_json(path, value):
    _write(path, (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8'))


def write_jsonl(path, rows):
    _write(path, ''.join(canonical(row) + '\n' for row in rows).encode('utf-8'))


def safe_child(root, relative):
    root = Path(root).resolve()
    child = (root / relative).resolve()
    require(child != root and root in child.parents, f'Path leaves intended directory: {relative}')
    return child


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def marker_path(run, stage):
    require(re.fullmatch(r'[a-z_]+', stage) is not None, 'Invalid stage name')
    return Path(run) / 'stages' / f'{stage}.json'


def require_stage(run, stage, _seen=None):
    run = Path(run)
    seen = set() if _seen is None else _seen
    require(stage not in seen, 'Cyclic stage dependencies')
    seen = seen | {stage}
    path = marker_path(run, stage)
    require(path.is_file(), f'Required stage not complete: {stage}')
    marker = read_json(path)
    require(marker['stage'] == stage and marker['status'] == 'PASSED', f'Invalid stage: {stage}')
    require(marker['manifest_sha256'] == digest(run / 'manifest.json'), 'Manifest changed')
    for parent, sha in marker['parents'].items():
        require(digest(marker_path(run, parent)) == sha, f'Parent stage changed: {parent}')
        require_stage(run, parent, seen)
    for name, sha in marker['files'].items():
        require(digest(safe_child(run, name)) == sha, f'Artifact changed: {name}')
    return marker


def write_stage(run, stage, files, metadata=None, parents=None):
    run = Path(run)
    require(len(files) == len(set(files)) and files, 'Duplicate/empty stage file list')
    parent_hashes = {}
    for parent in parents or []:
        require_stage(run, parent)
        parent_hashes[parent] = digest(marker_path(run, parent))
    marker = dict(stage=stage, status='PASSED', created_utc=utc_now(),
                  manifest_sha256=digest(run / 'manifest.json'), parents=parent_hashes,
                  files={name: digest(safe_child(run, name)) for name in files},
                  metadata=metadata or {})
    write_json(marker_path(run, stage), marker)
    return marker


def validate_inputs(rows, config):
    require(len(rows) == sum(config['counts'].values()), 'Wrong frozen input count')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate expression ID')
    images = {}
    for row in rows:
        require(set(row) == INPUT_KEYS, f'Input schema includes missing/unapproved fields: {set(row) ^ INPUT_KEYS}')
        require(row['split'] in config['counts'], 'Unknown split')
        require(isinstance(row['query'], str) and row['query'].strip(), 'Empty expression')
        require(row['width'] > 0 and row['height'] > 0, 'Invalid dimensions')
        boxes = row['candidate_boxes']
        require(len(boxes) == config['candidate_count'], 'Changed candidate count')
        require(object_hash(boxes) == row['candidate_sha256'], 'Changed candidate order/coordinates')
        for b in boxes:
            require(len(b) == 4 and all(math.isfinite(x) for x in b), 'Nonfinite box')
            require(0 <= b[0] <= b[2] <= row['width'] and 0 <= b[1] <= b[3] <= row['height'], 'Invalid box geometry')
        previous = images.setdefault(row['image_key'], (row['split'], row['image_sha256'], row['candidate_sha256']))
        require(previous == (row['split'], row['image_sha256'], row['candidate_sha256']), 'Image split leakage or inconsistent image/candidates')
    for split, count in config['counts'].items():
        require(sum(r['split'] == split for r in rows) == count, f'Wrong {split} count')


def load_run(run):
    run = Path(run)
    require_stage(run, 'prepare')
    manifest = read_json(run / 'manifest.json')
    config = read_json(run / 'config.json')
    require(object_hash(config) == manifest['config_sha256'], 'Configuration changed')
    for name, hashes in manifest['sources'].items():
        require(source_digest(ROOT / name) == hashes['lf_sha256'], f'Source changed: {name}; use a new run')
    rows = read_jsonl(run / 'inputs.jsonl')
    validate_inputs(rows, config)
    return dict(manifest=manifest, config=config, inputs=rows)


@contextmanager
def stage_lock(run, stage, resume=False):
    """No concurrent writers, no implicit restart, no deletion of old artifacts."""
    run = Path(run)
    require(not marker_path(run, stage).exists(), f'{stage} already complete; do not overwrite')
    partial = run / 'parts' / stage
    require(resume or not partial.exists(), f'Partial {stage} exists; inspect failure then use --resume')
    lock = marker_path(run, stage).with_suffix('.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    # Kernel locks are released on SIGKILL/process death. The file stays in
    # place so a stale pathname never blocks an otherwise valid --resume.
    import os
    import socket
    with lock.open('a+b') as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b'\n')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError(f'Another process owns this stage lock: {lock}') from exc
        try:
            stream.seek(1)
            stream.truncate()
            stream.write(canonical(dict(pid=os.getpid(), host=socket.gethostname(), started_utc=utc_now())).encode())
            stream.flush()
            partial.mkdir(parents=True, exist_ok=True)
            yield partial
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def part_write(path, payload, manifest_sha256):
    write_json(path, dict(manifest_sha256=manifest_sha256,
                          payload_sha256=object_hash(payload), payload=payload))


def part_read(path, manifest_sha256):
    record = read_json(path)
    require(record['manifest_sha256'] == manifest_sha256, 'Partial artifact belongs to another manifest')
    require(record['payload_sha256'] == object_hash(record['payload']), 'Partial artifact corrupted')
    return record['payload']


def split_development(rows, validation_images, counts, seed):
    """Pick image groups without ever examining labels/predictions/coverage."""
    groups = {}
    for row in rows:
        require(row['split'] == 'dev', 'Only the frozen old dev pool is eligible')
        require(row['image_key'] not in validation_images, 'Development/validation image leakage')
        groups.setdefault(row['image_key'], []).append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    selected, cursor = [], 0
    for split in ('calibration', 'evaluation'):
        chosen = []
        while len(chosen) < counts[split]:
            require(cursor < len(keys), 'Insufficient image-disjoint development pool')
            group = sorted(groups[keys[cursor]], key=lambda r: r['id'])
            cursor += 1
            chosen.extend(group[:counts[split] - len(chosen)])
        selected.extend((split, row) for row in chosen)
    return selected


def arm_specs(selection, config):
    for gate in config['gates']:
        for kind, seed, head in [('sinkS', None, selection['sink_head'])] + [
                ('random', seed, selection['random_heads'][str(seed)]) for seed in config['random_seeds']]:
            layer = selection['layer']
            key = f'{kind}_L{layer}_g{gate:g}' + (f'_s{seed}' if seed is not None else '')
            yield dict(arm_id=key, kind=kind, layer=layer, gate=gate, seed=seed,
                       head=head, k=selection['k'])
