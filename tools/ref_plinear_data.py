"""Image-disjoint selection and verified, bounded-memory probe input batches."""
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import re

from humanref_pipeline import digest


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def image_key(name):
    """Match COCO aliases, e.g. train2014/COCO_train2014_000000000123.jpg and 123.jpg."""
    stem = Path(name.replace('\\', '/')).stem
    match = re.fullmatch(r'(?:COCO_(?:train|val|test)\d{4}_)?(\d+)', stem)
    assert match, f'Expected COCO filename, cannot safely check image split: {name}'
    return int(match[1])


def split_train_dev(rows, validation, train_n, dev_n, seed):
    assert train_n > 0 and dev_n > 0
    assert rows and validation
    assert len({r['id'] for r in rows}) == len(rows)
    assert all('_train_' in r['id'] for r in rows), 'Require processed RefCOCOg TRAIN IDs'
    assert not ({r['id'] for r in rows} & {r['id'] for r in validation})
    assert not ({image_key(r['image_name']) for r in rows} &
                {image_key(r['image_name']) for r in validation}), 'TRAIN/validation image leakage'
    groups = defaultdict(list)
    for row in rows:
        groups[image_key(row['image_name'])].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    cursor = 0
    selected = {}
    # Reserve complete images for dev, then train. Trim only the last group's
    # expressions; unused expressions from that image NEVER enter the other set.
    for split, count in [('dev', dev_n), ('train', train_n)]:
        chosen = []
        while len(chosen) < count:
            assert cursor < len(keys), 'Not enough image-disjoint training expressions'
            group = sorted(groups[keys[cursor]], key=lambda r: r['id'])
            chosen.extend(group[:count - len(chosen)])
            cursor += 1
        selected[split] = chosen
    assert not ({image_key(r['image_name']) for r in selected['train']} &
                {image_key(r['image_name']) for r in selected['dev']})
    return selected


def open_cache(root, expected_stage):
    root = Path(root)
    complete = read_json(root / 'COMPLETE.json')
    assert complete['status'] == 'PASSED' and complete['stage'] == expected_stage
    assert digest(root / 'manifest.json') == complete['manifest_sha256']
    assert digest(root / 'index.json') == complete['index_sha256']
    manifest, entries = read_json(root / 'manifest.json'), read_json(root / 'index.json')
    assert len(entries) == complete['samples'] > 0
    assert len({e['id'] for e in entries}) == len(entries)
    assert len({e['file'] for e in entries}) == len(entries)
    assert [e['id'] for e in entries] == manifest['sample_ids']
    return manifest, entries


def read_sample(root, entry):
    import torch
    data = (Path(root) / entry['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry['sha256'], f'Changed cache: {entry["file"]}'
    sample = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    assert sample['id'] == entry['id'] and sample['split'] == entry['split']
    assert sample['image_key'] == entry['image_key']
    assert sample['features'].dtype == torch.bfloat16 and sample['features'].ndim == 3
    k, n, d = sample['features'].shape
    assert n > 0 and d > 0 and k == len(sample['depths'])
    assert sample['overlaps'].shape == sample['labels'].shape == (n,)
    assert torch.isfinite(sample['features']).all() and torch.isfinite(sample['overlaps']).all()
    assert ((sample['overlaps'] >= 0) & (sample['overlaps'] <= 1)).all()
    expected = torch.where(sample['overlaps'] > .5, sample['overlaps'], 0)
    assert torch.equal(sample['labels'], expected), 'Invalid IoU soft targets'
    return sample


def batches(root, entries, batch_size, device):
    import torch
    assert batch_size > 0 and entries
    for start in range(0, len(entries), batch_size):
        samples = [read_sample(root, e) for e in entries[start:start + batch_size]]
        k, _, d = samples[0]['features'].shape
        n = max(s['features'].shape[1] for s in samples)
        x = torch.zeros(len(samples), k, n, d, dtype=torch.float32)
        y, overlaps = torch.zeros(len(samples), n), torch.zeros(len(samples), n)
        valid = torch.zeros(len(samples), n, dtype=torch.bool)
        for i, s in enumerate(samples):
            ni = s['features'].shape[1]
            assert s['features'].shape == (k, ni, d) and s['depths'] == samples[0]['depths']
            x[i, :, :ni] = s['features'].float()
            y[i, :ni], overlaps[i, :ni], valid[i, :ni] = s['labels'], s['overlaps'], True
        yield x.to(device), y.to(device), valid.to(device), overlaps.to(device), samples
