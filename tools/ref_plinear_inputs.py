"""Select expressions BEFORE requiring candidates; optional frozen audit split."""
import math
from pathlib import Path

from humanref_pipeline import digest, validate_boxes
from ref_plinear_data import read_json, split_train_dev


def load_rec_rows(path):
    records = read_json(path)
    assert isinstance(records, list) and records
    rows = []
    for r in records:
        query, gt = r['conversations'][1]['value'], r['bounding_boxes']
        assert isinstance(r['id'], str) and isinstance(r['image'], str)
        assert isinstance(query, str) and query.strip() and len(gt) == 1
        validate_boxes(gt)
        rows.append(dict(id=r['id'], image_name=r['image'], referring=query, answer_boxes=gt))
    assert len({r['id'] for r in rows}) == len(rows)
    return rows


def select_rows(train, validation, train_n, dev_n, seed, selection_path=None):
    if selection_path is None:
        return split_train_dev(train, validation, train_n, dev_n, seed)
    lock = read_json(selection_path)
    assert lock['seed'] == seed
    frozen = lock['rows']
    assert set(frozen) == {'train', 'dev'}
    expected = split_train_dev(train, validation, len(frozen['train']), len(frozen['dev']), seed)
    assert frozen == expected, 'Audit selection changed or annotations differ; never resample to fit candidates'
    assert 0 < train_n <= len(frozen['train']) and 0 < dev_n <= len(frozen['dev'])
    # Smoke takes prefixes WITHIN frozen splits, not a new split with fewer dev images.
    return dict(train=frozen['train'][:train_n], dev=frozen['dev'][:dev_n])


def load_selected_candidates(path, names):
    source = read_json(path)
    assert isinstance(source, dict) and names
    result = {}
    for name in sorted(set(names)):
        assert name in source, f'Missing SELECTED image candidates: {name}; no dropping/resampling/GT insertion'
        entry = source[name]
        assert isinstance(entry, list)
        paired = len(entry) == 2 and isinstance(entry[0], list) and (not entry[0] or isinstance(entry[0][0], list))
        boxes = entry[0] if paired else entry
        assert boxes, f'Empty candidates: {name}'
        validate_boxes(boxes)
        if paired:
            scores = entry[1]
            assert isinstance(scores, list) and len(scores) == len(boxes)
            assert all(isinstance(s, (float, int)) and math.isfinite(s) for s in scores)
        result[name] = dict(boxes=boxes)
    return result


def verify_uni_artifact(root, proposals, selected, train_ann, selection_path):
    """Check complete extraction provenance; never accept a six-image smoke run."""
    root = Path(root)
    complete = read_json(root / 'COMPLETE.json')
    assert complete['status'] == 'PASSED'
    for name, sha in complete['artifacts'].items():
        assert digest(root / name) == sha, f'Changed Uni artifact: {name}'
    assert digest(proposals) == complete['artifacts']['train_proposals.json']
    manifest = read_json(root / 'manifest.json')
    assert not manifest['smoke_only'], 'Uni smoke output cannot be used as the complete training candidates'
    assert selection_path is not None and digest(selection_path) == manifest['selection_sha256']
    assert digest(train_ann) == manifest['train_annotation_sha256']
    expected = {r['image_name'] for rows in selected.values() for r in rows}
    assert expected <= set(manifest['images']), 'Selected images absent from Uni extraction'
    assert set(read_json(proposals)) == set(manifest['images'])
    return dict(complete_sha256=digest(root / 'COMPLETE.json'), manifest_sha256=digest(root / 'manifest.json'))
