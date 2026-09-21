"""Random stdlib fixtures; no GPU/model/download. Never fabricate real training data."""
from copy import deepcopy
from pathlib import Path
import pickle
import random
import tempfile
from unittest.mock import patch

from humanref_pipeline import digest, save_json
import ref_plinear_prepare as prep
from ref_e0_data import load_rec_annotations


def must_fail(fn, kind=AssertionError):
    try:
        fn()
    except kind:
        return
    raise AssertionError('Expected invalid input to fail')


def fixture():
    rng = random.Random(23)
    images, anns, refs = [], [], []
    for i in range(11):
        images.append(dict(id=i, file_name=f'COCO_train2014_{i:012d}.jpg', width=640, height=480))
        box = [rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(1, 200), rng.uniform(1, 200)]
        assert len(box) == 4
        xyxy = prep.forward_algorithm(box)
        assert len(xyxy) == 4 and xyxy[2]-xyxy[0] > 0 and xyxy[3]-xyxy[1] > 0
        anns.append(dict(id=100+i, image_id=i, category_id=1, bbox=box))
        refs.append(dict(ref_id=i, image_id=i, ann_id=100+i, category_id=1,
            file_name=f'WRONG_ROI_NAME_{i}.jpg', split='train' if i < 8 else ('val' if i < 10 else 'test'),
            sentences=[dict(sent_id=i*2+j, sent=f'processed expression {i} {j}', raw='DO NOT USE RAW') for j in range(2)],
            sent_ids=[i*2, i*2+1]))
    return dict(images=images, annotations=anns), refs


def test_conversion():
    instances, refs = fixture()
    train, val, counts, inventory = prep.convert_annotations(instances, refs)
    assert len(train) == 16 and len(val) == 2 and len(inventory) == 10
    assert counts['val'] == dict(references=2, expressions=4, images=2)
    assert train[0]['image'] == 'train2014/COCO_train2014_000000000000.jpg'
    assert train[0]['conversations'][1]['value'] == 'processed expression 0 0'
    assert val[1]['id'] == 'refcocog_val_1' and val[1]['sent_id'] == 18
    records = [dict(id=r['id'], image_name=r['image'], query=r['conversations'][1]['value'],
                    answer_boxes=r['bounding_boxes'], passed=True) for r in val]
    assert prep.check_validation(val, records)['max_box_abs_error'] == 0
    changed = deepcopy(records)
    changed[0]['query'] = 'raw instead of processed'
    must_fail(lambda: prep.check_validation(val, changed))
    must_fail(lambda: prep.check_validation(val, records[::-1]))
    changed = deepcopy(refs)
    changed[8]['image_id'], changed[8]['ann_id'] = 0, 100
    must_fail(lambda: prep.convert_annotations(instances, changed))
    for box in ([0, 1, 2], [0, 0, 0, 3], [0, 0, -2, 3], [0, float('nan'), 2, 3]):
        must_fail(lambda: prep.forward_algorithm(box))
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory) / 'data.p'
        for protocol in (0, 2, 4, 5):
            p.write_bytes(pickle.dumps(refs, protocol=protocol))
            assert prep.read_primitive_pickle(p) == refs
        p.write_bytes(b'cos\nsystem\n.')  # GLOBAL opcode rejected before any unpickle
        must_fail(lambda: prep.read_primitive_pickle(p))
        p.write_bytes(pickle.dumps(refs) + b'extra')
        must_fail(lambda: prep.read_primitive_pickle(p))
    print('Random xywh shapes / text+filename mapping / UMD leakage / pickle safety: PASS')


def test_proposals():
    names = {'train2014/COCO_train2014_000000000001.jpg', 'train2014/COCO_train2014_000000000002.jpg'}
    boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
    source = {'1.jpg': boxes, 'COCO_train2014_000000000002.jpg': [boxes, [.8, .3]]}
    normalized, report = prep.audit_proposals(names, source)
    assert report['covered'] == report['required'] == 2 and len(report['renamed_keys']) == 2
    assert normalized['train2014/COCO_train2014_000000000001.jpg'] == boxes
    assert normalized['train2014/COCO_train2014_000000000002.jpg'] == [boxes, [.8, .3]]
    bad = dict(source, **{'000000000001.jpg': [[0, 0, 50, 50]]})
    must_fail(lambda: prep.audit_proposals(names, bad))
    _, missing = prep.audit_proposals(names, {'1.jpg': []})
    assert len(missing['empty']) == len(missing['missing']) == 1
    must_fail(lambda: prep.candidate_entry([boxes, [.5]]))
    must_fail(lambda: prep.candidate_entry([[10, 0, 0, 5]]))
    print('Bare/paired candidates / key aliases / preservation / conflicts / missing: PASS')


def test_workflow():
    instances, refs = fixture()
    _, val, _, _ = prep.convert_annotations(instances, refs)
    with tempfile.TemporaryDirectory(prefix='ref-prepare-test-') as directory:
        root = Path(directory)
        raw, out, reference = root / 'raw', root / 'converted', root / 'reference'
        images, proposals = root / 'images', root / 'proposals.json'
        save_json(raw / 'instances.json', instances)
        (raw / 'refs(umd).p').write_bytes(pickle.dumps(refs, protocol=2))
        image_hashes = {}
        for r in instances['images']:
            path = images / 'train2014' / r['file_name']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f'SYNTHETIC_NON_JPEG_{r["id"]}'.encode())
            image_hashes[str(path)] = digest(path)
        hashes = {}
        for i, r in enumerate(val):
            path = reference / 'samples' / f'{i:05d}.json'
            save_json(path, dict(id=r['id'], image_name=r['image'], query=r['conversations'][1]['value'],
                                answer_boxes=r['bounding_boxes'], passed=True))
            hashes[path.name] = digest(path)
        save_json(reference / 'sample_hashes.json', hashes)
        save_json(reference / 'summary.json', dict(status='PASSED'))
        save_json(reference / 'manifest.json', dict(full_split=True, sample_ids=[r['id'] for r in val], image_sha256=image_hashes))
        # Deliberately not GT candidates. Tests never use oracle insertion as fallback.
        source = {f'{i}.jpg': [[0, 0, 1, 1], [1, 1, 2, 2]] for i in range(8)}
        save_json(proposals, source)
        with patch.multiple(prep, RAW=raw, OUT=out, REFERENCE=reference, IMAGES=images,
                            PROPOSALS=proposals, AUDIT_OUT=root / 'audit_ready', TRAIN_N=3, DEV_N=2):
            prep.convert()
            must_fail(prep.convert)  # no overwrite
            prep.audit()
            report = prep.read_json(root / 'audit_ready/summary.json')
            assert report['status'] == 'READY' and report['train_images_covered'] == 8
            rows, boxes = load_rec_annotations(out / 'refcocog_train.json', root / 'audit_ready/train_proposals.json')
            assert len(rows) == 16 and len(boxes) == 8  # real P-linear input loader accepts artifacts
            with patch.multiple(prep, PROPOSALS=root / 'nonexistent.json', AUDIT_OUT=root / 'audit_missing'):
                must_fail(prep.audit, SystemExit)
                assert prep.read_json(root / 'audit_missing/summary.json')['status'] == 'NEEDS_INPUTS'
                assert not (root / 'audit_missing/train_proposals.json').exists()
            (images / val[0]['image']).write_bytes(b'CHANGED')
            with patch.object(prep, 'AUDIT_OUT', root / 'audit_changed'):
                must_fail(prep.audit, SystemExit)
                assert prep.read_json(root / 'audit_changed/summary.json')['changed_validation_images'] == 1
    print('Convert -> audit -> actual P-linear loader / incomplete reports / val image hashes: PASS')


if __name__ == '__main__':
    assert __debug__
    test_conversion()
    test_proposals()
    test_workflow()
