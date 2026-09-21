"""Convert genuine UMD annotations and audit server inputs. No GPU or training.

PREP_STAGE=convert|audit; see REF_PLINEAR_PREPARE.md. Writes new files only.
"""
from collections import Counter
import math
import os
from pathlib import Path
import pickle
import pickletools

from humanref_pipeline import ROOT, digest, save_json, validate_boxes
from ref_plinear_data import image_key, read_json, split_train_dev
from ref_plinear import SPLIT_SEED

RAW = Path(os.environ.get('PREP_RAW', ROOT / 'data/refcocog'))
OUT = Path(os.environ.get('PREP_OUT', ROOT / 'data/refcocog_plinear_umd_v1'))
REFERENCE = Path(os.environ.get('PREP_REFERENCE', ROOT / 'results/ref_full_d30_refcocog_validation'))
IMAGES = Path(os.environ.get('PREP_IMAGES', ROOT / 'data/coco2014'))
PROPOSALS = Path(os.environ.get('PREP_PROPOSALS', ROOT / 'wedetect_ref/eval_grounding/eval_refcoco/refcoco_proposals_all.json'))
AUDIT_OUT = Path(os.environ.get('PREP_AUDIT_OUT', ROOT / 'results/ref_plinear_data_audit_v1'))
TRAIN_N, DEV_N = 5000, 1000


def forward_algorithm(box: list) -> list:
    """COCO xywh -> unclipped xyxy. Clipping remains in the inference pipeline."""
    assert isinstance(box, list) and len(box) == 4
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box)
    x, y, w, h = box
    assert w > 0 and h > 0, f'Invalid GT size: {box}'
    result = [x, y, x + w, y + h]
    assert len(result) == 4 and all(math.isfinite(v) for v in result)
    return result


def read_primitive_pickle(path):
    """REFER files here contain only built-in data, never callable constructors.

    Reject ALL executable/extension/persistent opcodes BEFORE unpickling; unknown
    opcodes also fail. No find_class bypass or allowlisted external classes.
    """
    allowed = set(('PROTO FRAME STOP MARK POP POP_MARK DUP NONE NEWTRUE NEWFALSE '
        'INT BININT BININT1 BININT2 LONG LONG1 LONG4 FLOAT BINFLOAT '
        'STRING BINSTRING SHORT_BINSTRING UNICODE BINUNICODE SHORT_BINUNICODE BINUNICODE8 '
        'BINBYTES SHORT_BINBYTES BINBYTES8 EMPTY_LIST LIST APPEND APPENDS '
        'EMPTY_TUPLE TUPLE TUPLE1 TUPLE2 TUPLE3 EMPTY_DICT DICT SETITEM SETITEMS '
        'EMPTY_SET ADDITEMS FROZENSET PUT BINPUT LONG_BINPUT GET BINGET LONG_BINGET MEMOIZE').split())
    data = Path(path).read_bytes()
    last = None
    for op, _, position in pickletools.genops(data):
        assert op.name in allowed, f'Unsafe/unsupported pickle opcode: {op.name} at {position}'
        last = (op.name, position)
    assert last == ('STOP', len(data) - 1), 'Missing STOP or trailing pickle data'
    return pickle.loads(data, encoding='latin1')


def convert_annotations(instances, refs):
    """Train: all sentences; validation: first per reference, original order."""
    images = {r['id']: r for r in instances['images']}
    anns = {r['id']: r for r in instances['annotations']}
    assert len(images) == len(instances['images']) and len(anns) == len(instances['annotations'])
    assert refs and len({r['ref_id'] for r in refs}) == len(refs)
    split_images = {s: set() for s in ('train', 'val', 'test')}
    counts = {s: dict(references=0, expressions=0) for s in split_images}
    train, validation, sent_ids = [], [], set()
    for r in refs:
        split = r['split']
        assert split in split_images, split
        a, image = anns[r['ann_id']], images[r['image_id']]
        assert a['image_id'] == r['image_id'] and a['category_id'] == r['category_id']
        assert image['width'] > 0 and image['height'] > 0
        name = image['file_name']  # ref.file_name has an ANN suffix: not an image!
        assert name == f'COCO_train2014_{r["image_id"]:012d}.jpg'
        name = 'train2014/' + name
        box = forward_algorithm(a['bbox'])
        sentences = r['sentences']
        assert sentences and [s['sent_id'] for s in sentences] == r['sent_ids']
        split_images[split].add(r['image_id'])
        counts[split]['references'] += 1
        counts[split]['expressions'] += len(sentences)
        for j, sentence in enumerate(sentences):
            sid = sentence['sent_id']
            assert isinstance(sid, int) and sid not in sent_ids
            sent_ids.add(sid)
            query = sentence['sent']  # processed text, NOT raw/case/punctuation variant
            assert isinstance(query, str) and query.strip()
            if split == 'test' or (split == 'val' and j != 0):
                continue  # reserve test; never export it as probe supervision
            identifier = f'refcocog_train_{sid}' if split == 'train' else f'refcocog_val_{len(validation)}'
            record = dict(id=identifier, image=name,
                conversations=[dict(**{'from': 'human'}, value='<image>'),
                               dict(**{'from': 'gpt'}, value=query)],
                bounding_boxes=[box], ref_id=r['ref_id'], ann_id=r['ann_id'], sent_id=sid)
            (train if split == 'train' else validation).append(record)
    for i, s in enumerate(split_images):
        for t in list(split_images)[i+1:]:
            assert not (split_images[s] & split_images[t]), f'Image leakage {s}/{t}; not UMD!'
    assert train and validation and split_images['test']
    for s in counts:
        counts[s]['images'] = len(split_images[s])
    inventory = [dict(image='train2014/' + images[i]['file_name'], image_id=i,
                      width=images[i]['width'], height=images[i]['height'], split=s)
                 for s in ('train', 'val') for i in sorted(split_images[s])]
    return train, validation, counts, inventory


def check_validation(rows, records):
    assert len(rows) == len(records) > 0
    max_error = 0.
    for i, (a, b) in enumerate(zip(rows, records)):
        assert b['passed'] and a['id'] == b['id'], f'Validation ID/order mismatch at {i}'
        assert a['image'] == b['image_name'], f'Validation image mismatch at {i}'
        assert a['conversations'][1]['value'] == b['query'], f'Validation expression mismatch at {i}'
        assert len(a['bounding_boxes']) == len(b['answer_boxes']) == 1
        validate_boxes(a['bounding_boxes'])
        validate_boxes(b['answer_boxes'])
        error = max(abs(x-y) for x, y in zip(a['bounding_boxes'][0], b['answer_boxes'][0]))
        assert error <= 1e-6, f'Validation GT mismatch at {i}: {error}'
        max_error = max(max_error, error)
    return dict(rows=len(rows), same_order=True, same_ids=True, same_images=True,
                same_queries=True, max_box_abs_error=max_error, box_tolerance=1e-6)


def convert():
    assert not OUT.exists(), f'Refusing to overwrite {OUT}; set a new PREP_OUT'
    refs_path = RAW / 'refs(umd).p'  # fixed: never select google by filename heuristics
    print('Reading primitive-only UMD pickle and instances...', flush=True)
    train, validation, counts, inventory = convert_annotations(
        read_json(RAW / 'instances.json'), read_primitive_pickle(refs_path))
    manifest = read_json(REFERENCE / 'manifest.json')
    assert read_json(REFERENCE / 'summary.json')['status'] == 'PASSED' and manifest['full_split']
    files = sorted((REFERENCE / 'samples').glob('*.json'))
    hashes = read_json(REFERENCE / 'sample_hashes.json')
    assert {p.name for p in files} == set(hashes)
    assert all(digest(p) == hashes[p.name] for p in files), 'Changed previous validation results'
    records = [read_json(p) for p in files]
    assert [r['id'] for r in records] == manifest['sample_ids']
    checked = check_validation(validation, records)
    image_hashes = {image_key(name): sha for name, sha in manifest['image_sha256'].items()}
    for row in inventory:
        if row['split'] == 'val':
            row['reference_sha256'] = image_hashes[row['image_id']]
    outputs = {'refcocog_train.json': train, 'refcocog_validation_first.json': validation,
               'image_inventory.json': inventory}
    for name, data in outputs.items():
        save_json(OUT / name, data)
    save_json(OUT / 'conversion.json', dict(status='PASSED', split='umd', counts=counts,
        train_expressions=len(train), validation_expressions=len(validation), validation_match=checked,
        source_sha256={'refs(umd).p': digest(refs_path), 'instances.json': digest(RAW / 'instances.json')},
        converter_sha256=digest(__file__),
        output_sha256={name: digest(OUT / name) for name in outputs},
        reference_manifest_sha256=digest(REFERENCE / 'manifest.json'),
        reference_samples_sha256=digest(REFERENCE / 'sample_hashes.json'),
        image_root_relative='data/coco2014', image_prefix='train2014/',
        rule='UMD train all sentences; val first per ref; test reserved. No candidates generated. '
             'Converted val is a CHECK artifact, not a replacement for author val JSON.'))
    print(f'Conversion PASS: train={len(train)}, val={len(validation)}, val_match={checked}', flush=True)
    print(f'Output: {OUT.resolve()}\nNext: PREP_STAGE=audit on the server.', flush=True)


def candidate_entry(entry):
    assert isinstance(entry, list)
    paired = len(entry) == 2 and isinstance(entry[0], list) and (not entry[0] or isinstance(entry[0][0], list))
    boxes = entry[0] if paired else entry
    validate_boxes(boxes)
    if paired:
        scores = entry[1]
        assert isinstance(scores, list) and len(scores) == len(boxes)
        assert all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores)
    return boxes


def audit_proposals(names, source):
    """Normalize only filename keys. Never add/reorder/clip/filter boxes or scores."""
    assert isinstance(source, dict)
    required_ids = {image_key(name) for name in names}
    by_id = {}
    for name, entry in source.items():
        key = image_key(name)
        if key not in required_ids:
            continue
        candidate_entry(entry)
        if key in by_id:
            assert by_id[key][1] == entry, f'Conflicting candidate aliases: {name} vs {by_id[key][0]}'
        else:
            by_id[key] = (name, entry)
    normalized, missing, empty, counts, aliases = {}, [], [], Counter(), []
    for name in sorted(names):
        key = image_key(name)
        if key not in by_id:
            missing.append(name)
            continue
        source_name, entry = by_id[key]
        boxes = candidate_entry(entry)
        if not boxes:
            empty.append(name)
            continue
        normalized[name] = entry
        counts[len(boxes)] += 1
        if source_name != name:
            aliases.append(dict(source=source_name, target=name))
    return normalized, dict(required=len(names), covered=len(normalized), missing=missing,
                            empty=empty, box_count_histogram=dict(counts), renamed_keys=aliases)


def probe_rows(rows):
    return [dict(id=r['id'], image_name=r['image'], referring=r['conversations'][1]['value'],
                 answer_boxes=r['bounding_boxes']) for r in rows]


def audit():
    assert not AUDIT_OUT.exists(), f'Refusing to overwrite {AUDIT_OUT}; set PREP_AUDIT_OUT'
    conversion = read_json(OUT / 'conversion.json')
    assert conversion['status'] == 'PASSED' and conversion['split'] == 'umd'
    assert conversion['converter_sha256'] == digest(__file__), 'Converter changed; review and regenerate'
    for name, sha in conversion['output_sha256'].items():
        assert digest(OUT / name) == sha, f'Changed converted artifact: {name}'
    train = read_json(OUT / 'refcocog_train.json')
    validation = read_json(OUT / 'refcocog_validation_first.json')
    selected = split_train_dev(probe_rows(train), probe_rows(validation), TRAIN_N, DEV_N, SPLIT_SEED)
    required = {r['image'] for r in train}
    selected_names = {r['image_name'] for rows in selected.values() for r in rows}
    inventory = read_json(OUT / 'image_inventory.json')
    missing_images, changed_images = [], []
    for r in inventory:
        path = IMAGES / r['image']
        if not path.is_file() or path.stat().st_size == 0:
            missing_images.append(r['image'])
        elif r['split'] == 'val' and digest(path) != r['reference_sha256']:
            changed_images.append(r['image'])
    print(f'Images checked: {len(inventory)}; missing/empty={len(missing_images)}; '
          f'changed validation images={len(changed_images)}', flush=True)
    source_exists = PROPOSALS.is_file()
    source = read_json(PROPOSALS) if source_exists else {}  # Explicit missing-source diagnostic, NOT training fallback.
    normalized, proposals = audit_proposals(required, source)
    ready = not missing_images and not changed_images and proposals['covered'] == proposals['required']
    status = 'READY' if ready else 'NEEDS_INPUTS'
    save_json(AUDIT_OUT / 'missing_images.json', missing_images)
    save_json(AUDIT_OUT / 'changed_validation_images.json', changed_images)
    save_json(AUDIT_OUT / 'missing_train_proposals.json', sorted(proposals['missing'] + proposals['empty']))
    save_json(AUDIT_OUT / 'proposal_coverage.json', proposals)
    save_json(AUDIT_OUT / 'planned_selection.json', dict(seed=SPLIT_SEED, rows=selected,
        note='Preview only; P-linear preflight independently recomputes the same image-disjoint selection.'))
    # Publish ready candidate data only when every train annotation can be loaded.
    # A partial file would fail the current P-linear loader or tempt filtering by availability.
    candidate_file = None
    if proposals['covered'] == proposals['required']:
        path = AUDIT_OUT / 'train_proposals.json'
        save_json(path, normalized)
        candidate_file = dict(path=str(path.resolve()), sha256=digest(path))
    summary = dict(status=status, images_root=str(IMAGES.resolve()), image_files_checked=len(inventory),
        missing_or_empty_images=len(missing_images), changed_validation_images=len(changed_images),
        image_check='All required train/val paths nonempty; val SHA matches prior run; train JPEG decoding NOT checked',
        proposals_source=str(PROPOSALS.resolve()), proposals_source_exists=source_exists,
        proposals_source_sha256=digest(PROPOSALS) if source_exists else None,
        train_images_required=len(required), train_images_covered=len(normalized),
        train_images_missing_proposals=len(proposals['missing']), train_images_empty_proposals=len(proposals['empty']),
        selected_images_required=len(selected_names), selected_images_covered=len(selected_names & set(normalized)),
        candidate_file=candidate_file, train_annotations=str((OUT / 'refcocog_train.json').resolve()),
        conversion_sha256=digest(OUT / 'conversion.json'), source_sha256=digest(__file__),
        gt_insertion=False, proposal_generation=False, proposal_order_changed=False)
    save_json(AUDIT_OUT / 'summary.json', summary)
    with (AUDIT_OUT / 'summary.md').open('x', encoding='utf-8') as stream:
        stream.write(f'# P-linear input audit: {status}\n\n'
            f'- Missing/empty images: {len(missing_images)} / {len(inventory)}\n'
            f'- Changed validation image hashes: {len(changed_images)}\n'
            f'- Train candidate coverage: {len(normalized)} / {len(required)} images\n'
            f'- Planned 5000+1000 expression candidate coverage: {summary["selected_images_covered"]} / {len(selected_names)} images\n\n'
            'No GT boxes inserted; no samples dropped based on candidate availability. '
            'READY checks file presence, not full training-image decode. '
            'If proposals are missing, extract fixed Uni candidates before P-linear preflight.\n')
    print(f'{status}: candidates {len(normalized)}/{len(required)}; report {AUDIT_OUT / "summary.md"}', flush=True)
    if not ready:
        raise SystemExit(2)  # Expected incomplete-input report; never pretend training is ready.


if __name__ == '__main__':
    assert __debug__, 'Do not use python -O'
    stage = os.environ.get('PREP_STAGE', 'convert')
    assert stage in ('convert', 'audit')
    if stage == 'convert':
        convert()
    else:
        audit()
