"""Regenerate Uni-Base top100 for the FROZEN train/dev images. Single GPU.

Use UNI_N=6 for extractor smoke, UNI_N=0 for the full frozen set.
UNI_RESUME=1 resumes verified per-image records only on the same environment.
"""
from collections import Counter
import ast
import os
from pathlib import Path
import platform

from humanref_pipeline import ROOT, digest, iou, save_json, validate_boxes
from ref_e0 import package_version
from ref_plinear import SPLIT_SEED
from ref_plinear_data import read_json
from ref_plinear_inputs import load_rec_rows, select_rows

DATA = Path(os.environ.get('UNI_DATA', ROOT / 'data/refcocog_plinear_umd_v1'))
SELECTION = Path(os.environ.get('UNI_SELECTION', ROOT / 'results/ref_plinear_data_audit_v1/planned_selection.json'))
IMAGES = Path(os.environ.get('UNI_IMAGES', ROOT / 'data/coco2014'))
CHECKPOINT = Path(os.environ.get('UNI_CHECKPOINT', ROOT / 'checkpoints/wedetect_base_uni.pth'))
OUTPUT = Path(os.environ.get('UNI_OUT', ROOT / 'results/ref_plinear_uni_k100_v1'))
LIMIT = int(os.environ.get('UNI_N', '0'))
RESUME = os.environ.get('UNI_RESUME', '0') == '1'
TOPK = 100
ALLOWED_UNUSED = {'backbone.norm.weight', 'backbone.norm.bias', 'backbone.head.weight', 'backbone.head.bias'}


def frozen_inputs(data, selection):
    conversion = read_json(data / 'conversion.json')
    assert conversion['status'] == 'PASSED' and conversion['split'] == 'umd'
    for name, sha in conversion['output_sha256'].items():
        assert digest(data / name) == sha, f'Converted input changed: {name}'
    train = load_rec_rows(data / 'refcocog_train.json')
    val = load_rec_rows(data / 'refcocog_validation_first.json')
    lock = read_json(selection)
    assert len(lock['rows']['train']) == 5000 and len(lock['rows']['dev']) == 1000
    selected = select_rows(train, val, 5000, 1000, SPLIT_SEED, selection)
    names = sorted({r['image_name'] for rows in selected.values() for r in rows})
    assert names
    inventory = {r['image']: r for r in read_json(data / 'image_inventory.json')}
    assert all(inventory[name]['split'] == 'train' for name in names)
    return selected, names, inventory


def check_generator_protocol(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SimpleYOLOWorldDetector')
    head = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'head_predict')
    calls = [n for n in ast.walk(head) if isinstance(n, ast.Call)]
    nms = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'batched_nms']
    filters = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'filter_scores_and_topk']
    assert len(nms) == len(filters) == 1, 'Uni postprocessing structure changed; review protocol'
    assert isinstance(nms[0].args[3], ast.Constant) and nms[0].args[3].value == .7
    assert all(isinstance(n, ast.Constant) for n in filters[0].args[1:3])
    assert [n.value for n in filters[0].args[1:3]] == [0., 30000]


def write_or_verify(path, data):
    if path.exists():
        assert RESUME and read_json(path) == data, f'Existing artifact differs: {path}'
    else:
        save_json(path, data)


def validate_record(record, name, image_sha256, width, height):
    assert record['image_name'] == name and record['image_sha256'] == image_sha256
    assert record['width'] == width and record['height'] == height
    boxes, scores = record['boxes'], record['scores']
    assert 0 < len(boxes) <= TOPK and len(boxes) == len(scores)
    validate_boxes(boxes)
    assert all(0 <= s <= 1 for s in scores) and scores == sorted(scores, reverse=True)
    assert all(0 <= b[0] <= b[2] <= width and 0 <= b[1] <= b[3] <= height for b in boxes)
    return [boxes, scores]


def summarize_coverage(selected, proposals, inventory):
    """GT used AFTER extraction, only for a diagnostic; never selection or repair."""
    output = {}
    for split, rows in selected.items():
        complete = [r for r in rows if r['image_name'] in proposals]
        maxima = []
        for r in complete:
            image = inventory[r['image_name']]
            width, height = image['width'], image['height']
            gt = [max(0, min(limit, value)) for value, limit in zip(r['answer_boxes'][0], [width, height, width, height])]
            maxima.append(max(iou(b, gt) for b in proposals[r['image_name']][0]))
        output[split] = dict(evaluated_expressions=len(maxima), planned_expressions=len(rows),
            hits_iou50=sum(v >= .5 for v in maxima), hits_iou75=sum(v >= .75 for v in maxima),
            coverage_iou50=sum(v >= .5 for v in maxima) / len(maxima) if maxima else None,
            note='Diagnostics only; all uncovered expressions retained with zero IoU-soft targets.')
    return output


def run():
    assert __debug__ and int(os.environ.get('WORLD_SIZE', '1')) == 1, 'Use python, one GPU, not multi-rank torchrun'
    assert LIMIT >= 0
    assert not OUTPUT.exists() or RESUME, f'Refusing to overwrite {OUTPUT}; new UNI_OUT or explicit UNI_RESUME=1'
    if RESUME:
        assert (OUTPUT / 'manifest.json').is_file(), 'Resume needs original manifest'
    selected, all_names, inventory = frozen_inputs(DATA, SELECTION)
    assert LIMIT <= len(all_names)
    names = all_names if LIMIT == 0 else all_names[:LIMIT]
    print(f'Fixed Uni-Base extraction: {len(names)}/{len(all_names)} images; smoke={LIMIT != 0}', flush=True)
    assert CHECKPOINT.is_file(), f'Missing Uni checkpoint: {CHECKPOINT}'
    check_generator_protocol(ROOT / 'generate_proposal.py')
    image_hashes = {}
    for name in names:
        path = IMAGES / name
        assert path.is_file() and path.stat().st_size > 0, f'Missing image: {path}'
        image_hashes[name] = digest(path)
    import torch
    import torchvision
    from PIL import Image
    from generate_proposal import SimpleYOLOWorldDetector
    from ref_plinear_uni_core import forward_algorithm, remap_checkpoint

    assert torch.cuda.is_available()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source_files = [Path(__file__), ROOT / 'generate_proposal.py', ROOT / 'vis.py',
        ROOT / 'tools/ref_plinear_uni_core.py', ROOT / 'tools/ref_plinear_inputs.py',
        ROOT / 'tools/ref_plinear_data.py', ROOT / 'tools/humanref_pipeline.py']
    manifest = dict(stage='P-linear-Uni', backbone='base', dtype='float32', input_size=[640, 640],
        num_prompts=256, prompt_dim=768, max_proposals=TOPK, pre_nms_topk=30000,
        score_threshold=0.0, nms_iou=.7, nms='original torchvision batched_nms by learned prompt label',
        preprocessing='original RGB /255, bilinear letterbox 640, fill114, original offsets/rescale/clipping',
        additional_nms=False, gt_insertion=False, sample_filtering=False,
        candidate_order='original NMS score descending; ties not re-sorted',
        smoke_only=LIMIT != 0, images=names, total_frozen_images=len(all_names),
        selection_sha256=digest(SELECTION), conversion_sha256=digest(DATA / 'conversion.json'),
        train_annotation_sha256=digest(DATA / 'refcocog_train.json'),
        validation_annotation_sha256=digest(DATA / 'refcocog_validation_first.json'),
        checkpoint_sha256=digest(CHECKPOINT), image_sha256=image_hashes,
        source_sha256={p.name: digest(p) for p in source_files},
        environment=dict(python=platform.python_version(), torch=str(torch.__version__),
            torchvision=str(torchvision.__version__), pillow=package_version('pillow'), numpy=package_version('numpy'),
            cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
            capability=list(torch.cuda.get_device_capability(0)), host=platform.node(),
            tf32=False, cudnn_benchmark=False, cudnn_deterministic=True))
    # No mixing checkpoint/selected set/source/dependencies/GPU model/host on resume.
    write_or_verify(OUTPUT / 'manifest.json', manifest)
    manifest_sha = digest(OUTPUT / 'manifest.json')
    (OUTPUT / 'samples').mkdir(exist_ok=True)
    model = SimpleYOLOWorldDetector(backbone_size='base', prompt_dim=768, num_prompts=256, num_proposals=TOPK)
    weights = remap_checkpoint(torch.load(CHECKPOINT, map_location='cpu', weights_only=True))
    info = model.load_state_dict(weights, strict=False)
    assert not info.missing_keys and set(info.unexpected_keys) <= ALLOWED_UNUSED, info
    write_or_verify(OUTPUT / 'loading_info.json', dict(missing_keys=info.missing_keys, unexpected_keys=info.unexpected_keys))
    del weights
    model = model.cuda().float().eval().requires_grad_(False)
    assert model.img_size == (640, 640) and model.num_proposals == TOPK
    assert not any(p.requires_grad for p in model.parameters())
    proposals, record_hashes = {}, {}
    expected_files = {f'{i:05d}.json' for i in range(len(names))}
    assert {p.name for p in (OUTPUT / 'samples').glob('*.json')} <= expected_files
    with torch.inference_mode():
        for i, name in enumerate(names):
            path, image_info = IMAGES / name, inventory[name]
            record_path = OUTPUT / 'samples' / f'{i:05d}.json'
            if record_path.exists():
                assert RESUME
                record = read_json(record_path)
                assert record['manifest_sha256'] == manifest_sha
                assert digest(path) == image_hashes[name]
                state = 'verified cached'
            else:
                assert digest(path) == image_hashes[name], f'Image changed during run: {name}'
                with Image.open(path) as source:
                    image = source.convert('RGB')  # decodes pixels; malformed images fail here
                assert image.size == (image_info['width'], image_info['height'])
                pred = model([image])
                assert len(pred) == 1
                boxes, scores = forward_algorithm(pred[0]['bboxes'], pred[0]['scores'], *image.size)
                record = dict(image_name=name, image_sha256=image_hashes[name], manifest_sha256=manifest_sha,
                    width=image.width, height=image.height, boxes=boxes.cpu().tolist(), scores=scores.cpu().tolist())
                # Exclusive per-image writes; interrupted/corrupt JSON is rejected, never silently repaired.
                save_json(record_path, record)
                del pred, boxes, scores
                state = 'extracted'
            proposals[name] = validate_record(record, name, image_hashes[name], image_info['width'], image_info['height'])
            record_hashes[record_path.name] = digest(record_path)
            print(f'Uni {i+1}/{len(names)} {name} {state} K={len(proposals[name][0])}', flush=True)
    assert set(proposals) == set(names)
    summary = dict(status='PASSED', images=len(names), total_frozen_images=len(all_names), smoke_only=LIMIT != 0,
        box_count_histogram={str(k): v for k, v in Counter(len(v[0]) for v in proposals.values()).items()},
        coverage=summarize_coverage(selected, proposals, inventory),
        note='Self-generated fixed training/dev candidates, not claimed identical to author training candidates. '
             'Author validation candidates remain unchanged. No Ref training/timing in this stage.')
    write_or_verify(OUTPUT / 'train_proposals.json', proposals)
    write_or_verify(OUTPUT / 'sample_hashes.json', record_hashes)
    write_or_verify(OUTPUT / 'summary.json', summary)
    artifacts = ['manifest.json', 'train_proposals.json', 'summary.json', 'sample_hashes.json', 'loading_info.json']
    write_or_verify(OUTPUT / 'COMPLETE.json', dict(status='PASSED', artifacts={p: digest(OUTPUT / p) for p in artifacts}))
    print(f'Completed {OUTPUT / "summary.json"}', flush=True)


if __name__ == '__main__':
    run()
