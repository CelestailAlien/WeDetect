"""Cached Uni -> Ref evaluation. See tools/HUMANREF_PIPELINE.md."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'wedetect_ref'))
sys.path.insert(0, str(ROOT / 'wedetect_ref/eval_grounding'))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_boxes(boxes):
    for b in boxes:
        assert len(b) == 4 and all(math.isfinite(x) for x in b), b
        assert b[2] >= b[0] and b[3] >= b[1], b


def load_annotations(path, limit):
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len({r['id'] for r in rows}) == len(rows), 'Duplicate annotation IDs'
    if limit:
        rows = rows[:limit]
    assert rows, 'No annotations'
    for r in rows:
        for key in ('id', 'image_name', 'referring', 'domain', 'sub_domain',
                    'answer_boxes', 'candidate_boxes'):
            assert key in r, key
        validate_boxes(r['answer_boxes'])
        assert (r['domain'] == 'rejection') == (len(r['answer_boxes']) == 0)
    return rows


def image_path(root, name):
    # The official loader repairs decomposed Unicode names; try both forms.
    for candidate in (name, unicodedata.normalize('NFD', name),
                      unicodedata.normalize('NFC', name)):
        p = Path(root) / candidate
        if p.is_file():
            return p
    raise FileNotFoundError(str(Path(root) / name))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently mixing/overwriting experiments.
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False)


def load_shards(directory, stage):
    paths = sorted(Path(directory).glob(f'{stage}.rank*.json'))
    assert paths, f'No {stage} shards in {directory}'
    docs = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    world = docs[0]['world_size']
    assert len(docs) == world and {d['rank'] for d in docs} == set(range(world)), 'Incomplete shards'
    assert all(d['meta'] == docs[0]['meta'] and d['world_size'] == world for d in docs), 'Mixed runs'
    records = [r for d in docs for r in d['records']]
    key = 'image_name' if stage == 'uni' else 'id'
    assert len({r[key] for r in records}) == len(records), 'Duplicate records'
    return {r[key]: r for r in records}, docs[0]['meta'], [digest(p) for p in paths]


def iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0.


def diagnose(gt, boxes, scores, threshold, match_iou):
    validate_boxes(gt)
    validate_boxes(boxes)
    assert len(boxes) == len(scores) and all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
    selected = [b for b, s in zip(boxes, scores) if s > threshold]
    covered = [any(iou(g, b) >= match_iou for b in boxes) for g in gt]
    recovered = [any(iou(g, b) >= match_iou for b in selected) for g in gt]
    n, c, h = len(gt), sum(covered), sum(recovered)
    if not n:
        category = 'correct_rejection' if not selected else 'false_positive_rejection'
    elif c == 0:
        category = 'no_target_covered'
    elif c < n:
        category = 'partial_target_coverage'
    elif h < n:
        category = 'covered_but_ref_missed'
    else:
        category = 'all_targets_recovered'
    fp = sum(not any(iou(g, b) >= match_iou for g in gt) for b in selected)
    return dict(category=category, gt_count=n, covered_targets=c,
                recovered_targets=h, proposal_missed_targets=n-c,
                ref_missed_covered_targets=c-h, selected_count=len(selected),
                unmatched_selected_boxes=fp,
                # Multi-target hits are geometric coverage, not one-to-one AP matching.
                selected_boxes=selected)


def aggregate(rows):
    pos = [r for r in rows if r['gt_count']]
    neg = [r for r in rows if not r['gt_count']]
    n = sum(r['gt_count'] for r in pos)
    c = sum(r['covered_targets'] for r in pos)
    h = sum(r['recovered_targets'] for r in pos)
    fully = [r for r in pos if r['covered_targets'] == r['gt_count']]
    ratio = lambda a, b: a / b if b else None
    counts = {}
    for r in rows:
        counts[r['category']] = counts.get(r['category'], 0) + 1
    return dict(samples=len(rows), positives=len(pos), negatives=len(neg), categories=counts,
                gt_targets=n, covered_targets=c, recovered_targets=h,
                proposal_missed_targets=n-c, ref_missed_covered_targets=c-h,
                target_coverage=ratio(c, n), target_recovery=ratio(h, n),
                recovery_given_covered_target=ratio(h, c),
                fully_covered_sample_rate=ratio(len(fully), len(pos)),
                all_targets_recovered_given_full_coverage=ratio(
                    sum(r['recovered_targets'] == r['gt_count'] for r in fully), len(fully)),
                rejection_accuracy=ratio(sum(r['selected_count'] == 0 for r in neg), len(neg)),
                unmatched_selected_boxes=sum(r['unmatched_selected_boxes'] for r in rows))


def run_uni(args, rows):
    import torch
    from generate_proposal import SimpleYOLOWorldDetector
    model = SimpleYOLOWorldDetector(args.backbone, prompt_dim=768,
                                    num_prompts=256, num_proposals=args.num_proposals)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    remapped = {}
    for key, value in ckpt.items():
        if 'backbone' in key:
            key = key.replace('backbone.image_model.model.', 'backbone.')
        if 'bbox_head' in key:
            for old, new in [('bbox_head.head_module.', 'bbox_head.'),
                             ('0.2.', '0.6.'), ('1.2.', '1.6.'), ('2.2.', '2.6.'),
                             ('1.bn', '4'), ('1.conv', '3'), ('0.bn', '1'), ('0.conv', '0')]:
                key = key.replace(old, new)
        assert key not in remapped, key
        remapped[key] = value
    incompatible = model.load_state_dict(remapped, strict=False)
    allowed = {'backbone.norm.weight', 'backbone.norm.bias', 'backbone.head.weight', 'backbone.head.bias'}
    assert not incompatible.missing_keys and set(incompatible.unexpected_keys) <= allowed, incompatible
    print(incompatible, flush=True)
    model = model.cuda().eval()
    names = list(dict.fromkeys(r['image_name'] for r in rows))[args.rank::args.world_size]
    records = []
    for index, name in enumerate(names):
        path = image_path(args.images, name)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = model([str(path)])[0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        boxes, scores = out['bboxes'].float().cpu().tolist(), out['scores'].float().cpu().tolist()
        validate_boxes(boxes)
        assert len(boxes) == len(scores) and len(boxes) <= args.num_proposals
        records.append(dict(image_name=name, boxes=boxes, scores=scores, seconds=seconds))
        if index % 25 == 0:
            print(f'uni rank {args.rank}: {index+1}/{len(names)}', flush=True)
    return records


def run_ref(args, rows, proposals):
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from models.vision_process import process_vision_info
    from models.qwen3vl_referring import Qwen3VLGroundingForConditionalGeneration
    model = Qwen3VLGroundingForConditionalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, attn_implementation=args.attention).cuda().eval()
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    object_id = processor.tokenizer.convert_tokens_to_ids('<object>')
    model.model.object_token_id = object_id
    records = []
    for index, ann in enumerate(rows[args.rank::args.world_size]):
        with Image.open(image_path(args.images, ann['image_name'])) as f:
            image = f.convert('RGB')
        width, height = image.size
        boxes = proposals[ann['image_name']]['boxes'][:args.num_proposals]
        boxes = [[max(0, min(width, b[0])), max(0, min(height, b[1])),
                  max(0, min(width, b[2])), max(0, min(height, b[3]))] for b in boxes]
        validate_boxes(boxes)
        torch.cuda.synchronize()
        start = time.perf_counter()
        scores = []
        if boxes:
            messages = [dict(role='user', content=[dict(type='image', image=image),
                        dict(type='text', text=f'Please detect the "{ann["referring"]}" in the image')]),
                        dict(role='assistant', content=[dict(type='text', text='<object>' * len(boxes))])]
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            inputs = processor(text=[processor.apply_chat_template(messages, tokenize=False)],
                               images=image_inputs, videos=video_inputs, return_tensors='pt',
                               padding=True, do_resize=False).to(model.device)
            positions = inputs['input_ids'] == object_id
            assert int(positions.sum()) == len(boxes), 'Object token count mismatch'
            with torch.inference_mode():
                pred = model(**inputs, bboxes=[torch.tensor(boxes, device=model.device, dtype=model.dtype)],
                             ori_shapes=[image.size], bboxes_id=object_id, image_inputs=image_inputs)
            scores = pred.logits.sigmoid()[positions].flatten().float().cpu().tolist()
            assert len(scores) == len(boxes)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        records.append(dict(id=ann['id'], image_name=ann['image_name'], boxes=boxes,
                            scores=scores, seconds=seconds))
        if index % 25 == 0:
            print(f'ref rank {args.rank}: {index+1}/{len(rows[args.rank::args.world_size])}', flush=True)
    return records


def analyze(args, rows, refs):
    assert set(refs) == {r['id'] for r in rows}, 'Missing/extra prediction IDs'
    output = Path(args.output)
    assert not output.exists(), f'Output exists: {output}; choose a new directory'
    details, predictions = [], []
    for ann in rows:
        r = refs[ann['id']]
        assert r['image_name'] == ann['image_name']
        d = diagnose(ann['answer_boxes'], r['boxes'], r['scores'], args.score_threshold, args.iou)
        predictions.append(dict(id=ann['id'], extracted_predictions=d.pop('selected_boxes')))
        details.append(dict(id=ann['id'], image_name=ann['image_name'],
                            domain=ann['domain'], referring=ann['referring'], **d))
    summary = dict(score_threshold=args.score_threshold, diagnostic_iou=args.iou,
                   overall=aggregate(details), by_domain={domain: aggregate([
                       r for r in details if r['domain'] == domain]) for domain in sorted({r['domain'] for r in details})},
                   timing_note='Cached stages; includes cold calls, excludes model loading. Not online E2E FPS.',
                   ref_seconds=sum(r['seconds'] for r in refs.values()))
    save_json(output / 'diagnostics.json', summary)
    save_json(output / 'samples.json', details)
    with (output / 'predictions.jsonl').open('x', encoding='utf-8') as f:
        for r in predictions:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    if not args.skip_official_metrics:
        from recall_precision_densityf1 import evaluate_dataset, print_comparative_metrics
        # Keep original candidate_boxes: official DensityF1 uses their count.
        metrics = evaluate_dataset(rows, predictions)
        print_comparative_metrics({'Uni-Ref': metrics}, rows, str(output / 'official'))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['uni', 'ref', 'analyze'])
    p.add_argument('--annotations', required=True)
    p.add_argument('--images')
    p.add_argument('--checkpoint')
    p.add_argument('--output', required=True)
    p.add_argument('--uni-dir')
    p.add_argument('--ref-dir')
    p.add_argument('--num-proposals', type=int, default=100)
    p.add_argument('--backbone', choices=['base', 'large'], default='base')
    p.add_argument('--attention', default='flash_attention_2', choices=['flash_attention_2', 'sdpa', 'eager'])
    p.add_argument('--score-threshold', type=float, default=0.35)
    p.add_argument('--iou', type=float, default=0.5)
    p.add_argument('--limit', type=int, default=0, help='First N expressions; 0 means complete benchmark')
    p.add_argument('--skip-official-metrics', action='store_true', help='Diagnostics only; useful for CPU tests')
    args = p.parse_args()
    assert args.num_proposals > 0 and args.limit >= 0
    assert 0 <= args.score_threshold <= 1 and 0 < args.iou <= 1
    args.rank, args.world_size = int(os.getenv('RANK', '0')), int(os.getenv('WORLD_SIZE', '1'))
    rows = load_annotations(args.annotations, args.limit)
    meta = dict(annotation_sha256=digest(args.annotations), ids=[r['id'] for r in rows],
                num_proposals=args.num_proposals)
    proposals = None
    if args.stage == 'ref':
        assert args.uni_dir
        proposals, source, hashes = load_shards(args.uni_dir, 'uni')
        assert source['annotation_sha256'] == meta['annotation_sha256'] and source['ids'] == meta['ids']
        assert source['num_proposals'] >= args.num_proposals
        assert set(proposals) == {r['image_name'] for r in rows}
        meta['uni_shard_sha256'] = hashes
    if args.stage == 'analyze':
        assert args.world_size == 1 and args.ref_dir, 'Run analysis with plain python'
        refs, source, hashes = load_shards(args.ref_dir, 'ref')
        assert source['annotation_sha256'] == meta['annotation_sha256'] and source['ids'] == meta['ids']
        analyze(args, rows, refs)
        save_json(Path(args.output) / 'provenance.json', dict(ref_meta=source, ref_shard_sha256=hashes))
        return
    assert args.images and args.checkpoint
    target = Path(args.output) / f'{args.stage}.rank{args.rank:03d}.json'
    assert not target.exists(), f'{target} exists; choose a new directory'
    import torch
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
    torch.manual_seed(0)
    meta.update(checkpoint=str(Path(args.checkpoint).resolve()), torch_version=str(torch.__version__),
                script_sha256=digest(__file__), gpu=torch.cuda.get_device_name(),
                images=str(Path(args.images).resolve()), attention=args.attention,
                backbone=args.backbone,
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    if args.stage == 'uni':
        meta['checkpoint_sha256'] = digest(args.checkpoint)
    records = run_uni(args, rows) if args.stage == 'uni' else run_ref(args, rows, proposals)
    save_json(target, dict(meta=meta, rank=args.rank, world_size=args.world_size, records=records))
    print(f'Saved {target}', flush=True)


if __name__ == '__main__':
    main()
