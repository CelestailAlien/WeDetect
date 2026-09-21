"""P-linear data preflight and frozen feature extraction. See REF_PLINEAR.md.

PL_STAGE=preflight|cache|cache_val; training/evaluation have separate scripts.
Never reuse validation features as training data, or overwrite a previous run.
"""
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path

from humanref_pipeline import ROOT, digest, image_path, iou, save_json, validate_boxes
from ref_e0 import (ATTENTION, EXPECTED_TRANSFORMERS, MAX_PROPOSALS, prepare_input,
                    package_version, write_tensor, STRICT_ATOL, STRICT_RTOL,
                    COMPACT_ATOL, COMPACT_RTOL)
from ref_e0_data import canonical_hash, load_rec_annotations
from ref_plinear_data import image_key, read_json, split_train_dev

DEPTHS = [9, 18, 24, 30, 36]
SPLIT_SEED = 20260921
TRAIN_N = int(os.environ.get('PL_TRAIN_N', '5000'))
DEV_N = int(os.environ.get('PL_DEV_N', '1000'))
VAL_N = int(os.environ.get('PL_VAL_N', '0'))  # 0 = complete validation; positive = smoke
OUTPUT = Path(os.environ.get('PL_OUT', ROOT / 'results/ref_plinear_refcocog_v1'))
CHECKPOINT = Path(os.environ.get('PL_REF', ROOT / 'checkpoints/WeDetect-Ref-4B'))
IMAGES = Path(os.environ.get('PL_IMAGES', ROOT / 'data/coco2014'))
REFERENCE = Path(os.environ.get('PL_REFERENCE', ROOT / 'results/ref_full_d30_refcocog_validation'))
VAL_ANN = Path(os.environ.get('PL_VAL_ANN', ROOT / 'wedetect_ref/eval_grounding/eval_refcoco/refcocog_validation.json'))
VAL_PROPOSALS = Path(os.environ.get('PL_VAL_PROPOSALS', ROOT / 'wedetect_ref/eval_grounding/eval_refcoco/refcoco_proposals_all.json'))


def source_hashes():
    paths = [Path(__file__), ROOT / 'tools/ref_plinear_core.py', ROOT / 'tools/ref_plinear_data.py',
             ROOT / 'tools/ref_e0.py', ROOT / 'tools/ref_e0_core.py', ROOT / 'tools/ref_e0_data.py',
             ROOT / 'tools/ref_pnative_core.py', ROOT / 'tools/humanref_pipeline.py',
             ROOT / 'wedetect_ref/models/qwen3vl_referring.py', ROOT / 'wedetect_ref/models/vision_process.py']
    return {p.name: digest(p) for p in paths}


def preflight():
    assert not OUTPUT.exists(), f'Refusing to overwrite {OUTPUT}; choose a new PL_OUT'
    for name in ('PL_TRAIN_ANN', 'PL_TRAIN_PROPOSALS'):
        assert os.environ.get(name), f'{name} is REQUIRED: genuine training data, not validation!'
    train_ann, train_props = Path(os.environ['PL_TRAIN_ANN']), Path(os.environ['PL_TRAIN_PROPOSALS'])
    assert read_json(REFERENCE / 'summary.json')['status'] == 'PASSED'
    ref = read_json(REFERENCE / 'manifest.json')
    assert ref['full_split'] and ref['num_layers'] == 36
    assert digest(VAL_ANN) == ref['annotation_sha256']
    assert digest(VAL_PROPOSALS) == ref['proposals_sha256']
    train, train_boxes = load_rec_annotations(train_ann, train_props)
    validation, val_boxes = load_rec_annotations(VAL_ANN, VAL_PROPOSALS)
    assert all(r['id'].startswith('refcocog_train_') for r in train), 'Use RefCOCOg train, not mixed REC data'
    assert [r['id'] for r in validation] == ref['sample_ids']
    selected = split_train_dev(train, validation, TRAIN_N, DEV_N, SPLIT_SEED)
    assert 0 <= VAL_N <= len(validation)
    selected['validation'] = validation if VAL_N == 0 else validation[:VAL_N]
    hashes, paths = {}, {}
    reference_images = {}
    for path, sha in ref['image_sha256'].items():
        key = image_key(path)
        assert key not in reference_images or reference_images[key] == sha
        reference_images[key] = sha
    records = []
    for split in ('train', 'dev', 'validation'):
        for row in selected[split]:
            name = row['image_name']
            if name not in paths:
                paths[name] = str(image_path(IMAGES, name).resolve())
                hashes[name] = digest(paths[name])
            if split == 'validation':
                assert hashes[name] == reference_images[image_key(name)], f'Baseline validation image changed: {name}'
            boxes = (val_boxes if split == 'validation' else train_boxes)[name]['boxes'][:MAX_PROPOSALS]
            assert boxes, f'No candidates: {name}; never insert GT'
            records.append(dict(row, split=split, image_key=image_key(name), image_path=paths[name],
                                image_sha256=hashes[name], candidate_boxes=boxes))
    hash_sets = [{r['image_sha256'] for r in records if r['split'] == split}
                 for split in ('train', 'dev', 'validation')]
    assert all(not (hash_sets[i] & hash_sets[j]) for i in range(3) for j in range(i)), 'Identical image bytes cross splits'
    plan = dict(version=1, depths=DEPTHS, split_seed=SPLIT_SEED,
        counts={s: len(rs) for s, rs in selected.items()}, rows=records,
        smoke_only=(TRAIN_N != 5000 or DEV_N != 1000 or VAL_N != 0),
        train_annotation_sha256=digest(train_ann), train_proposals_sha256=digest(train_props),
        validation_annotation_sha256=digest(VAL_ANN), validation_proposals_sha256=digest(VAL_PROPOSALS),
        source_paths=dict(train_annotations=str(train_ann.resolve()), train_proposals=str(train_props.resolve()),
                          validation_annotations=str(VAL_ANN.resolve()), validation_proposals=str(VAL_PROPOSALS.resolve())),
        reference_manifest_sha256=digest(REFERENCE / 'manifest.json'),
        reference_summary_sha256=digest(REFERENCE / 'summary.json'),
        sources=source_hashes(), selection='seeded image groups; dev first, train second; no GT-based sampling',
        protocol='first 100 fixed proposals; clipped; no GT insertion/shuffle/NMS/score cutoff; IoU>=0.5',
        ranking='FP32 raw logits for all FP32 heads; original BF16 raw/sigmoid retained separately')
    save_json(OUTPUT / 'plan.json', plan)
    size = len(records) * len(DEPTHS) * 100 * 2560 * 2 / 1e9
    print(f'Preflight PASS: {plan["counts"]}; smoke={plan["smoke_only"]}; feature payload <= {size:.2f} GB')
    print(f'Frozen split: {OUTPUT / "plan.json"}. Cache train/dev next; validation stays unused by training.')


def cache(validation=False):
    # Check inexpensive provenance before allocating the model.
    plan = read_json(OUTPUT / 'plan.json')
    assert plan['depths'] == DEPTHS and plan['sources'] == source_hashes(), 'Sources changed after preflight'
    assert digest(REFERENCE / 'manifest.json') == plan['reference_manifest_sha256']
    assert digest(REFERENCE / 'summary.json') == plan['reference_summary_sha256']
    if validation:
        trained = read_json(OUTPUT / 'train/COMPLETE.json')
        assert trained['status'] == 'PASSED', 'Freeze heads BEFORE validation'
        assert trained['plan_sha256'] == digest(OUTPUT / 'plan.json')
        assert trained['protocol_sha256'] == digest(OUTPUT / 'train/protocol.json')
        assert trained['checkpoints']
        for checkpoint in trained['checkpoints']:
            assert digest(OUTPUT / 'train' / checkpoint['file']) == checkpoint['sha256']
    stage = 'validation' if validation else 'fit'
    out = OUTPUT / ('cache_validation' if validation else 'cache_fit')
    assert not out.exists(), f'Partial/existing cache: {out}. Keep evidence and start a new PL_OUT'
    rows = [r for r in plan['rows'] if (r['split'] == 'validation') == validation]
    assert rows
    for r in rows:
        assert digest(r['image_path']) == r['image_sha256'], f'Changed image: {r["image_path"]}'
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor
    from models.qwen3vl_referring import Qwen3VLGroundingForConditionalGeneration
    from ref_e0_core import compare_tensors, forward_algorithm as original_readout, object_logits
    from ref_plinear_core import frozen_norm
    from ref_pnative_core import capture_depths

    assert int(os.environ.get('WORLD_SIZE', '1')) == 1
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert transformers.__version__ == EXPECTED_TRANSFORMERS
    ref = read_json(REFERENCE / 'manifest.json')
    weights = sorted(list(CHECKPOINT.glob('*.safetensors')) + list(CHECKPOINT.glob('*.bin')))
    assert weights
    print('Hashing/loading frozen model...', flush=True)
    hashes = {p.name: digest(p) for p in weights}
    json_hashes = {p.name: digest(p) for p in sorted(CHECKPOINT.glob('*.json'))}
    assert hashes == ref['checkpoint_sha256'] and json_hashes == ref['checkpoint_json_sha256']
    prior_sources = {Path(p).name: h for p, h in ref['source_sha256'].items()}
    for name, sha in source_hashes().items():
        if name in prior_sources:
            assert sha == prior_sources[name], f'Previously validated source changed: {name}'
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, loading = Qwen3VLGroundingForConditionalGeneration.from_pretrained(
        str(CHECKPOINT), torch_dtype=torch.bfloat16, attn_implementation=ATTENTION,
        local_files_only=True, output_loading_info=True)
    assert not any(loading[k] for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), loading
    model = model.cuda().eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(str(CHECKPOINT), local_files_only=True)
    object_id = processor.tokenizer.convert_tokens_to_ids('<object>')
    assert object_id != processor.tokenizer.unk_token_id
    assert processor.tokenizer.encode('<object>', add_special_tokens=False) == [object_id]
    model.model.object_token_id = object_id
    lm, head = model.model.language_model, model.out_proj
    assert not any(m.training for m in model.modules()) and not any(p.requires_grad for p in model.parameters())
    signature = dict(checkpoint_sha256=hashes, checkpoint_json_sha256=json_hashes,
        torch=str(torch.__version__), transformers=transformers.__version__,
        torchvision=package_version('torchvision'), flash_attn=package_version('flash-attn'),
        dtype=str(model.dtype), attention=ATTENTION, text_attention=lm.config._attn_implementation,
        vision_attention=model.model.visual.config._attn_implementation,
        num_layers=len(lm.layers), hidden_size=lm.config.hidden_size,
        object_token_id=object_id, checkpoint_use_cache=lm.config.use_cache,
        final_norm_epsilon=lm.norm.variance_epsilon)
    for key, value in signature.items():
        assert value == ref[key], f'Baseline protocol changed: {key}'
    hf_source = Path(inspect.getfile(type(lm)))
    assert digest(hf_source) == prior_sources[hf_source.name]
    signature['sources'] = source_hashes()
    signature['transformers_source'] = digest(hf_source)
    if validation:
        fit = read_json(OUTPUT / 'cache_fit/manifest.json')
        assert signature == fit['model_signature'], 'Train/eval features use different model/protocol'
    manifest = dict(stage=stage, depths=DEPTHS, plan_sha256=digest(OUTPUT / 'plan.json'),
        model_signature=signature, gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
        created_utc=datetime.now(timezone.utc).isoformat(), sample_ids=[r['id'] for r in rows],
        feature_format='BF16 original frozen final RMSNorm(h_depth); FP32 head inputs',
        actual_early_exit=False, training=False, loading_info=loading)
    if validation:
        manifest['frozen_train_complete_sha256'] = digest(OUTPUT / 'train/COMPLETE.json')
    save_json(out / 'manifest.json', manifest)
    (out / 'samples').mkdir()
    write_tensor(out / 'native_head.pt', dict(weight=head.weight.detach().cpu(),
        bias=head.bias.detach().cpu(), norm_weight=lm.norm.weight.detach().cpu(),
        norm_epsilon=lm.norm.variance_epsilon), torch)
    entries = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            with Image.open(row['image_path']) as file:
                image = file.convert('RGB')
            width, height = image.size
            def clip(box):
                return [max(0, min(width, box[0])), max(0, min(height, box[1])),
                        max(0, min(width, box[2])), max(0, min(height, box[3]))]
            boxes, gt = [clip(b) for b in row['candidate_boxes']], [clip(b) for b in row['answer_boxes']]
            validate_boxes(boxes)
            validate_boxes(gt)
            inputs, positions, prompt = prepare_input(model, processor, image, row['referring'], boxes, object_id)
            # One plain pass + one read-only capture: no unchecked hook side effects.
            pred = model(**inputs)
            baseline = object_logits(pred.logits, positions).clone()
            del pred
            with capture_depths(model, positions, [0] + DEPTHS) as captured:
                pred = model(**inputs)
            hooked = object_logits(pred.logits, positions)
            full = original_readout(captured.pop('hL_full'), lm.norm.weight,
                lm.norm.variance_epsilon, head.weight, head.bias)
            checks = dict(hooks=compare_tensors(baseline, hooked, STRICT_ATOL, STRICT_RTOL),
                full_readout=compare_tensors(baseline, object_logits(full, positions), STRICT_ATOL, STRICT_RTOL))
            features = [frozen_norm(captured['states'][k], lm.norm.weight, lm.norm.variance_epsilon) for k in DEPTHS]
            native = torch.stack([torch.nn.functional.linear(x, head.weight, head.bias)[:, 0] for x in features])
            fp32 = torch.stack([torch.nn.functional.linear(x.float(), head.weight.float(), head.bias.float())[:, 0] for x in features])
            checks['compact_readout'] = compare_tensors(baseline, native[-1], COMPACT_ATOL, COMPACT_RTOL)
            meta = dict(id=row['id'], split=row['split'], image_key=row['image_key'], image_name=row['image_name'],
                query=row['referring'], boxes=boxes, gt=gt, prompt=prompt, depths=DEPTHS,
                checks=checks, compact_raw_winner_equal=int(baseline.argmax()) == int(native[-1].argmax()),
                compact_sigmoid_winner_equal=int(baseline.sigmoid().argmax()) == int(native[-1].sigmoid().argmax()))
            save_json(out / 'samples' / f'{index:05d}.json', meta)
            assert all(c['passed'] for c in checks.values()), f'Consistency gate failed at {index}; do not loosen tolerances'
            overlaps = torch.tensor([iou(b, gt[0]) for b in boxes], dtype=torch.float32)
            path = out / 'samples' / f'{index:05d}.pt'
            write_tensor(path, dict(meta, features=torch.stack(features).cpu(), overlaps=overlaps,
                labels=torch.where(overlaps > .5, overlaps, 0), native_bf16=native.cpu(),
                native_fp32=fp32.cpu(), baseline=baseline.cpu()), torch)
            entries.append(dict(id=row['id'], split=row['split'], image_key=row['image_key'],
                file=f'samples/{index:05d}.pt', sha256=digest(path)))
            print(f'P-linear cache {stage} {index+1}/{len(rows)} {row["id"]} PASS', flush=True)
            del inputs, positions, pred, hooked, full, captured, features, native, fp32, baseline
    save_json(out / 'index.json', entries)
    save_json(out / 'COMPLETE.json', dict(status='PASSED', stage=stage, samples=len(entries),
        manifest_sha256=digest(out / 'manifest.json'), index_sha256=digest(out / 'index.json'),
        native_head_sha256=digest(out / 'native_head.pt')))
    print(f'Completed {out}', flush=True)


if __name__ == '__main__':
    assert __debug__, 'Do not disable assertions'
    stage = os.environ.get('PL_STAGE', 'preflight')
    assert stage in ('preflight', 'cache', 'cache_val')
    if stage == 'preflight':
        preflight()
    else:
        cache(validation=stage == 'cache_val')
