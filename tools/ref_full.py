"""Full RefCOCOg validation: frozen full-36 vs real exit-30, accuracy only.

Run at repository root. FULL_N=0 means ALL annotation rows, not a sampled prefix.
See REF_FULL.md. Existing E0/P-native/static-exit files are never modified.
"""
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
import platform
import subprocess
import traceback

from humanref_pipeline import ROOT, digest, image_path, iou, save_json
from ref_e0 import (ATTENTION, EXPECTED_TRANSFORMERS, MAX_PROPOSALS, SEED,
                    STRICT_ATOL, STRICT_RTOL, COMPACT_ATOL, COMPACT_RTOL,
                    prepare_input, scores_from_logits, package_version)
from ref_e0_data import canonical_hash, load_rec_annotations
from ref_exit import clipped_boxes
from ref_exit_analysis import decisions
from ref_pnative import read_json, check_e0_protocol
from ref_full_analysis import select_rows, history_diagnostic, summarize, markdown

ANNOTATIONS = Path(os.environ.get('FULL_ANN', ROOT / 'wedetect_ref/eval_grounding/eval_refcoco/refcocog_validation.json'))
PROPOSALS = Path(os.environ.get('FULL_PROPOSALS', ROOT / 'wedetect_ref/eval_grounding/eval_refcoco/refcoco_proposals_all.json'))
IMAGES = Path(os.environ.get('FULL_IMAGES', ROOT / 'data/coco2014'))
CHECKPOINT = Path(os.environ.get('FULL_REF', ROOT / 'checkpoints/WeDetect-Ref-4B'))
REFERENCE = Path(os.environ.get('FULL_REFERENCE', ROOT / 'results/ref_exit_d30_refcocog_a6000_val500'))
LIMIT = int(os.environ.get('FULL_N', '0'))
DEPTH = 30  # Frozen BEFORE expanding validation; not an environment sweep option.
OUTPUT = Path(os.environ.get('FULL_OUT', ROOT / ('results/ref_full_d30_refcocog_validation'
                                               if LIMIT == 0 else f'results/ref_full_d30_refcocog_smoke{LIMIT}')))


def load_reference(directory):
    summary, manifest = read_json(directory / 'summary.json'), read_json(directory / 'manifest.json')
    assert not (directory / 'FAILED.json').exists()
    assert summary['status'] == 'PASSED' and manifest['stage'] == 'static-exit'
    assert summary['actual_early_exit'] and manifest['actual_early_exit']
    assert summary['exit_depth'] == manifest['exit_depth'] == DEPTH
    assert summary['total_depth'] == manifest['num_layers'] == 36
    files = sorted((directory / 'samples').glob('*.json'))
    records = [read_json(p) for p in files]
    assert len(records) == summary['samples'] > 0
    assert [r['id'] for r in records] == manifest['sample_ids']
    assert len({r['id'] for r in records}) == len(records)
    assert all(r['passed'] and all(c['passed'] for c in r['checks'].values()) for r in records)
    assert all(r['executed_full'] == list(range(36)) and r['executed_exit'] == list(range(DEPTH)) for r in records)
    return manifest, {r['id']: r for r in records}, {p.name: digest(p) for p in files}


def write_reports(output, records, selected_ids, all_ids, prior_ids, prior_images):
    report, errors = summarize(records, selected_ids, all_ids, prior_ids, prior_images)
    # Re-read every written record; do not silently publish a complete result
    # when files are missing, mixed, or truncated during transfer/write.
    paths = sorted((output / 'samples').glob('*.json'))
    assert len(paths) == len(records)
    assert [read_json(p) for p in paths] == records
    save_json(output / 'summary.json', report)
    save_json(output / 'paired_errors.json', errors)
    save_json(output / 'sample_hashes.json', {p.name: digest(p) for p in paths})
    with (output / 'summary.md').open('x', encoding='utf-8') as stream:
        stream.write(markdown(report))
    return report


def run():
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor
    from models.qwen3vl_referring import Qwen3VLGroundingForConditionalGeneration
    from ref_full_core import evaluate_pair

    assert __debug__ and int(os.environ.get('WORLD_SIZE', '1')) == 1
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert transformers.__version__ == EXPECTED_TRANSFORMERS
    assert not OUTPUT.exists(), f'Refusing overwrite: {OUTPUT}; choose a new FULL_OUT'
    assert CHECKPOINT.is_dir() and IMAGES.is_dir()
    assert ANNOTATIONS.stem == 'refcocog_validation', 'This frozen experiment is RefCOCOg validation only'
    prior, references, reference_hashes = load_reference(REFERENCE)
    rows, proposals = load_rec_annotations(ANNOTATIONS, PROPOSALS)
    selected = select_rows(rows, LIMIT)
    all_ids, selected_ids = [r['id'] for r in rows], [r['id'] for r in selected]
    assert set(references) <= set(all_ids), 'Reference belongs to a different split'
    assert digest(ANNOTATIONS) == prior['annotation_sha256']
    assert digest(PROPOSALS) == prior['proposals_sha256']
    assert all(proposals[r['image_name']]['boxes'] for r in rows), 'Missing/empty author candidates; do not skip rows'
    paths = {name: image_path(IMAGES, name) for name in sorted({r['image_name'] for r in selected})}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'samples').mkdir()
    scope = 'ALL validation rows' if len(selected) == len(rows) else 'SMOKE ONLY'
    print(f'RefCOCOg {scope}: {len(selected)}/{len(rows)} expressions; full 36 vs exit 30. No timing.', flush=True)
    print('Hashing checkpoint/images and loading frozen model...', flush=True)
    weights = sorted(list(CHECKPOINT.glob('*.safetensors')) + list(CHECKPOINT.glob('*.bin')))
    assert weights
    hashes = {p.name: digest(p) for p in weights}
    assert hashes == prior['checkpoint_sha256'], 'Checkpoint changed'
    torch.manual_seed(SEED)
    model, loading = Qwen3VLGroundingForConditionalGeneration.from_pretrained(
        str(CHECKPOINT), torch_dtype=torch.bfloat16, attn_implementation=ATTENTION,
        local_files_only=True, output_loading_info=True)
    save_json(OUTPUT / 'loading_info.json', loading)
    assert not any(loading[k] for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    model = model.cuda().eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(str(CHECKPOINT), local_files_only=True)
    object_id = processor.tokenizer.convert_tokens_to_ids('<object>')
    assert object_id != processor.tokenizer.unk_token_id
    assert processor.tokenizer.encode('<object>', add_special_tokens=False) == [object_id]
    model.model.object_token_id = object_id
    lm = model.model.language_model
    assert len(lm.layers) == lm.config.num_hidden_layers == 36
    assert not any(m.training for m in model.modules()) and not any(p.requires_grad for p in model.parameters())
    sources = [Path(__file__), ROOT / 'tools/ref_full_core.py', ROOT / 'tools/ref_full_analysis.py',
        ROOT / 'tools/ref_exit.py', ROOT / 'tools/ref_exit_core.py', ROOT / 'tools/ref_exit_analysis.py',
        ROOT / 'tools/ref_pnative.py', ROOT / 'tools/ref_pnative_core.py', ROOT / 'tools/ref_pnative_analysis.py',
        ROOT / 'tools/ref_e0.py', ROOT / 'tools/ref_e0_core.py', ROOT / 'tools/ref_e0_data.py',
        ROOT / 'tools/humanref_pipeline.py', ROOT / 'wedetect_ref/models/qwen3vl_referring.py',
        ROOT / 'wedetect_ref/models/vision_process.py', Path(inspect.getfile(type(lm)))]
    properties = torch.cuda.get_device_properties(0)
    manifest = dict(stage='ref-full-validation', version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        transformers=transformers.__version__, torchvision=package_version('torchvision'),
        flash_attn=package_version('flash-attn'), gpu=torch.cuda.get_device_name(0),
        gpu_total_memory_bytes=properties.total_memory, gpu_capability=list(torch.cuda.get_device_capability(0)),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), reference_gpu=prior['gpu'],
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        matmul_allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        git_commit=subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=True).stdout.strip(),
        checkpoint=str(CHECKPOINT.resolve()), checkpoint_sha256=hashes,
        checkpoint_json_sha256={p.name: digest(p) for p in sorted(CHECKPOINT.glob('*.json'))},
        source_sha256={str(p): digest(p) for p in sources},
        reference=str(REFERENCE.resolve()), reference_manifest_sha256=digest(REFERENCE / 'manifest.json'),
        reference_summary_sha256=digest(REFERENCE / 'summary.json'), reference_samples_sha256=reference_hashes,
        annotations=str(ANNOTATIONS.resolve()), annotation_sha256=digest(ANNOTATIONS),
        proposals=str(PROPOSALS.resolve()), proposals_sha256=digest(PROPOSALS),
        image_sha256={name: digest(p) for name, p in paths.items()},
        dataset_split=ANNOTATIONS.stem, annotation_samples=len(rows), sample_ids=selected_ids,
        annotation_sample_ids=all_ids, full_split=selected_ids == all_ids,
        sample_selection='All annotation rows in original order; FULL_N>0 is smoke only',
        seed=SEED, max_proposals=MAX_PROPOSALS, gt_insertion=False,
        coordinate_policy='FP32 saved geometry; BF16 model box input',
        selection='Top-1 by Ref sigmoid score; ties use first original proposal',
        score_threshold=None, nms_iou=None, evaluation_iou=0.5,
        sigmoid_policy='model dtype sigmoid, then FP32 (matches existing pipeline)',
        diagnostic_selection='Top-1 by raw BF16 logits converted to FP32; first original proposal on ties; BOTH arms',
        dtype=str(model.dtype), attention=ATTENTION, text_attention=lm.config._attn_implementation,
        vision_attention=model.model.visual.config._attn_implementation,
        checkpoint_use_cache=lm.config.use_cache, num_layers=36, hidden_size=lm.config.hidden_size,
        object_token_id=object_id, final_norm_epsilon=lm.norm.variance_epsilon,
        strict_tolerance=dict(atol=STRICT_ATOL, rtol=STRICT_RTOL),
        compact_tolerance=dict(atol=COMPACT_ATOL, rtol=COMPACT_RTOL),
        exit_depth=DEPTH, training=False, actual_early_exit=True, timing_measured=False,
        history_policy='Input/protocol provenance is strict; historic score drift never gates same-run correctness',
        local_checks='Every expression: full/exit block trace, boundary, full-shape readout and both winners. First: unhooked/sham controls.',
        tail_parameters_remain_resident=True, cache_reuse=False)
    save_json(OUTPUT / 'manifest.json', manifest)
    check_e0_protocol(manifest, prior)  # Deliberately does not require the same GPU.
    save_json(OUTPUT / 'selection.json', dict(rows=selected, full_split=selected_ids == all_ids,
        annotation_samples=len(rows), rule=manifest['sample_selection']))
    save_json(OUTPUT / 'model_config.json', model.config.to_dict())
    if manifest['gpu'] != prior['gpu']:
        print(f'GPU changed: {prior["gpu"]} -> {manifest["gpu"]}; history is diagnostic, same-run checks remain strict.', flush=True)
    records = []
    prior_images = {r['image_name'] for r in references.values()}
    with torch.inference_mode():
        for index, ann in enumerate(selected):
            path = paths[ann['image_name']]
            with Image.open(path) as source:
                image = source.convert('RGB')
            boxes = clipped_boxes(proposals[ann['image_name']]['boxes'][:MAX_PROPOSALS], *image.size)
            gt = clipped_boxes(ann['answer_boxes'], *image.size)
            inputs, positions, prompt = prepare_input(model, processor, image, ann['referring'], boxes, object_id)
            values, info = evaluate_pair(model, inputs, positions, DEPTH, STRICT_ATOL, STRICT_RTOL, control=index == 0)
            logits = {arm: v.float().cpu().tolist() for arm, v in values.items()}
            scores = {arm: scores_from_logits(v) for arm, v in values.items()}
            choices = {arm: decisions(boxes, gt, logits[arm], scores[arm]) for arm in values}
            record = dict(index=index, id=ann['id'], image_name=ann['image_name'], query=ann['referring'],
                boxes=boxes, answer_boxes=gt, prompt_sha256=canonical_hash(prompt),
                num_candidates=len(boxes), sequence_length=positions.shape[1],
                object_token_positions=positions[0].nonzero()[:, 0].cpu().tolist(),
                candidate_covers_gt=any(iou(b, gt[0]) >= .5 for b in boxes),
                logits=logits, scores=scores, decisions=choices, history=None, **info)
            if ann['id'] in references:
                old_hashes = [sha for name, sha in prior['image_sha256'].items()
                              if name.replace('\\', '/').endswith('/' + ann['image_name'])]
                assert old_hashes and all(sha == manifest['image_sha256'][ann['image_name']] for sha in old_hashes)
                record['history'] = history_diagnostic(record, references[ann['id']])
            sample_path = OUTPUT / 'samples' / f'{index:05d}.json'
            save_json(sample_path, record)
            assert record['passed'], f'Same-run exit consistency failed: {sample_path}; do not relax gates'
            records.append(record)
            history_changed = record['history'] is not None and any(not h['logits_exact'] for h in record['history'].values())
            print(f'Full-val {index+1}/{len(selected)} {ann["id"]} PASS '
                f'sigmoid={int(choices["full"]["bf16_sigmoid"]["correct"])}->{int(choices["exit"]["bf16_sigmoid"]["correct"])} '
                f'raw={int(choices["full"]["raw_logit"]["correct"])}->{int(choices["exit"]["raw_logit"]["correct"])} '
                f'history_drift={history_changed}', flush=True)
            del inputs, positions, values
    assert len(records) == len(selected)
    report = write_reports(OUTPUT, records, selected_ids, all_ids, list(references), prior_images)
    print(f'Completed {report["evaluation_scope"]}: {OUTPUT / "summary.md"}', flush=True)


if __name__ == '__main__':
    existed = OUTPUT.exists()
    try:
        run()
    except Exception as error:
        if not existed and OUTPUT.is_dir():
            save_json(OUTPUT / 'FAILED.json', dict(status='FAILED', error=repr(error), traceback=traceback.format_exc()))
        raise
