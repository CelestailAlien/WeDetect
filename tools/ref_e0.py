"""E0: validate fixed-proposal Ref boundaries before building probes/exit paths.

Run from repo root: python -B tools/ref_e0.py. See REF_E0.md for paths and gates.
Only E0_* environment variables below are supported; no training configuration.
"""
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import inspect
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import traceback

from humanref_pipeline import ROOT, digest, image_path, save_json, validate_boxes
from ref_e0_data import canonical_hash, choose_samples, compare_decisions, load_rec_annotations

# Match the already-tested Ref setup; never silently switch attention backends.
ANNOTATIONS = Path(os.environ.get('E0_ANN', ROOT / 'wedetect_ref/eval_grounding/eval_refcoco/refcocog_validation.json'))
PROPOSALS = Path(os.environ.get('E0_PROPOSALS', ROOT / 'wedetect_ref/eval_grounding/eval_refcoco/refcoco_proposals_all.json'))
IMAGES = Path(os.environ.get('E0_IMAGES', ROOT / 'data/coco2014'))
CHECKPOINT = Path(os.environ.get('E0_REF', ROOT / 'checkpoints/WeDetect-Ref-4B'))
OUTPUT = Path(os.environ.get('E0_OUT', ROOT / 'results/ref_e0_refcocog_val60'))
SAMPLE_COUNT = int(os.environ.get('E0_N', '60'))
SEED = 42
MAX_PROPOSALS = 100
ATTENTION = 'flash_attention_2'
EXPECTED_TRANSFORMERS = '4.57.1'
# REC chooses one highest-scoring candidate. No Ref score cutoff or extra NMS.
WARMUP_PASSES = 2
TIMING_REPEATS = 3
STRICT_ATOL, STRICT_RTOL = 1e-5, 1e-5
# Compact GEMM shape may change BF16 rounding. Also require unchanged decisions.
COMPACT_ATOL, COMPACT_RTOL = 0.03125, 0.01


def write_tensor(path, data, torch):
    with path.open('xb') as stream:
        torch.save(data, stream)


def prepare_input(model, processor, image, query, boxes, object_id):
    import torch
    from models.vision_process import process_vision_info

    messages = [dict(role='user', content=[dict(type='image', image=image),
                dict(type='text', text=f'Please detect the "{query}" in the image')]),
                dict(role='assistant', content=[dict(type='text', text='<object>' * len(boxes))])]
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    rendered = processor.apply_chat_template(messages, tokenize=False)
    inputs = processor(text=[rendered], images=image_inputs, videos=video_inputs,
                       return_tensors='pt', padding=True, do_resize=False).to(model.device)
    positions = inputs['input_ids'] == object_id
    assert positions.shape[0] == 1 and int(positions.sum()) == len(boxes)
    assert len(boxes) > 0, 'E0 cannot validate hidden-state readout with no object tokens'
    inputs.update(bboxes=[torch.tensor(boxes, device=model.device, dtype=model.dtype)],
                  ori_shapes=[image.size], bboxes_id=object_id, image_inputs=image_inputs)
    # Preserve checkpoint's cache setting, like the original eval.py. No cache reuse.
    return dict(inputs), positions, rendered


def scores_from_logits(values):
    # Official path applies sigmoid in model dtype, THEN converts to FP32.
    return values.sigmoid().float().cpu().tolist()


def package_version(name):
    # A missing dependency is a real error: do not invent an environment version.
    return importlib.metadata.version(name)


def run():
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor
    from models.qwen3vl_referring import Qwen3VLGroundingForConditionalGeneration
    from ref_e0_core import (capture_boundaries, compare_tensors, forward_algorithm,
                             object_logits, profile_forward)

    assert __debug__, 'Do not run with python -O: E0 uses assertions as gates'
    assert int(os.environ.get('WORLD_SIZE', '1')) == 1, 'E0 is single-GPU; use python, not multi-rank torchrun'
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert transformers.__version__ == EXPECTED_TRANSFORMERS, (
        f'E0 audited {EXPECTED_TRANSFORMERS}, found {transformers.__version__}; review before changing versions')
    assert ANNOTATIONS.is_file(), f'Missing annotations: {ANNOTATIONS}'
    assert PROPOSALS.is_file(), f'Missing author proposals: {PROPOSALS}'
    assert IMAGES.is_dir(), f'Missing image root: {IMAGES}'
    assert CHECKPOINT.is_dir(), f'Missing Ref checkpoint: {CHECKPOINT}'
    assert not OUTPUT.exists(), f'Refusing to overwrite {OUTPUT}; choose a new E0_OUT'
    rows, proposals = load_rec_annotations(ANNOTATIONS, PROPOSALS)
    selected = choose_samples(rows, SAMPLE_COUNT, SEED)
    assert all(proposals[row['image_name']]['boxes'] for row in selected)
    paths = [image_path(IMAGES, row['image_name']) for row in selected]
    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'samples').mkdir()
    save_json(OUTPUT / 'selection.json', dict(seed=SEED, rows=selected,
              selection_rule='real query pair first; seeded shuffle of remaining expressions',
              first_pair_ids=[row['id'] for row in selected[:2]],
              annotation_sha256=digest(ANNOTATIONS), proposals_sha256=digest(PROPOSALS)))
    print(f'E0 output: {OUTPUT.resolve()}\nHashing checkpoint and loading Ref (not timed)...', flush=True)
    weight_paths = sorted(list(CHECKPOINT.glob('*.safetensors')) + list(CHECKPOINT.glob('*.bin')))
    assert weight_paths, 'No checkpoint weight files found'
    checkpoint_hashes = {path.name: digest(path) for path in weight_paths}
    torch.manual_seed(SEED)
    model, loading = Qwen3VLGroundingForConditionalGeneration.from_pretrained(
        str(CHECKPOINT), torch_dtype=torch.bfloat16, attn_implementation=ATTENTION,
        local_files_only=True, output_loading_info=True)
    save_json(OUTPUT / 'loading_info.json', loading)
    assert not loading['missing_keys'], 'Missing parameters: refusing randomly initialized weights'
    assert not loading['mismatched_keys'] and not loading['error_msgs'], loading
    assert not loading['unexpected_keys'], 'Unexpected weights: inspect loading_info.json before proceeding'
    model = model.cuda().eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(str(CHECKPOINT), local_files_only=True)
    object_id = processor.tokenizer.convert_tokens_to_ids('<object>')
    assert object_id != processor.tokenizer.unk_token_id
    assert processor.tokenizer.encode('<object>', add_special_tokens=False) == [object_id]
    model.model.object_token_id = object_id
    lm = model.model.language_model
    assert len(lm.layers) == lm.config.num_hidden_layers > 0
    assert model.out_proj.weight.shape == (1, lm.config.hidden_size)
    assert not any(module.training for module in model.modules())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    sources = [Path(__file__), ROOT / 'tools/ref_e0_core.py', ROOT / 'tools/ref_e0_data.py',
               ROOT / 'tools/humanref_pipeline.py', ROOT / 'wedetect_ref/models/qwen3vl_referring.py',
               ROOT / 'wedetect_ref/models/vision_process.py', Path(inspect.getfile(type(lm)))]
    git = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True, check=True).stdout.strip()
    manifest = dict(stage='E0', created_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        transformers=transformers.__version__, torchvision=package_version('torchvision'),
        flash_attn=package_version('flash-attn'), gpu=torch.cuda.get_device_name(0),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), git_commit=git,
        checkpoint=str(CHECKPOINT.resolve()), checkpoint_sha256=checkpoint_hashes,
        checkpoint_json_sha256={p.name: digest(p) for p in sorted(CHECKPOINT.glob('*.json'))},
        source_sha256={str(path): digest(path) for path in sources},
        annotations=str(ANNOTATIONS.resolve()), annotation_sha256=digest(ANNOTATIONS),
        dataset_split=ANNOTATIONS.stem, proposal_file=str(PROPOSALS.resolve()),
        proposals_sha256=digest(PROPOSALS),
        image_sha256={str(path): digest(path) for path in paths},
        sample_ids=[row['id'] for row in selected], seed=SEED,
        domain_counts=dict(Counter(row['domain'] for row in selected)),
        candidate_source='Author REC proposal JSON keyed by image; first K clipped to bounds',
        max_proposals=MAX_PROPOSALS, gt_insertion=False, dtype=str(model.dtype),
        coordinate_policy='FP32 saved geometry; BF16 model box input',
        selection='Top-1 by Ref sigmoid score; ties use first original proposal',
        score_threshold=None, nms_iou=None, evaluation_iou=0.5,
        sigmoid_policy='model dtype sigmoid, then FP32 (matches existing pipeline)',
        attention=ATTENTION, text_attention=lm.config._attn_implementation,
        vision_attention=model.model.visual.config._attn_implementation,
        checkpoint_use_cache=lm.config.use_cache, num_layers=len(lm.layers),
        hidden_size=lm.config.hidden_size, object_token_id=object_id,
        final_norm_epsilon=lm.norm.variance_epsilon, warmup_per_sample=WARMUP_PASSES,
        timing_repeats=TIMING_REPEATS,
        strict_tolerance=dict(atol=STRICT_ATOL, rtol=STRICT_RTOL),
        compact_tolerance=dict(atol=COMPACT_ATOL, rtol=COMPACT_RTOL),
        excluded=['Uni generation', 'training', 'intermediate-layer sweep',
                  'decoder split/resume', 'visual pathway removal', 'benchmark accuracy claims'])
    save_json(OUTPUT / 'manifest.json', manifest)
    save_json(OUTPUT / 'model_config.json', model.config.to_dict())
    summaries = []
    pair_h0, pair_boxes, pair_pixels = None, None, None

    with torch.inference_mode():
        for index, (ann, path) in enumerate(zip(selected, paths)):
            image_start = time.perf_counter()
            with Image.open(path) as file:
                image = file.convert('RGB')
            image_ms = (time.perf_counter() - image_start) * 1000
            width, height = image.size
            boxes = [[max(0, min(width, b[0])), max(0, min(height, b[1])),
                      max(0, min(width, b[2])), max(0, min(height, b[3]))]
                     for b in proposals[ann['image_name']]['boxes'][:MAX_PROPOSALS]]
            validate_boxes(boxes)
            gt = [[max(0, min(width, b[0])), max(0, min(height, b[1])),
                   max(0, min(width, b[2])), max(0, min(height, b[3]))]
                  for b in ann['answer_boxes']]
            validate_boxes(gt)
            torch.cuda.synchronize()
            preparation_start = time.perf_counter()
            inputs, positions, prompt = prepare_input(model, processor, image, ann['referring'], boxes, object_id)
            torch.cuda.synchronize()
            preparation_ms = (time.perf_counter() - preparation_start) * 1000
            for _ in range(WARMUP_PASSES):
                warm = model(**inputs)
                del warm
            torch.cuda.synchronize()
            timing_rows, baseline = [], None
            for _ in range(TIMING_REPEATS):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                pred = model(**inputs)  # unmodified, unhooked baseline
                torch.cuda.synchronize()
                timing_rows.append((time.perf_counter() - begin) * 1000)
                current = object_logits(pred.logits, positions).clone()
                if baseline is None:
                    baseline = current
                    repeat_check = compare_tensors(baseline, baseline, STRICT_ATOL, STRICT_RTOL)
                else:
                    check = compare_tensors(baseline, current, STRICT_ATOL, STRICT_RTOL)
                    if check['max_abs'] >= repeat_check['max_abs']:
                        repeat_check = check
                    assert check['passed'], f'Unhooked repeated forward is unstable: {check}'
                del pred

            with capture_boundaries(model, positions) as captured:
                pred = model(**inputs)
            hooked = object_logits(pred.logits, positions).clone()
            del pred
            norm, head = lm.norm, model.out_proj
            # Same-shape original module chain and pure tensor chain, separately.
            native_full = head(norm(captured['hL_full']))
            functional_full = forward_algorithm(captured['hL_full'], norm.weight,
                                                 norm.variance_epsilon, head.weight, head.bias)
            compact = forward_algorithm(captured['hL'], norm.weight, norm.variance_epsilon,
                                        head.weight, head.bias)[:, 0]
            reconstructed = object_logits(functional_full, positions)
            checks = dict(repeated_forward=repeat_check,
                hooks_preserve_logits=compare_tensors(baseline, hooked, STRICT_ATOL, STRICT_RTOL),
                native_final_readout=compare_tensors(hooked, object_logits(native_full, positions),
                                                    STRICT_ATOL, STRICT_RTOL),
                functional_final_readout=compare_tensors(hooked, reconstructed, STRICT_ATOL, STRICT_RTOL),
                compact_final_readout=compare_tensors(hooked, compact, COMPACT_ATOL, COMPACT_RTOL))
            del native_full, functional_full
            del captured['hL_full']  # no large sequence snapshot retained during timing
            baseline_scores = scores_from_logits(baseline)
            decision_checks = {name: compare_decisions(boxes, baseline_scores, scores_from_logits(value),
                               gt)
                               for name, value in (('hooked', hooked), ('reconstructed', reconstructed),
                                                   ('compact', compact))}
            # Same image + same proposals, genuinely different annotated queries.
            if index == 0:
                pair_h0, pair_boxes = captured['h0'].cpu(), boxes
                pair_pixels = inputs['pixel_values'].cpu().clone()
            elif index == 1:
                assert ann['image_name'] == selected[0]['image_name'] and boxes == pair_boxes
                assert torch.equal(pair_pixels, inputs['pixel_values'].cpu()), 'Pair changed image preprocessing'
                checks['h0_query_invariant'] = compare_tensors(pair_h0, captured['h0'].cpu(),
                                                              STRICT_ATOL, STRICT_RTOL)
                pair_h0, pair_pixels = None, None

            # No capture hooks active during profiling. Never use capture time as latency.
            profiled, spans = profile_forward(model, inputs)
            checks['timing_hooks_preserve_logits'] = compare_tensors(
                baseline, object_logits(profiled.logits, positions), STRICT_ATOL, STRICT_RTOL)
            del profiled
            torch.cuda.synchronize()
            post_start = time.perf_counter()
            post = compare_decisions(boxes, baseline_scores, baseline_scores, gt)
            post_ms = (time.perf_counter() - post_start) * 1000
            # Post is only fixed CPU selection bookkeeping, excludes GPU score transfer.
            passed = all(c['passed'] for c in checks.values()) and all(
                d['top1_equal'] for d in decision_checks.values())
            record = dict(index=index, id=ann['id'], image_name=ann['image_name'],
                query=ann['referring'], domain=ann['domain'], sub_domain=ann['sub_domain'],
                answer_boxes=gt, boxes=boxes, boxes_sha256=canonical_hash(boxes),
                prompt=prompt, prompt_sha256=canonical_hash(prompt),
                num_candidates=len(boxes), sequence_length=positions.shape[1],
                num_visual_tokens=captured['num_visual_tokens'],
                deepstack_count=captured['deepstack_count'],
                object_token_positions=positions[0].nonzero()[:, 0].cpu().tolist(),
                logits=baseline.float().cpu().tolist(), scores=baseline_scores,
                hooked_logits=hooked.float().cpu().tolist(),
                reconstructed_logits=reconstructed.float().cpu().tolist(),
                compact_logits=compact.float().cpu().tolist(),
                checks=checks, decisions=decision_checks, passed=passed,
                timing=dict(image_load_ms=image_ms, preparation_ms=preparation_ms,
                    unhooked_forward_wall_ms=timing_rows, profile_event_spans=spans,
                    cpu_selection_check_ms=post_ms),
                selected_indices=post['baseline_indices'])
            save_json(OUTPUT / 'samples' / f'{index:04d}.json', record)
            write_tensor(OUTPUT / 'samples' / f'{index:04d}.pt', dict(
                id=ann['id'], boxes_sha256=canonical_hash(boxes),
                h0=captured['h0'].cpu(), hL_pre_norm=captured['hL'].cpu(),
                input_ids=inputs['input_ids'].cpu(), position_ids=captured['position_ids'].cpu(),
                model_boxes_bf16=inputs['bboxes'][0].cpu(),
                logits=baseline.cpu()), torch)
            print(f'E0 {index + 1}/{len(selected)} id={ann["id"]} '
                  f'{"PASS" if passed else "FAIL"} N={len(boxes)} S={positions.shape[1]} '
                  f'hook_error={checks["hooks_preserve_logits"]["max_abs"]:.6g} '
                  f'compact_error={checks["compact_final_readout"]["max_abs"]:.6g}', flush=True)
            summaries.append(record)
            assert passed, f'E0 failed for {ann["id"]}; inspect samples/{index:04d}.json, do not loosen tolerances blindly'
            del captured, inputs, positions, hooked, compact, reconstructed, baseline, current

    mean_spans = {key: statistics.mean(r['timing']['profile_event_spans'][key] for r in summaries)
                  for key in summaries[0]['timing']['profile_event_spans']}
    report = dict(status='PASSED', samples=len(summaries),
        checks='all numeric and decision gates passed; same-image h0 pair checked',
        worst_errors={key: max(row['checks'][key]['max_abs'] for row in summaries
                              if key in row['checks'])
                      for key in sorted({key for row in summaries for key in row['checks']})},
        mean_profile_event_spans_ms=mean_spans,
        llm_fraction_of_profiled_ref_forward=mean_spans['llm_ms'] / mean_spans['total_event_ms'],
        median_unhooked_forward_wall_ms=statistics.median(
            t for row in summaries for t in row['timing']['unhooked_forward_wall_ms']),
        pending=['Actual decoder split/resume gate (before V interventions)',
                 'Full validation/test task evaluation (E0 subset is not a benchmark)',
                 'Uni-inclusive end-to-end latency'],
        interpretation='Engineering validation only. No accuracy improvement or speedup established.')
    report['engineering_subset_task_checks'] = dict(
        baseline_top1_acc_iou50=statistics.mean(r['decisions']['hooked']['baseline_correct'] for r in summaries),
        candidate_coverage_iou50=statistics.mean(r['decisions']['hooked']['candidate_covers_gt'] for r in summaries),
        note='Selected engineering subset; not the full split or a paper-comparable accuracy.')
    save_json(OUTPUT / 'summary.json', report)
    with (OUTPUT / 'summary.md').open('x', encoding='utf-8') as stream:
        stream.write('# E0: PASSED\n\n')
        stream.write(f'Checked {len(summaries)} {ANNOTATIONS.stem} expressions, '
                     'fixed author candidates, Top-1, no score cutoff/NMS, frozen Ref.\n\n')
        stream.write('| Check | Worst absolute error |\n|---|---:|\n')
        for key, error in report['worst_errors'].items():
            stream.write(f'| {key} | {error:.8g} |\n')
        stream.write('\n## Separate timing pass\n\n| Interval | Mean ms |\n|---|---:|\n')
        for key, value in mean_spans.items():
            stream.write(f'| {key} | {value:.3f} |\n')
        stream.write('\nThese are instrumented CUDA event spans, not pure kernel times. '
                     'Unhooked wall timings are saved separately. ROI/input assembly is a combined interval. '
                     'Proposals are fixed: no Uni latency or end-to-end speedup is reported.\n\n')
        stream.write('Pending: decoder split/resume validation before V; RefCOCOg evaluation; '
                     'representative latency benchmark. Passing E0 does not establish early-exit feasibility.\n')
    print(f'All E0 gates passed. Read {OUTPUT / "summary.md"}', flush=True)


if __name__ == '__main__':
    # Keep a failure artifact without swallowing the exception or producing a PASS.
    output_existed = OUTPUT.exists()
    try:
        run()
    except Exception as error:
        if not output_existed and OUTPUT.is_dir():
            save_json(OUTPUT / 'FAILED.json', dict(status='FAILED', error=repr(error),
                                                  traceback=traceback.format_exc()))
        raise
