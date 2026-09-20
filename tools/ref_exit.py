"""True static Ref exit + paired accuracy/latency. Run from repository root.

See REF_EXIT.md. Requires the existing PASSED P-native run, not its .pt caches.
Never changes the checkpoint, E0/P-native code, or prior experiment results.
"""
from contextlib import nullcontext
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

from humanref_pipeline import ROOT, digest, image_path, save_json, validate_boxes
from ref_e0 import (ATTENTION, EXPECTED_TRANSFORMERS, MAX_PROPOSALS, SEED,
                    STRICT_ATOL, STRICT_RTOL, COMPACT_ATOL, COMPACT_RTOL,
                    prepare_input, scores_from_logits, package_version)
from ref_e0_data import choose_samples, load_rec_annotations
from ref_pnative import read_json, check_e0_protocol
from ref_exit_analysis import decisions, summarize, markdown

PNATIVE = Path(os.environ.get('EXIT_PNATIVE', ROOT / 'results/ref_pnative_refcocog_val500'))
ANNOTATIONS = Path(os.environ.get('EXIT_ANN', ROOT / 'wedetect_ref/eval_grounding/eval_refcoco/refcocog_validation.json'))
PROPOSALS = Path(os.environ.get('EXIT_PROPOSALS', ROOT / 'wedetect_ref/eval_grounding/eval_refcoco/refcoco_proposals_all.json'))
IMAGES = Path(os.environ.get('EXIT_IMAGES', ROOT / 'data/coco2014'))
CHECKPOINT = Path(os.environ.get('EXIT_REF', ROOT / 'checkpoints/WeDetect-Ref-4B'))
SAMPLE_COUNT = int(os.environ.get('EXIT_N', '6'))
DEPTH = int(os.environ.get('EXIT_DEPTH', '30'))
OUTPUT = Path(os.environ.get('EXIT_OUT', ROOT / f'results/ref_exit_d{DEPTH}_refcocog_val{SAMPLE_COUNT}'))
WARMUP = 2
REPEATS = 3


def clipped_boxes(boxes, width, height):
    result = [[max(0, min(width, b[0])), max(0, min(height, b[1])),
               max(0, min(width, b[2])), max(0, min(height, b[3]))] for b in boxes]
    validate_boxes(result)
    return result


def run():
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor
    from models.qwen3vl_referring import Qwen3VLGroundingForConditionalGeneration
    from ref_e0_core import compare_tensors, forward_algorithm, object_logits
    from ref_exit_core import audit_execution, decoder_prefix

    assert __debug__ and int(os.environ.get('WORLD_SIZE', '1')) == 1
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert transformers.__version__ == EXPECTED_TRANSFORMERS
    assert not OUTPUT.exists(), f'Refusing overwrite: {OUTPUT}; choose a new EXIT_OUT'
    assert CHECKPOINT.is_dir() and IMAGES.is_dir()
    prior, prior_summary = read_json(PNATIVE / 'manifest.json'), read_json(PNATIVE / 'summary.json')
    assert prior['stage'] == 'P-native' and prior_summary['status'] == 'PASSED'
    assert not (PNATIVE / 'FAILED.json').exists()
    assert len(prior['sample_ids']) == prior_summary['samples']
    assert 2 <= SAMPLE_COUNT <= prior_summary['samples']
    assert DEPTH in prior['depths'] and 0 < DEPTH < prior['num_layers']
    rows, proposals = load_rec_annotations(ANNOTATIONS, PROPOSALS)
    selected = choose_samples(rows, SAMPLE_COUNT, SEED)
    assert [r['id'] for r in selected] == prior['sample_ids'][:SAMPLE_COUNT]
    assert selected == read_json(PNATIVE / 'selection.json')['rows'][:SAMPLE_COUNT]
    reference_paths = [PNATIVE / 'samples' / f'{i:05d}.json' for i in range(SAMPLE_COUNT)]
    references = [read_json(p) for p in reference_paths]
    assert all(r['passed'] and r['id'] == a['id'] for r, a in zip(references, selected))
    paths = [image_path(IMAGES, a['image_name']) for a in selected]
    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'samples').mkdir()
    print(f'Static exit: depth={DEPTH}, N={SAMPLE_COUNT}, output={OUTPUT.resolve()}', flush=True)
    print('Hashing and loading model; not timed.', flush=True)
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
    lm, head = model.model.language_model, model.out_proj
    original_layers = lm.layers
    total = len(original_layers)
    assert total == lm.config.num_hidden_layers and not any(m.training for m in model.modules())
    sources = [Path(__file__), ROOT / 'tools/ref_exit_core.py', ROOT / 'tools/ref_exit_analysis.py',
        ROOT / 'tools/ref_pnative.py', ROOT / 'tools/ref_pnative_core.py', ROOT / 'tools/ref_pnative_analysis.py',
        ROOT / 'tools/ref_e0.py', ROOT / 'tools/ref_e0_core.py', ROOT / 'tools/ref_e0_data.py',
        ROOT / 'tools/humanref_pipeline.py', ROOT / 'wedetect_ref/models/qwen3vl_referring.py',
        ROOT / 'wedetect_ref/models/vision_process.py', Path(inspect.getfile(type(lm)))]
    manifest = dict(stage='static-exit', version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        transformers=transformers.__version__, torchvision=package_version('torchvision'),
        flash_attn=package_version('flash-attn'), gpu=torch.cuda.get_device_name(0),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        git_commit=subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=True).stdout.strip(),
        checkpoint=str(CHECKPOINT.resolve()), checkpoint_sha256=hashes,
        checkpoint_json_sha256={p.name: digest(p) for p in sorted(CHECKPOINT.glob('*.json'))},
        source_sha256={str(p): digest(p) for p in sources},
        pnative_manifest_sha256=digest(PNATIVE / 'manifest.json'),
        pnative_summary_sha256=digest(PNATIVE / 'summary.json'),
        pnative_samples_sha256={p.name: digest(p) for p in reference_paths},
        annotations=str(ANNOTATIONS.resolve()), annotation_sha256=digest(ANNOTATIONS),
        proposals=str(PROPOSALS.resolve()), proposals_sha256=digest(PROPOSALS),
        image_sha256={str(p): digest(p) for p in paths}, sample_ids=[a['id'] for a in selected],
        seed=SEED, max_proposals=MAX_PROPOSALS, gt_insertion=False,
        coordinate_policy='FP32 saved geometry; BF16 model box input',
        selection='Top-1 by Ref sigmoid score; ties use first original proposal',
        score_threshold=None, nms_iou=None, evaluation_iou=0.5,
        sigmoid_policy='model dtype sigmoid, then FP32 (matches existing pipeline)',
        diagnostic_selection='Top-1 by raw BF16 logits converted to FP32; first original proposal on ties; BOTH arms',
        dtype=str(model.dtype), attention=ATTENTION, text_attention=lm.config._attn_implementation,
        vision_attention=model.model.visual.config._attn_implementation,
        checkpoint_use_cache=lm.config.use_cache, num_layers=total, hidden_size=lm.config.hidden_size,
        object_token_id=object_id, final_norm_epsilon=lm.norm.variance_epsilon,
        strict_tolerance=dict(atol=STRICT_ATOL, rtol=STRICT_RTOL),
        compact_tolerance=dict(atol=COMPACT_ATOL, rtol=COMPACT_RTOL),
        exit_depth=DEPTH, training=False, actual_early_exit=True,
        readout='Original full-sequence final norm/head; compact P-native comparison is separately checked',
        warmup_per_arm_per_scope=WARMUP, repeats_per_arm_per_scope=REPEATS,
        timing_order='Alternating full/exit by sample, scope, and repetition; no capture/check hooks',
        request_scope='Warm image decode + preparation/H2D + Ref + object-score D2H + BOTH Top-1 selections',
        timing_excluded=['Uni', 'candidate loading/clipping', 'GT/evaluation', 'model loading',
                         'hashing', 'layer-list switching', 'verification', 'report I/O'],
        tail_parameters_remain_resident=True, cache_reuse=False)
    save_json(OUTPUT / 'manifest.json', manifest)
    check_e0_protocol(manifest, prior)
    save_json(OUTPUT / 'selection.json', dict(rows=selected, seed=SEED))
    save_json(OUTPUT / 'model_config.json', model.config.to_dict())
    records = []

    with torch.inference_mode():
        for index, (ann, path, old) in enumerate(zip(selected, paths, references)):
            old_hashes = [sha for name, sha in prior['image_sha256'].items()
                          if name.replace('\\', '/').endswith('/' + ann['image_name'])]
            assert old_hashes and all(sha == manifest['image_sha256'][str(path)] for sha in old_hashes)
            with Image.open(path) as source:
                image = source.convert('RGB')
            boxes = clipped_boxes(proposals[ann['image_name']]['boxes'][:MAX_PROPOSALS], *image.size)
            gt = clipped_boxes(ann['answer_boxes'], *image.size)
            inputs, positions, prompt = prepare_input(model, processor, image, ann['referring'], boxes, object_id)
            assert (old['image_name'], old['query'], old['boxes'], old['answer_boxes'], old['prompt']) == (
                ann['image_name'], ann['referring'], boxes, gt, prompt)
            assert old['object_token_positions'] == positions[0].nonzero()[:, 0].cpu().tolist()
            deepstack = old['deepstack_count']
            assert 0 < deepstack <= DEPTH
            assert 'past_key_values' not in inputs
            # Plain full reference, followed by a read-only boundary capture.
            pred = model(**inputs)
            baseline = object_logits(pred.logits, positions).clone()
            del pred
            with audit_execution(lm, head, DEPTH, total) as captured:
                pred = model(**inputs)
            checks = dict(hooks_preserve_full=compare_tensors(
                baseline, object_logits(pred.logits, positions), STRICT_ATOL, STRICT_RTOL))
            del pred
            boundary = captured['boundary_full']
            expected_full = object_logits(forward_algorithm(boundary, lm.norm.weight,
                lm.norm.variance_epsilon, head.weight, head.bias), positions)
            expected_compact = forward_algorithm(boundary[positions], lm.norm.weight,
                lm.norm.variance_epsilon, head.weight, head.bias)[:, 0]
            checks['saved_pnative_full'] = compare_tensors(torch.tensor(old['baseline_logits']),
                baseline.float().cpu(), STRICT_ATOL, STRICT_RTOL)
            checks['saved_pnative_compact_exit'] = compare_tensors(torch.tensor(old['layers'][str(DEPTH)]['logits']),
                expected_compact.float().cpu(), STRICT_ATOL, STRICT_RTOL)
            # A full-depth sham through the same layer-list replacement mechanism.
            with audit_execution(lm, head, total, total) as sham_audit:
                with decoder_prefix(lm, total, deepstack):
                    pred = model(**inputs)
            checks['full_depth_sham'] = compare_tensors(baseline,
                object_logits(pred.logits, positions), STRICT_ATOL, STRICT_RTOL)
            del pred, sham_audit
            # True exit. The tail blocks are registered with audit hooks but MUST NOT run.
            with audit_execution(lm, head, DEPTH, DEPTH) as exit_audit:
                with decoder_prefix(lm, DEPTH, deepstack):
                    pred = model(**inputs)
            early = object_logits(pred.logits, positions).clone()
            del pred
            checks['exit_boundary'] = compare_tensors(boundary, exit_audit['final_full'], STRICT_ATOL, STRICT_RTOL)
            checks['exit_full_shape_readout'] = compare_tensors(expected_full, early, STRICT_ATOL, STRICT_RTOL)
            checks['exit_compact_readout'] = compare_tensors(expected_compact, early, COMPACT_ATOL, COMPACT_RTOL)
            assert captured['deepstack_count'] == exit_audit['deepstack_count'] == deepstack
            assert lm.layers is original_layers
            values = {'full': baseline, 'exit': early}
            logits = {arm: value.float().cpu().tolist() for arm, value in values.items()}
            scores = {arm: scores_from_logits(value) for arm, value in values.items()}
            choice = {arm: decisions(boxes, gt, logits[arm], scores[arm]) for arm in values}
            cached_choice = decisions(boxes, gt, old['layers'][str(DEPTH)]['logits'], old['layers'][str(DEPTH)]['scores'])
            same_decisions = all(choice['exit'][p]['index'] == cached_choice[p]['index'] for p in cached_choice)
            same_scores = scores['full'] == old['baseline_scores']
            record = dict(index=index, id=ann['id'], image_name=ann['image_name'], query=ann['referring'],
                boxes=boxes, answer_boxes=gt, logits=logits, scores=scores, decisions=choice,
                checks=checks, passed=all(c['passed'] for c in checks.values()) and same_decisions and same_scores,
                pnative_exit_decisions_equal=same_decisions, pnative_full_scores_equal=same_scores,
                executed_full=captured['blocks'], executed_exit=exit_audit['blocks'],
                deepstack_count=deepstack, norm_calls_exit=exit_audit['norm_calls'], head_calls_exit=exit_audit['head_calls'])
            del captured, exit_audit, boundary, expected_full, expected_compact
            sample_path = OUTPUT / 'samples' / f'{index:05d}.json'
            if not record['passed']:
                save_json(sample_path, record)
                raise AssertionError(f'Consistency/decision failure: {sample_path}; do not relax gates blindly')

            # Only model execution and real request work are inside timed regions.
            # Static prefix setup, audit, GT metrics, and tensor checks are outside.
            assert all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules())

            def request():
                with Image.open(path) as source:
                    fresh_image = source.convert('RGB')
                fresh, mask, _ = prepare_input(model, processor, fresh_image, ann['referring'], boxes, object_id)
                output = model(**fresh)
                v = output.logits[mask][:, 0]
                raw, sig = v.float().cpu().tolist(), v.sigmoid().float().cpu().tolist()
                # No GT or evaluation metrics in a production request.
                selected_indices = {name: max(range(len(vs)), key=vs.__getitem__)
                                    for name, vs in (('raw_logit', raw), ('bf16_sigmoid', sig))}
                return raw, sig, selected_indices

            timing = {}
            for scope_index, scope in enumerate(('forward', 'request')):
                callback = (lambda: model(**inputs)) if scope == 'forward' else request
                timing[scope] = []
                for arm in ('full', 'exit'):
                    context = decoder_prefix(lm, DEPTH, deepstack) if arm == 'exit' else nullcontext()
                    with context:
                        for _ in range(WARMUP):
                            warm = callback()
                            del warm
                    torch.cuda.synchronize()
                for repeat in range(REPEATS):
                    order = ('full', 'exit') if (index + scope_index + repeat) % 2 == 0 else ('exit', 'full')
                    for arm in order:
                        context = decoder_prefix(lm, DEPTH, deepstack) if arm == 'exit' else nullcontext()
                        with context:
                            torch.cuda.synchronize()
                            begin = time.perf_counter()
                            output = callback()
                            torch.cuda.synchronize()
                            elapsed = (time.perf_counter() - begin) * 1000
                        # Outside timing: ensure timed forwards produce the audited decisions.
                        if scope == 'forward':
                            v = object_logits(output.logits, positions)
                            raw, sig = v.float().cpu().tolist(), scores_from_logits(v)
                            timed_indices = {name: max(range(len(vs)), key=vs.__getitem__)
                                             for name, vs in (('raw_logit', raw), ('bf16_sigmoid', sig))}
                            del v
                        else:
                            raw, sig, timed_indices = output
                        check = compare_tensors(torch.tensor(logits[arm]), torch.tensor(raw), STRICT_ATOL, STRICT_RTOL)
                        stable = (check['passed'] and sig == scores[arm]
                                  and all(timed_indices[p] == choice[arm][p]['index'] for p in timed_indices))
                        if not stable:
                            record.update(passed=False, timing=timing, timing_failure=dict(
                                scope=scope, arm=arm, repeat=repeat, check=check,
                                logits=raw, scores=sig, indices=timed_indices))
                            save_json(sample_path, record)
                            raise AssertionError(f'Unstable timed output: {ann["id"]} {scope} {arm}; see {sample_path}')
                        timing[scope].append(dict(repeat=repeat, arm=arm, wall_ms=elapsed,
                                                  logits_max_abs=check['max_abs']))
                        del output
                assert lm.layers is original_layers
            record['timing'] = timing
            save_json(sample_path, record)
            records.append(record)
            print(f'Exit {index+1}/{SAMPLE_COUNT} {ann["id"]} PASS '
                  f'sigmoid={int(choice["full"]["bf16_sigmoid"]["correct"])}->{int(choice["exit"]["bf16_sigmoid"]["correct"])} '
                  f'raw={int(choice["full"]["raw_logit"]["correct"])}->{int(choice["exit"]["raw_logit"]["correct"])}', flush=True)
            del inputs, positions, baseline, early, values
    report, errors = summarize(records)
    report.update(exit_depth=DEPTH, total_depth=total, actual_early_exit=True)
    save_json(OUTPUT / 'summary.json', report)
    save_json(OUTPUT / 'paired_errors.json', errors)
    with (OUTPUT / 'summary.md').open('x', encoding='utf-8') as stream:
        stream.write(markdown(report, DEPTH))
    print(f'Completed: {OUTPUT / "summary.md"}', flush=True)


if __name__ == '__main__':
    existed = OUTPUT.exists()
    try:
        run()
    except Exception as error:
        if not existed and OUTPUT.is_dir():
            save_json(OUTPUT / 'FAILED.json', dict(status='FAILED', error=repr(error), traceback=traceback.format_exc()))
        raise
