"""P-native: frozen per-depth readout on fixed REC proposals. See REF_PNATIVE.md."""
from datetime import datetime, timezone
import csv
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import traceback

from humanref_pipeline import ROOT, digest, image_path, iou, save_json, validate_boxes
from ref_e0 import (ATTENTION, EXPECTED_TRANSFORMERS, MAX_PROPOSALS, SEED,
                    STRICT_ATOL, STRICT_RTOL, COMPACT_ATOL, COMPACT_RTOL,
                    prepare_input, scores_from_logits, write_tensor, package_version)
from ref_e0_data import canonical_hash, choose_samples, load_rec_annotations
from ref_pnative_analysis import selected_depths, layer_result, summarize, query_pairs

ANNOTATIONS = Path(os.environ.get('P_ANN', ROOT / 'wedetect_ref/eval_grounding/eval_refcoco/refcocog_validation.json'))
PROPOSALS = Path(os.environ.get('P_PROPOSALS', ROOT / 'wedetect_ref/eval_grounding/eval_refcoco/refcoco_proposals_all.json'))
IMAGES = Path(os.environ.get('P_IMAGES', ROOT / 'data/coco2014'))
CHECKPOINT = Path(os.environ.get('P_REF', ROOT / 'checkpoints/WeDetect-Ref-4B'))
E0 = Path(os.environ.get('P_E0', ROOT / 'results/ref_e0_refcocog_val60'))
SAMPLE_COUNT = int(os.environ.get('P_N', '60'))
OUTPUT = Path(os.environ.get('P_OUT', ROOT / f'results/ref_pnative_refcocog_val{SAMPLE_COUNT}'))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_e0_protocol(current, reference):
    # Paths may move; content and numeric protocol may not silently change.
    for key in ('annotation_sha256', 'proposals_sha256', 'checkpoint_sha256',
                'checkpoint_json_sha256', 'torch', 'transformers', 'torchvision', 'flash_attn',
                'num_layers', 'hidden_size', 'object_token_id', 'dtype', 'attention',
                'text_attention', 'vision_attention', 'checkpoint_use_cache', 'final_norm_epsilon',
                'seed', 'max_proposals', 'gt_insertion', 'coordinate_policy',
                'selection', 'score_threshold', 'nms_iou', 'evaluation_iou', 'sigmoid_policy',
                'strict_tolerance', 'compact_tolerance'):
        assert current[key] == reference[key], f'E0 protocol mismatch: {key}'
    prior = {Path(path).name: value for path, value in reference['source_sha256'].items()}
    now = {Path(path).name: value for path, value in current['source_sha256'].items()}
    assert len(prior) == len(reference['source_sha256'])
    assert len(now) == len(current['source_sha256']), 'Ambiguous source filenames'
    for name, sha in prior.items():
        assert now[name] == sha, f'Validated source changed: {name}; do not mix with old E0'


def write_reports(records, depths):
    summary, errors = summarize(records, depths)
    pairs = query_pairs(records, depths)
    summary.update(status='PASSED', stage='P-native', depths=depths,
                   e0_overlap_checked=sum('saved_e0_logits' in r['checks'] for r in records),
                   checks_max_abs={key: max(r['checks'][key]['max_abs'] for r in records if key in r['checks'])
                                   for key in sorted({key for r in records for key in r['checks']})},
                   note='Frozen original-head readout only; no training, actual early exit, or speedup claim.')
    save_json(OUTPUT / 'summary.json', summary)
    save_json(OUTPUT / 'paired_errors.json', errors)
    save_json(OUTPUT / 'query_pairs.json', pairs)
    with (OUTPUT / 'curve.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary['curve'][0]))
        writer.writeheader()
        writer.writerows(summary['curve'])
    with (OUTPUT / 'summary.md').open('x', encoding='utf-8') as stream:
        stream.write('# P-native: completed, all consistency gates passed\n\n')
        stream.write(f'{summary["samples"]} expressions / {summary["images"]} images; '
                     f'baseline accuracy {summary["baseline_accuracy"]:.2%}; '
                     f'candidate coverage {summary["candidate_coverage"]:.2%}.\n\n')
        stream.write('| Depth | Top-1 Acc | Delta pp | Index differs | Harmed | Recovered | Raw-logit Acc* | Tied Top-1 |\n'
                     '|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for row in summary['curve']:
            stream.write(f'| {row["depth"]} | {row["accuracy"]:.2%} | {row["delta_pp"]:+.2f} '
                         f'| {row["index_disagreement_rate"]:.2%} | {row["harmed_count"]} '
                         f'| {row["recovered_count"]} | {row["raw_logit_accuracy"]:.2%} | {row["tied_top1_count"]} |\n')
        stream.write('\nPrimary ranking preserves E0 BF16-sigmoid scores and first-index tie handling. '
                     '*Raw-logit accuracy is a separate rounding/saturation diagnostic, not a replacement metric.\n\n'
                     'Harmed = baseline correct / this depth wrong; recovered = reverse. '
                     'Different candidate index need not imply a task error. See curve.csv for conditional denominators.\n\n')
        stream.write(f'Same-image distinct-target query pairs: {pairs["pair_count"]}; see query_pairs.json. '
                     'Pairs can share expressions and are not independent observations.\n\n'
                     'This is a selected validation subset, not an independent test. Native-head failure does not '
                     'show that a trained probe cannot read the representation. All decoder layers executed; '
                     'cache extraction does not measure early-exit latency.\n')


def run():
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor
    from models.qwen3vl_referring import Qwen3VLGroundingForConditionalGeneration
    from ref_e0_core import compare_tensors, forward_algorithm, object_logits
    from ref_pnative_core import capture_depths, read_depths

    assert __debug__ and int(os.environ.get('WORLD_SIZE', '1')) == 1
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert transformers.__version__ == EXPECTED_TRANSFORMERS
    for path in (ANNOTATIONS, PROPOSALS, E0 / 'summary.json', E0 / 'manifest.json'):
        assert path.is_file(), f'Missing required input: {path}'
    assert IMAGES.is_dir() and CHECKPOINT.is_dir()
    assert not OUTPUT.exists(), f'Refusing to overwrite {OUTPUT}; set a new P_OUT'
    e0_summary, e0_manifest = read_json(E0 / 'summary.json'), read_json(E0 / 'manifest.json')
    assert e0_summary['status'] == 'PASSED' and not (E0 / 'FAILED.json').exists()
    e0_records = [read_json(p) for p in sorted((E0 / 'samples').glob('*.json'))]
    assert len(e0_records) == e0_summary['samples'] == len(e0_manifest['sample_ids'])
    assert [r['id'] for r in e0_records] == e0_manifest['sample_ids']
    assert len({r['id'] for r in e0_records}) == len(e0_records) and all(r['passed'] for r in e0_records)
    reference = {r['id']: r for r in e0_records}
    rows, proposals = load_rec_annotations(ANNOTATIONS, PROPOSALS)
    selected = choose_samples(rows, SAMPLE_COUNT, SEED)
    assert any(r['id'] in reference for r in selected), 'No E0 overlap to check'
    paths = [image_path(IMAGES, row['image_name']) for row in selected]
    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'samples').mkdir()
    save_json(OUTPUT / 'selection.json', dict(rows=selected, seed=SEED,
              annotation_sha256=digest(ANNOTATIONS), proposals_sha256=digest(PROPOSALS)))
    print(f'P-native: {OUTPUT.resolve()}\nHashing/loading frozen model...', flush=True)
    weights = sorted(list(CHECKPOINT.glob('*.safetensors')) + list(CHECKPOINT.glob('*.bin')))
    assert weights
    hashes = {p.name: digest(p) for p in weights}
    assert hashes == e0_manifest['checkpoint_sha256'], 'Checkpoint differs from E0'
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
    assert not any(m.training for m in model.modules())
    assert all(not p.requires_grad for p in model.parameters())
    assert len(lm.layers) == lm.config.num_hidden_layers
    depths = selected_depths(len(lm.layers))
    sources = [Path(__file__), ROOT / 'tools/ref_pnative_core.py', ROOT / 'tools/ref_pnative_analysis.py',
               ROOT / 'tools/ref_e0.py', ROOT / 'tools/ref_e0_core.py', ROOT / 'tools/ref_e0_data.py',
               ROOT / 'tools/humanref_pipeline.py', ROOT / 'wedetect_ref/models/qwen3vl_referring.py',
               ROOT / 'wedetect_ref/models/vision_process.py', Path(inspect.getfile(type(lm)))]
    manifest = dict(stage='P-native', version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        transformers=transformers.__version__, torchvision=package_version('torchvision'),
        flash_attn=package_version('flash-attn'), gpu=torch.cuda.get_device_name(0),
        git_commit=subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=True).stdout.strip(),
        checkpoint=str(CHECKPOINT.resolve()), checkpoint_sha256=hashes,
        checkpoint_json_sha256={p.name: digest(p) for p in sorted(CHECKPOINT.glob('*.json'))},
        source_sha256={str(p): digest(p) for p in sources},
        e0_manifest_sha256=digest(E0 / 'manifest.json'), e0_summary_sha256=digest(E0 / 'summary.json'),
        e0_samples_sha256={p.name: digest(p) for p in sorted((E0 / 'samples').glob('*.json'))},
        annotations=str(ANNOTATIONS.resolve()), annotation_sha256=digest(ANNOTATIONS),
        proposals=str(PROPOSALS.resolve()), proposals_sha256=digest(PROPOSALS),
        image_sha256={str(p): digest(p) for p in paths},
        sample_ids=[r['id'] for r in selected], dataset_split=ANNOTATIONS.stem,
        seed=SEED, max_proposals=MAX_PROPOSALS, gt_insertion=False,
        coordinate_policy='FP32 saved geometry; BF16 model box input',
        selection='Top-1 by Ref sigmoid score; ties use first original proposal',
        score_threshold=None, nms_iou=None, evaluation_iou=0.5,
        sigmoid_policy='model dtype sigmoid, then FP32 (matches existing pipeline)',
        dtype=str(model.dtype), attention=ATTENTION, text_attention=lm.config._attn_implementation,
        vision_attention=model.model.visual.config._attn_implementation,
        checkpoint_use_cache=lm.config.use_cache, num_layers=len(lm.layers), hidden_size=lm.config.hidden_size,
        object_token_id=object_id, final_norm_epsilon=lm.norm.variance_epsilon,
        strict_tolerance=dict(atol=STRICT_ATOL, rtol=STRICT_RTOL),
        compact_tolerance=dict(atol=COMPACT_ATOL, rtol=COMPACT_RTOL), depths=depths,
        boundary='k<L: pre-hook of block k (0-based), after k completed blocks and DeepStack; L: pre final norm',
        cache_axis_order='depth,candidate,hidden', training=False, actual_early_exit=False)
    save_json(OUTPUT / 'manifest.json', manifest)
    check_e0_protocol(manifest, e0_manifest)
    save_json(OUTPUT / 'model_config.json', model.config.to_dict())
    write_tensor(OUTPUT / 'native_head.pt', dict(norm_weight=lm.norm.weight.detach().cpu(),
        norm_epsilon=lm.norm.variance_epsilon, head_weight=head.weight.detach().cpu(),
        head_bias=head.bias.detach().cpu()), torch)
    manifest_sha256 = digest(OUTPUT / 'manifest.json')
    native_head_sha256 = digest(OUTPUT / 'native_head.pt')
    print(f'Depths={depths}; cache approx {SAMPLE_COUNT * len(depths) * MAX_PROPOSALS * lm.config.hidden_size * 2 / 1e9:.2f} GB', flush=True)
    records, pair_h0, pair_pixels = [], None, None
    with torch.inference_mode():
        for index, (ann, path) in enumerate(zip(selected, paths)):
            with Image.open(path) as file:
                image = file.convert('RGB')
            width, height = image.size
            boxes = [[max(0, min(width, b[0])), max(0, min(height, b[1])),
                      max(0, min(width, b[2])), max(0, min(height, b[3]))]
                     for b in proposals[ann['image_name']]['boxes'][:MAX_PROPOSALS]]
            gt = [[max(0, min(width, b[0])), max(0, min(height, b[1])),
                   max(0, min(width, b[2])), max(0, min(height, b[3]))] for b in ann['answer_boxes']]
            validate_boxes(boxes)
            validate_boxes(gt)
            inputs, positions, prompt = prepare_input(model, processor, image, ann['referring'], boxes, object_id)
            # One plain reference pass + one capture pass, not six separate model runs.
            pred = model(**inputs)
            baseline = object_logits(pred.logits, positions).clone()
            del pred
            with capture_depths(model, positions, depths) as captured:
                pred = model(**inputs)
            hooked = object_logits(pred.logits, positions).clone()
            del pred
            outputs = read_depths(captured['states'], lm.norm.weight, lm.norm.variance_epsilon, head.weight, head.bias)
            full = forward_algorithm(captured.pop('hL_full'), lm.norm.weight,
                                     lm.norm.variance_epsilon, head.weight, head.bias)
            checks = dict(hooks_preserve_logits=compare_tensors(baseline, hooked, STRICT_ATOL, STRICT_RTOL),
                full_final_readout=compare_tensors(baseline, object_logits(full, positions), STRICT_ATOL, STRICT_RTOL),
                compact_final_readout=compare_tensors(baseline, outputs[depths[-1]], COMPACT_ATOL, COMPACT_RTOL))
            del full
            scores = scores_from_logits(baseline)
            layers = {}
            for k in depths:
                logits = outputs[k].float().cpu().tolist()
                current_scores = scores_from_logits(outputs[k])
                layers[str(k)] = dict(logits=logits, scores=current_scores,
                    decision=layer_result(boxes, gt, scores, current_scores, logits))
            if ann['id'] in reference:
                old = reference[ann['id']]
                assert (old['query'], old['image_name'], old['boxes'], old['answer_boxes'], old['prompt']) == (
                    ann['referring'], ann['image_name'], boxes, gt, prompt), 'E0 input changed'
                old_image_hashes = [sha for name, sha in e0_manifest['image_sha256'].items()
                                    if name.endswith('/' + ann['image_name'])]
                assert old_image_hashes and all(sha == manifest['image_sha256'][str(path)] for sha in old_image_hashes)
                checks['saved_e0_logits'] = compare_tensors(torch.tensor(old['logits']), baseline.float().cpu(), STRICT_ATOL, STRICT_RTOL)
                assert scores == old['scores'], 'Saved E0 scores changed'
                assert layers[str(depths[-1])]['decision']['actual_top1'] == old['decisions']['hooked']['baseline_top1']
            if index == 0:
                pair_h0, pair_pixels = captured['states'][0].cpu(), inputs['pixel_values'].cpu().clone()
                pair_boxes = boxes
            elif index == 1:
                assert ann['image_name'] == selected[0]['image_name'] and boxes == pair_boxes
                assert torch.equal(pair_pixels, inputs['pixel_values'].cpu())
                checks['h0_query_invariant'] = compare_tensors(pair_h0, captured['states'][0].cpu(), STRICT_ATOL, STRICT_RTOL)
                pair_h0, pair_pixels = None, None
            passed = all(c['passed'] for c in checks.values()) and layers[str(depths[-1])]['decision']['top1_equal']
            record = dict(index=index, id=ann['id'], image_name=ann['image_name'], query=ann['referring'],
                boxes=boxes, boxes_sha256=canonical_hash(boxes), answer_boxes=gt, prompt=prompt,
                num_candidates=len(boxes), sequence_length=positions.shape[1],
                num_visual_tokens=captured['num_visual_tokens'], deepstack_count=captured['deepstack_count'],
                object_token_positions=positions[0].nonzero()[:, 0].cpu().tolist(),
                baseline_logits=baseline.float().cpu().tolist(), baseline_scores=scores,
                layers=layers, checks=checks, passed=passed)
            save_json(OUTPUT / 'samples' / f'{index:05d}.json', record)
            states = torch.stack([captured['states'][k].cpu() for k in depths])
            assert states.shape == (len(depths), len(boxes), lm.config.hidden_size)
            overlaps = [iou(b, gt[0]) for b in boxes]
            write_tensor(OUTPUT / 'samples' / f'{index:05d}.pt', dict(
                id=ann['id'], image_name=ann['image_name'], query=ann['referring'], depths=depths,
                manifest_sha256=manifest_sha256, native_head_sha256=native_head_sha256,
                sample_record_sha256=digest(OUTPUT / 'samples' / f'{index:05d}.json'),
                hidden_pre_norm=states, boxes=torch.tensor(boxes, dtype=torch.float32),
                boxes_sha256=canonical_hash(boxes), gt_boxes=torch.tensor(gt, dtype=torch.float32),
                max_gt_iou=torch.tensor(overlaps, dtype=torch.float32),
                iou_soft_labels=torch.tensor([x if x > .5 else 0 for x in overlaps], dtype=torch.float32),
                input_ids=inputs['input_ids'].cpu(), position_ids=captured['position_ids'].cpu(),
                object_positions=positions.cpu(), model_boxes_bf16=inputs['bboxes'][0].cpu(),
                baseline_logits=baseline.cpu(), native_logits=torch.stack([outputs[k].cpu() for k in depths])), torch)
            print(f'P-native {index+1}/{SAMPLE_COUNT} {ann["id"]} {"PASS" if passed else "FAIL"} '
                  + ' '.join(f'{k}:{int(layers[str(k)]["decision"]["actual_correct"])}' for k in depths), flush=True)
            assert passed, f'Consistency failed; inspect samples/{index:05d}.json. Do not relax tolerances blindly.'
            records.append(record)
            del inputs, positions, captured, outputs, baseline, hooked, states
    assert len(records) == SAMPLE_COUNT
    write_reports(records, depths)
    print(f'Completed. Read {OUTPUT / "summary.md"}', flush=True)


if __name__ == '__main__':
    output_existed = OUTPUT.exists()
    try:
        run()
    except Exception as error:
        if not output_existed and OUTPUT.is_dir():
            save_json(OUTPUT / 'FAILED.json', dict(status='FAILED', error=repr(error), traceback=traceback.format_exc()))
        raise
