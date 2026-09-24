"""Frozen SADS-inspired pilot. Each subcommand requires verified prior stages.

No training, proposal generation, early exit, or implicit model downloads.
Use prepare, check, collect/calibration, calibrate, collect/evaluation, select,
intervene, then tools/ref_sads_analysis.py. Nothing runs on import.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import inspect
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

from ref_sads_io import (ROOT, SOURCE_FILES, arm_specs, canonical, digest, load_run,
    marker_path, object_hash, part_read, part_write, read_json, read_jsonl, require,
    require_stage, safe_child, source_digest, split_development, stage_lock,
    utc_now, validate_inputs, write_json, write_jsonl, write_stage)


def resolve_source(path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def verify_checkpoint(checkpoint, signature):
    checkpoint = Path(checkpoint)
    require(checkpoint.is_dir(), f'Missing real checkpoint: {checkpoint}')
    expected = dict(signature['checkpoint_sha256'], **signature['checkpoint_json_sha256'])
    actual_names = {p.name for p in checkpoint.iterdir()
                    if p.suffix in ('.safetensors', '.bin', '.json')}
    require(actual_names == set(expected), 'Checkpoint file set changed')
    for name, sha in expected.items():
        require(digest(checkpoint / name) == sha, f'Checkpoint/tokenizer/processor changed: {name}')


def auxiliary_checkpoint_files(checkpoint):
    """Freeze current tokenizer/template auxiliaries absent from the old manifest."""
    root = Path(checkpoint)
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in ('.jinja', '.txt', '.model', '.tiktoken', '.vocab')}


def validate_config(config):
    # This CLI implements the approved v1, not an unrestricted sweep interface.
    fixed = dict(counts={'calibration': 100, 'evaluation': 200}, layers=[28, 32, 36],
        num_layers=36, num_heads=32, num_kv_heads=8, head_dim=128, hidden_size=2560,
        shared_head=0, gates=[0., .5], random_seeds=[11, 29, 47], candidate_count=100,
        dtype='bfloat16', query_scope='all_nonpadding', visual_groups=['G', 'O'],
        nonvisual_groups=['T', 'S', 'R'], max_attention_aggregation='max_key_of_mean_query',
        entropy='renormalize_rows_then_average_then_entropy',
        entropy_normalization='divide_log_nonvisual_key_count', check_samples=10,
        warmup_passes=6, min_valid_query_fraction=.99, atol=1e-5, rtol=1e-5)
    for key, expected in fixed.items():
        require(config.get(key) == expected, f'Approved v1 requires {key}={expected}')
    require(config['attention_backend'] in ('flash_attention_2', 'eager'), 'Unsupported backend')
    require(config['chunk_size'] > 0 and config['budget_seconds'] > 0, 'Invalid budget/chunk')


def prepare(args):
    started = time.perf_counter()
    config = read_json(args.config)
    if args.backend:
        config['attention_backend'] = args.backend
    validate_config(config)
    run = args.out.resolve()
    results_root = (ROOT / 'results').resolve()
    require(results_root in run.parents and run.name.startswith('ref_sads_pilot_'),
            'Use a new results/ref_sads_pilot_<name> directory')
    require(not run.exists(), f'Refusing to overwrite existing run: {run}')
    paths = {key: resolve_source(config[key]) for key in ('source_plan', 'proposals', 'signature_manifest')}
    for key, path in paths.items():
        require(digest(path) == config[key + '_sha256'], f'Frozen source changed: {key}')
    plan = read_json(paths['source_plan'])
    signature = read_json(paths['signature_manifest'])['model_signature']
    require(plan['counts'] == {'train': 5000, 'dev': 1000, 'validation': 2573}, 'Wrong prior pool')
    require(plan['train_proposals_sha256'] == config['proposals_sha256'], 'Proposal provenance mismatch')
    checkpoint = resolve_source(args.checkpoint or config['checkpoint'])
    images = resolve_source(args.images or config['images'])
    verify_checkpoint(checkpoint, signature)
    # Only relevant inherited code, allowing a documented CRLF/LF-only difference.
    old_sources = signature['sources']
    for name in SOURCE_FILES:
        if Path(name).name in old_sources:
            require(source_digest(ROOT / name) == old_sources[Path(name).name], f'Inherited implementation changed: {name}')
    proposals = read_json(paths['proposals'])
    validation_images = {r['image_key'] for r in plan['rows'] if r['split'] == 'validation'}
    dev = [r for r in plan['rows'] if r['split'] == 'dev']
    require(len(dev) == 1000 and len({r['id'] for r in dev}) == 1000, 'Invalid frozen dev pool')
    selected = split_development(dev, validation_images, config['counts'], config['split_seed'])
    from PIL import Image
    inputs, targets, image_meta = [], [], {}
    for split, row in selected:
        name = row['image_name']
        if name not in image_meta:
            path = safe_child(images, name)
            require(digest(path) == row['image_sha256'], f'Original image changed: {name}')
            with Image.open(path) as image:
                width, height = image.size
            image_meta[name] = width, height
        width, height = image_meta[name]
        source_boxes = proposals[name][0][:config['candidate_count']]
        clipped = [[max(0, min(limit, value)) for value, limit in zip(box, [width, height, width, height])]
                   for box in source_boxes]
        require(clipped == row['candidate_boxes'], f'Frozen candidate order/geometry mismatch: {row["id"]}')
        inputs.append(dict(id=row['id'], image_key=row['image_key'], split=split,
            image_name=name, query=row['referring'], image_sha256=row['image_sha256'],
            candidate_boxes=clipped, candidate_sha256=object_hash(clipped), width=width, height=height))
        # Pure export only. No selection, IoU, loss, or correctness computation here.
        if split == 'evaluation':
            targets.append(dict(id=row['id'], answer_boxes=row['answer_boxes']))
    validate_inputs(inputs, config)
    manifest = dict(schema_version=1, method='SADS-inspired', created_utc=utc_now(),
        config_sha256=object_hash(config), checkpoint=str(checkpoint), images=str(images),
        model_signature=signature, auxiliary_checkpoint_files=auxiliary_checkpoint_files(checkpoint),
        auxiliary_provenance='Frozen at pilot prepare; not a claim of historical SHA equivalence',
        source_inputs={k: dict(path=str(v), sha256=digest(v)) for k, v in paths.items()},
        sources={name: dict(raw_sha256=digest(ROOT / name), lf_sha256=source_digest(ROOT / name)) for name in SOURCE_FILES},
        preparation_seconds=time.perf_counter() - started,
        sample_counts=config['counts'], shared_head=0, full_depth=36, ranking='raw_logit_first_argmax',
        training=False, early_exit=False, gt_used_for_selection=False,
        excluded_validation_images=sorted(validation_images),
        maximum_forwards=5146 if config['attention_backend'] == 'flash_attention_2' else 5392)
    run.mkdir(parents=True)
    write_json(run / 'manifest.json', manifest)
    write_json(run / 'config.json', config)
    write_jsonl(run / 'inputs.jsonl', inputs)
    write_jsonl(run / 'eval_targets.jsonl', targets)
    write_stage(run, 'prepare', ['config.json', 'inputs.jsonl', 'eval_targets.jsonl'],
                metadata={'forward_count': 0, 'image_count': len(image_meta)})
    print(f'Prepared {len(inputs)} fixed expressions, no model forwards: {run}', flush=True)


def tensor_record(tensor):
    import torch
    x = tensor.detach().contiguous().cpu()
    sha = hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()
    return dict(shape=list(x.shape), dtype=str(x.dtype), sha256=sha)


class RefRuntime:
    """Actual full Ref model. Tests inject a separate fake runtime, never via CLI."""
    def __init__(self, loaded, backend=None):
        import torch
        import transformers
        from transformers import AutoProcessor
        sys.path.insert(0, str(ROOT / 'wedetect_ref'))
        from models.qwen3vl_referring import Qwen3VLGroundingForConditionalGeneration
        self.torch, self.loaded, self.config = torch, loaded, loaded['config']
        manifest = loaded['manifest']
        signature = manifest['model_signature']
        self.backend = backend or self.config['attention_backend']
        require(os.environ.get('WORLD_SIZE', '1') == '1', 'Use a single GPU process, not torchrun')
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA with BF16 is required')
        for package, expected in [('torch', signature['torch']), ('transformers', signature['transformers']),
                                  ('torchvision', signature['torchvision']), ('flash-attn', signature['flash_attn'])]:
            require(importlib.metadata.version(package) == expected, f'Pinned environment mismatch: {package}')
        verify_checkpoint(manifest['checkpoint'], signature)
        require(auxiliary_checkpoint_files(manifest['checkpoint']) == manifest['auxiliary_checkpoint_files'],
                'Auxiliary tokenizer/template file set or content changed')
        for row in {r['image_name']: r for r in loaded['inputs']}.values():
            require(digest(safe_child(manifest['images'], row['image_name'])) == row['image_sha256'], 'Image changed since prepare')
        torch.manual_seed(self.config['selection_seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model, info = Qwen3VLGroundingForConditionalGeneration.from_pretrained(
            manifest['checkpoint'], torch_dtype=torch.bfloat16, attn_implementation=self.backend,
            local_files_only=True, output_loading_info=True)
        require(not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), f'Invalid weight load: {info}')
        self.model = self.model.cuda().eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(manifest['checkpoint'], local_files_only=True)
        self.object_id = self.processor.tokenizer.convert_tokens_to_ids('<object>')
        require(self.object_id == self.config['object_token_id'], 'Object token ID changed')
        require(self.processor.tokenizer.encode('<object>', add_special_tokens=False) == [self.object_id], 'Object is not one token')
        self.model.model.object_token_id = self.object_id
        lm = self.model.model.language_model
        for key, value in [('num_hidden_layers', 36), ('num_attention_heads', 32),
                           ('num_key_value_heads', 8), ('head_dim', 128), ('hidden_size', 2560)]:
            require(getattr(lm.config, key) == value, f'Model dimension mismatch: {key}')
        require(len(lm.layers) == 36 and tuple(self.model.out_proj.weight.shape) == (1, 2560), 'Original depth/classifier changed')
        require(self.model.config.image_token_id == self.config['image_token_id'], 'Image token ID changed')
        require(lm.config.use_cache == signature['checkpoint_use_cache'], 'Cache setting changed')
        require(lm.config._attn_implementation == self.backend and
                self.model.model.visual.config._attn_implementation == self.backend, 'Unexpected text/vision backend')
        source_path = Path(inspect.getfile(type(lm)))
        require(source_digest(source_path) == signature['transformers_source'], 'Installed Qwen3-VL implementation changed')
        prop = torch.cuda.get_device_properties(0)
        self.environment = dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__,
            numpy=importlib.metadata.version('numpy'), pillow=importlib.metadata.version('pillow'),
            backend=self.backend, text_backend=lm.config._attn_implementation,
            vision_backend=self.model.model.visual.config._attn_implementation,
            gpu=prop.name, gpu_total_memory_bytes=prop.total_memory, cuda=torch.version.cuda,
            device_capability=list(torch.cuda.get_device_capability(0)),
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), hf_source_sha256=digest(source_path),
            tf32=False, cudnn_benchmark=False, cudnn_deterministic=True)

    def prepare_input(self, row):
        from PIL import Image
        from ref_e0 import prepare_input
        from ref_sads_core import partition_tokens
        start = time.perf_counter()
        with Image.open(safe_child(self.loaded['manifest']['images'], row['image_name'])) as src:
            image = src.convert('RGB')
        require(image.size == (row['width'], row['height']), 'Image dimensions changed')
        inputs, positions, prompt = prepare_input(self.model, self.processor, image,
            row['query'], row['candidate_boxes'], self.object_id)
        self.torch.cuda.synchronize()
        prep_ms = (time.perf_counter() - start) * 1000
        audit_start = time.perf_counter()
        partition = partition_tokens(inputs['input_ids'], inputs['attention_mask'], self.processor.tokenizer,
                                     self.config['image_token_id'], self.object_id)
        require(partition['counts']['O'] == self.config['candidate_count'], 'Object count changed')
        tensors = {key: tensor_record(value) for key, value in inputs.items() if isinstance(value, self.torch.Tensor)}
        tensors['bboxes'] = [tensor_record(v) for v in inputs['bboxes']]
        # Read-only expectation; do not inject position_ids into the real model.
        with self.torch.no_grad():
            position_ids, _ = self.model.model.get_rope_index(inputs['input_ids'],
                inputs.get('image_grid_thw'), inputs.get('video_grid_thw'), attention_mask=inputs['attention_mask'])
        expected_positions = dict(shape=list(position_ids.shape), dtype=str(position_ids.dtype),
                                  values=position_ids.detach().cpu().tolist())
        audit = dict(tensors=tensors, input_ids=inputs['input_ids'].cpu().tolist(),
            attention_mask=inputs['attention_mask'].cpu().tolist(), ori_shapes=inputs['ori_shapes'],
            expected_position_ids=expected_positions,
            prompt=prompt, partition=partition, sequence_length=int(inputs['input_ids'].shape[1]))
        input_sha256 = object_hash(audit)
        return dict(inputs=inputs, positions=positions, partition=partition, audit=audit,
                    input_sha256=input_sha256, preprocessing_ms=prep_ms,
                    input_audit_ms=(time.perf_counter() - audit_start) * 1000)

    def forward(self, row, mode, physical_id, phase, gate=None, expected_input=None):
        from ref_e0_core import object_logits
        from ref_sads_core import HeadIntervention
        torch = self.torch
        request_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        prepared = self.prepare_input(row)
        if expected_input is not None:
            require(prepared['input_sha256'] == expected_input, f'Input tensors changed: {row["id"]}')
        collect = mode in ('stats', 'reference_stats')
        gates = {} if gate is None else {gate['layer']: {gate['head']: gate['gate']}}
        context = nullcontext(None) if mode == 'plain' else HeadIntervention(self.model, self.config['layers'],
            prepared['partition'], gates=gates, collect=collect, chunk_size=self.config['chunk_size'],
            reference=(mode == 'reference_stats'))
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        forward_start = time.perf_counter()
        with torch.inference_mode(), context as hooks:
            begin.record()
            output = self.model(**prepared['inputs'])
            end.record()
            torch.cuda.synchronize()
            values = object_logits(output.logits, prepared['positions']).float().cpu().tolist()
        forward_wall_ms = (time.perf_counter() - forward_start) * 1000
        if hooks is not None:
            actual = hooks.diagnostics['position_ids']
            require(actual is not None and all(actual[k] == v for k, v in
                    prepared['audit']['expected_position_ids'].items()), 'Actual model position IDs changed')
        require(len(values) == self.config['candidate_count'] and all(math.isfinite(v) for v in values), 'Invalid model logits')
        resource = dict(physical_forward_id=physical_id, id=row['id'], phase=phase, collector=collect,
            backend=self.backend, forward_ms=begin.elapsed_time(end), forward_wall_ms=forward_wall_ms,
            request_ms=(time.perf_counter() - request_start) * 1000,
            preprocessing_ms=prepared['preprocessing_ms'],
            input_audit_ms=prepared['input_audit_ms'],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu=self.environment['gpu'], gpu_total_memory_bytes=self.environment['gpu_total_memory_bytes'])
        result = dict(id=row['id'], logits=values, input_sha256=prepared['input_sha256'],
            input_audit=prepared['audit'], resource=resource, backend=self.backend,
            operation=None if gate is None else {k: gate[k] for k in ('layer', 'head', 'gate')},
            statistics=hooks.statistics if hooks is not None else {},
            diagnostics=hooks.diagnostics if hooks is not None else {})
        del output, prepared, context, hooks
        return result

    def close(self):
        del self.model, self.processor
        import gc
        gc.collect()
        self.torch.cuda.empty_cache()


def runtime_environment(run, runtime, stage, reference=False):
    """All production phases must use the same software, device and backend."""
    path = Path(run) / ('reference_environment.json' if reference else 'environment.json')
    if path.exists():
        require(read_json(path) == runtime.environment, f'Runtime environment changed at {stage}; start a new run')
    else:
        write_json(path, runtime.environment)


def compare_identity(results, config):
    import numpy as np
    arrays = [np.asarray(r['logits'], dtype=np.float32) for r in results]
    require(len({r['input_sha256'] for r in results}) == 1, 'Identity input differs')
    a = arrays[0]
    comparisons = []
    for name, b in zip(('A2', 'B', 'C'), arrays[1:]):
        delta = np.abs(b - a)
        passed = bool(np.all(delta <= config['atol'] + config['rtol'] * np.abs(a)) and np.argmax(a) == np.argmax(b))
        comparisons.append(dict(path=name, passed=passed, exact=bool(np.array_equal(a, b)),
            max_abs=float(delta.max()), mean_abs=float(delta.mean()),
            baseline_top1=int(np.argmax(a)), actual_top1=int(np.argmax(b))))
    error = max(float(np.abs(a - b).max()) for a in arrays for b in arrays)
    magnitude = max(float(np.abs(a).max()) for a in arrays)
    return dict(passed=all(c['passed'] for c in comparisons), comparisons=comparisons,
                max_pairwise_error=error, max_logit_magnitude=magnitude)


def _publish(run, name, value, jsonl=False):
    """On explicit resume, accept only identical final files left before a crash."""
    path = Path(run) / name
    if path.exists():
        require((read_jsonl(path) if jsonl else read_json(path)) == value, f'Partial final artifact changed: {name}')
    else:
        (write_jsonl if jsonl else write_json)(path, value)


def check(args, runtime_class=RefRuntime):
    loaded = load_run(args.run)
    config = loaded['config']
    require(args.n == config['check_samples'], 'The approved identity check requires 10 expressions')
    manifest_sha = digest(args.run / 'manifest.json')
    runtime = None
    try:
        with stage_lock(args.run, 'check', args.resume) as partial:
            runtime = runtime_class(loaded)
            runtime_environment(args.run, runtime, 'check')
            cal = [r for r in loaded['inputs'] if r['split'] == 'calibration']
            selection_path = partial / 'selection.json'
            if selection_path.exists():
                selection = part_read(selection_path, manifest_sha)
            else:
                # No forward: actual tokenized lengths, before any task result exists.
                lengths = []
                for row in cal:
                    prepared = runtime.prepare_input(row)
                    lengths.append((prepared['audit']['sequence_length'], row['id']))
                    del prepared
                lengths.sort()
                indices = [round(i * (len(lengths) - 1) / (args.n - 1)) for i in range(args.n)]
                selection = dict(ids=[lengths[i][1] for i in indices], lengths=lengths)
                part_write(selection_path, selection, manifest_sha)
            rows = {r['id']: r for r in cal}
            resources, checks = [], []
            for i in range(config['warmup_passes']):
                p = partial / f'warmup_{i:02}.json'
                if p.exists():
                    value = part_read(p, manifest_sha)
                else:
                    value = runtime.forward(rows[selection['ids'][i % args.n]], 'plain', f'check:warmup:{i}', 'warmup')
                    part_write(p, value, manifest_sha)
                resources.append(value['resource'])
            for i, sid in enumerate(selection['ids']):
                results = []
                for name, mode in [('A', 'plain'), ('A2', 'plain'), ('B', 'gate'), ('C', 'reference_stats')]:
                    p = partial / f'{i:03}_{name}.json'
                    if p.exists():
                        value = part_read(p, manifest_sha)
                    else:
                        value = runtime.forward(rows[sid], mode, f'check:{i}:{name}', f'check_{name}',
                            expected_input=results[0]['input_sha256'] if results else None)
                        part_write(p, value, manifest_sha)
                    require(value['id'] == sid, 'Wrong identity partial')
                    results.append(value)
                    resources.append(value['resource'])
                comparison = compare_identity(results, config)
                comparison.update(id=sid, input_sha256=results[0]['input_sha256'])
                checks.append(comparison)
                require(comparison['passed'], f'Gate=1 does not reproduce raw logits: {sid}')
                print(f'Identity {i+1}/{len(selection["ids"])} PASS: {sid}', flush=True)
            error = max(r['max_pairwise_error'] for r in checks)
            magnitude = max(r['max_logit_magnitude'] for r in checks)
            eps = max(1e-6, 5 * error * .75 * (1 + .5 * (magnitude + math.log(2))))
            # Collector C is counted separately; never hide it in plain latency.
            plain = [r['request_ms'] for r in resources if r['phase'] == 'check_B']
            stats = [r['request_ms'] for r in resources if r['phase'] == 'check_C']
            estimate = (4836 * statistics.mean(plain) + 310 * statistics.mean(stats)) / 1000 + config['setup_allowance_seconds']
            if config['attention_backend'] != 'flash_attention_2':
                estimate += 246 * statistics.mean(plain) / 1000
            report = dict(passed=True, eps=eps, max_pairwise_error=error, max_logit_magnitude=magnitude,
                checks=checks, selected_ids=selection['ids'], budget_estimate_seconds=estimate,
                budget_limit_seconds=config['budget_seconds'], budget_passed=estimate <= config['budget_seconds'],
                same_backend=True, backend=config['attention_backend'])
            _publish(args.run, 'checks/consistency.json', report)
            _publish(args.run, 'checks/resources.jsonl', resources, True)
            require(report['budget_passed'], f'Estimated runtime {estimate:.0f}s exceeds fixed budget; do not change samples or precision')
            files = ['checks/consistency.json', 'checks/resources.jsonl', 'environment.json']
            files += [str(p.relative_to(args.run)).replace('\\', '/') for p in sorted(partial.glob('*.json'))]
            write_stage(args.run, 'check', files, dict(forward_count=len(resources)), ['prepare'])
    finally:
        if runtime:
            runtime.close()


def prediction(row, result, spec=None):
    spec = spec or dict(arm_id='baseline', kind='baseline', layer=None, gate=1., seed=None, head=None, k=0)
    values = result['logits']
    winner = max(range(len(values)), key=values.__getitem__)
    return dict(spec, id=row['id'], image_key=row['image_key'], logits=values,
        prediction_index=winner, prediction_box=row['candidate_boxes'][winner],
        input_sha256=result['input_sha256'], backend=result['backend'],
        physical_forward_id=result['resource']['physical_forward_id'])


def collect(args, runtime_class=RefRuntime):
    loaded = load_run(args.run)
    require_stage(args.run, 'check')
    if args.split == 'evaluation':
        require_stage(args.run, 'calibrate')
    stage = 'collect_' + args.split
    manifest_sha = digest(args.run / 'manifest.json')
    runtime = None
    try:
        with stage_lock(args.run, stage, args.resume) as partial:
            runtime = runtime_class(loaded)
            runtime_environment(args.run, runtime, stage)
            rows = [r for r in loaded['inputs'] if r['split'] == args.split]
            stats, predictions, resources = [], [], []
            for i, row in enumerate(rows):
                path = partial / f'{i:04}.json'
                if path.exists():
                    result = part_read(path, manifest_sha)
                else:
                    result = runtime.forward(row, 'stats', f'{stage}:{i}', stage)
                    part_write(path, result, manifest_sha)
                require(result['id'] == row['id'], 'Wrong collected sample')
                for layer in loaded['config']['layers']:
                    for head in result['statistics'].get(str(layer), result['statistics'].get(layer, [])):
                        stats.append(dict(head, id=row['id'], image_key=row['image_key'], split=args.split,
                                          layer=layer, stats_source='baseline'))
                predictions.append(prediction(row, result))
                resources.append(result['resource'])
                print(f'{stage} {i+1}/{len(rows)} {row["id"]}', flush=True)
            require(len(stats) == len(rows) * len(loaded['config']['layers']) * loaded['config']['num_heads'], 'Incomplete head statistics')
            names = [f'stats/{args.split}_heads.jsonl', f'baseline/{args.split}.jsonl', f'resources/{stage}.jsonl']
            for name, values in zip(names, (stats, predictions, resources)):
                _publish(args.run, name, values, True)
            names += [str(p.relative_to(args.run)).replace('\\', '/') for p in sorted(partial.glob('*.json'))]
            parents = ['check'] + (['calibrate'] if args.split == 'evaluation' else [])
            write_stage(args.run, stage, names, dict(forward_count=len(rows)), parents)
    finally:
        if runtime:
            runtime.close()


def calibrate(args):
    from ref_sads_stats import calibrate_heads
    loaded = load_run(args.run)
    require_stage(args.run, 'collect_calibration')
    with stage_lock(args.run, 'calibrate', args.resume):
        start = time.perf_counter()
        rows = read_jsonl(args.run / 'stats/calibration_heads.jsonl')
        result = calibrate_heads(rows, loaded['config'])
        _publish(args.run, 'calibration.json', result)
        write_stage(args.run, 'calibrate', ['calibration.json'],
                    dict(wall_seconds=time.perf_counter() - start, forward_count=0), ['collect_calibration'])


def select(args):
    from ref_sads_stats import select_heads
    loaded = load_run(args.run)
    require_stage(args.run, 'collect_evaluation')
    with stage_lock(args.run, 'select', args.resume):
        start = time.perf_counter()
        selections = select_heads(read_jsonl(args.run / 'stats/evaluation_heads.jsonl'),
                                  read_json(args.run / 'calibration.json'), loaded['config'])
        _publish(args.run, 'selections.jsonl', selections, True)
        write_stage(args.run, 'select', ['selections.jsonl'],
                    dict(wall_seconds=time.perf_counter() - start, forward_count=0), ['collect_evaluation', 'calibrate'])


def validate_resources(resources):
    require(len({r['physical_forward_id'] for r in resources}) == len(resources), 'Duplicate physical resource entry')
    for row in resources:
        for key in ('forward_ms', 'request_ms', 'peak_allocated_bytes', 'peak_reserved_bytes'):
            require(key in row and isinstance(row[key], (int, float)) and math.isfinite(row[key]) and row[key] >= 0,
                    f'Missing/invalid actual resource measurement: {key}')


def intervene(args, runtime_class=RefRuntime):
    loaded = load_run(args.run)
    require_stage(args.run, 'select')
    config = loaded['config']
    selections = read_jsonl(args.run / 'selections.jsonl')
    baseline = {r['id']: r for r in read_jsonl(args.run / 'baseline/evaluation.jsonl')}
    by_id = {}
    for choice in selections:
        by_id.setdefault(choice['id'], []).append(choice)
    rows = [r for r in loaded['inputs'] if r['split'] == 'evaluation']
    require(set(by_id) == set(baseline) == {r['id'] for r in rows}, 'Selection/evaluation mismatch')
    manifest_sha = digest(args.run / 'manifest.json')
    # No model load at all when every intervention is an honest no-op.
    runtime = None
    try:
        with stage_lock(args.run, 'intervene', args.resume) as partial:
            if any(s['k'] for s in selections):
                runtime = runtime_class(loaded)
                runtime_environment(args.run, runtime, 'intervene')
            predictions, actual_resources = [], []
            for i, row in enumerate(rows):
                base = baseline[row['id']]
                predictions.append(base)
                specs = [spec for choice in by_id[row['id']] for spec in arm_specs(choice, config)]
                random.Random(config['order_seed'] + i).shuffle(specs)
                completed = {}
                for spec in specs:
                    if spec['k'] == 0:
                        predictions.append(dict(base, **spec, no_op=True, duplicate=True))
                        continue
                    key = (spec['layer'], spec['head'], spec['gate'])
                    duplicate = key in completed
                    if not duplicate:
                        path = partial / f'{i:04}_L{key[0]}_h{key[1]}_g{key[2]:g}.json'
                        if path.exists():
                            result = part_read(path, manifest_sha)
                        else:
                            require(runtime is not None, 'Missing runtime for a nonempty intervention')
                            result = runtime.forward(row, 'gate', f'intervene:{i}:L{key[0]}:h{key[1]}:g{key[2]:g}',
                                                     'intervene', gate=spec, expected_input=base['input_sha256'])
                            part_write(path, result, manifest_sha)
                        require(result['id'] == row['id'] and result['input_sha256'] == base['input_sha256'], 'Mismatched intervention partial')
                        require(result['operation'] == {k: spec[k] for k in ('layer', 'head', 'gate')}, 'Partial intervention is a different operation')
                        completed[key] = result
                        actual_resources.append(result['resource'])
                    predictions.append(dict(prediction(row, completed[key], spec), no_op=False, duplicate=duplicate))
                print(f'Interventions {i+1}/{len(rows)} ({len(completed)} actual forwards)', flush=True)
            # If backend changed, a separately loaded original FA2 baseline is mandatory.
            if runtime:
                runtime.close()
                runtime = None
            if config['attention_backend'] != 'flash_attention_2':
                runtime = runtime_class(loaded, backend='flash_attention_2')
                runtime_environment(args.run, runtime, 'reference', reference=True)
                for i, row in enumerate(rows):
                    path = partial / f'{i:04}_reference_fa2.json'
                    if path.exists():
                        result = part_read(path, manifest_sha)
                    else:
                        result = runtime.forward(row, 'plain', f'reference:FA2:{i}', 'reference',
                                                 expected_input=baseline[row['id']]['input_sha256'])
                        part_write(path, result, manifest_sha)
                    predictions.append(prediction(row, result, dict(arm_id='reference_fa2', kind='reference',
                        layer=None, gate=1., seed=None, head=None, k=0)))
                    actual_resources.append(result['resource'])
            resources = read_jsonl(args.run / 'checks/resources.jsonl')
            for phase in ('collect_calibration', 'collect_evaluation'):
                resources.extend(read_jsonl(args.run / f'resources/{phase}.jsonl'))
            resources.extend(actual_resources)
            validate_resources(resources)
            expected = len(rows) * (25 + (config['attention_backend'] != 'flash_attention_2'))
            require(len(predictions) == expected, 'Logical arm matrix incomplete')
            require(len(resources) <= loaded['manifest']['maximum_forwards'], 'Physical forward budget exceeded')
            _publish(args.run, 'predictions.jsonl', predictions, True)
            _publish(args.run, 'resources.jsonl', resources, True)
            names = ['predictions.jsonl', 'resources.jsonl']
            if config['attention_backend'] != 'flash_attention_2':
                names.append('reference_environment.json')
            names += [str(p.relative_to(args.run)).replace('\\', '/') for p in sorted(partial.glob('*.json'))]
            write_stage(args.run, 'intervene', names,
                dict(logical_predictions=len(predictions), physical_forwards=len(resources)), ['select'])
    finally:
        if runtime:
            runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Verify and freeze inputs; no model forward')
    p.add_argument('--config', type=Path, default=ROOT / 'config/ref_sads_pilot_v1.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', help='Relocate the SAME hashed checkpoint')
    p.add_argument('--images', help='Relocate the SAME hashed original images')
    p.add_argument('--backend', choices=['flash_attention_2', 'eager'], help='Explicit pre-calibration backend; default FA2')
    for name in ('check', 'collect', 'calibrate', 'select', 'intervene'):
        p = sub.add_parser(name)
        p.add_argument('--run', type=Path, required=True)
        p.add_argument('--resume', action='store_true', help='Resume verified partial records with unchanged provenance')
        if name == 'check':
            p.add_argument('--n', type=int, default=10)
        if name == 'collect':
            p.add_argument('--split', required=True, choices=['calibration', 'evaluation'])
    args = parser.parse_args()
    if hasattr(args, 'run'):
        args.run = args.run.resolve()
    start = time.perf_counter()
    succeeded = False
    try:
        globals()[args.command](args)
        succeeded = True
    except Exception as exc:
        run = getattr(args, 'run', None)
        if run is not None and run.is_dir():
            failure = run / 'failures' / f'{args.command}_{time.time_ns()}.json'
            write_json(failure, dict(stage=args.command, error=type(exc).__name__, message=str(exc),
                traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - start, created_utc=utc_now()))
        raise
    finally:
        run = getattr(args, 'run', getattr(args, 'out', None))
        if run is not None and (run / 'manifest.json').is_file():
            write_json(run / 'timings' / f'{args.command}_{time.time_ns()}.json',
                       dict(command=args.command, split=getattr(args, 'split', None), succeeded=succeeded,
                            wall_seconds=time.perf_counter() - start, created_utc=utc_now()))
        print(f'{args.command} wall time: {time.perf_counter() - start:.2f}s', flush=True)


if __name__ == '__main__':
    main()
