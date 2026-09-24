"""Post-hoc, image-paired analysis for the frozen SADS-inspired pilot.

This is the only pilot stage that consumes evaluation ground truth. No model is
loaded. Logical duplicate/no-op arms retain their sample denominator, while
physical forward resources are counted once. NumPy is the only dependency.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _number(value):
    return float(value) if _finite(value) else None


def _gate_key(value):
    return format(float(value), '.6g')


def clipped_boxes(boxes, width, height):
    """Match the original independent xyxy clipping, including degenerate boxes."""
    out = np.asarray(boxes, dtype=np.float64)
    _require(out.ndim == 2 and out.shape[1] == 4 and len(out) > 0,
             'Expected a nonempty xyxy box array')
    _require(np.isfinite(out).all() and width > 0 and height > 0, 'Invalid geometry')
    out = out.copy()
    out[:, (0, 2)] = np.clip(out[:, (0, 2)], 0, width)
    out[:, (1, 3)] = np.clip(out[:, (1, 3)], 0, height)
    return out


def candidate_ious(boxes, answer_boxes, width, height):
    """RefCOCOg has exactly one target; preserve the original single-target task."""
    boxes = clipped_boxes(boxes, width, height)
    gt = clipped_boxes(answer_boxes, width, height)
    _require(len(gt) == 1, 'This preregistered RefCOCOg protocol requires one GT box')
    overlap = np.maximum(0, np.minimum(boxes[:, 2:], gt[0, 2:]) -
                         np.maximum(boxes[:, :2], gt[0, :2])).prod(axis=1)
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    gt_area = np.maximum(0, gt[0, 2:] - gt[0, :2]).prod()
    union = area + gt_area - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)


def focal_loss_fp32(logits, targets):
    """Original alpha=.25, gamma=2 sigmoid focal, FP32 candidate mean.

    The BCE form is stable even for saturated BF16 logits converted to FP32.
    Labels are soft IoUs; their construction is deliberately outside this helper.
    """
    x, y = np.asarray(logits, np.float32), np.asarray(targets, np.float32)
    _require(x.ndim == 1 and x.shape == y.shape and x.size > 0,
             'Loss requires matching nonempty candidate vectors')
    _require(np.isfinite(x).all() and np.isfinite(y).all() and
             ((y >= 0) & (y <= 1)).all(), 'Invalid focal input')
    p = np.empty_like(x)
    positive = x >= 0
    p[positive] = np.float32(1) / (np.float32(1) + np.exp(-x[positive]))
    ex = np.exp(x[~positive])
    p[~positive] = ex / (np.float32(1) + ex)
    ce = np.maximum(x, np.float32(0)) - x * y + np.log1p(np.exp(-np.abs(x)))
    pt = p * y + (np.float32(1) - p) * (np.float32(1) - y)
    alpha = np.float32(.25) * y + np.float32(.75) * (np.float32(1) - y)
    loss = (alpha * ce * (np.float32(1) - pt) ** 2).mean(dtype=np.float32)
    _require(np.isfinite(loss), 'Nonfinite focal loss')
    return float(loss)


def raw_top1(logits):
    x = np.asarray(logits, np.float32)
    _require(x.ndim == 1 and x.size > 0 and np.isfinite(x).all(), 'Invalid raw logits')
    return int(np.argmax(x))  # first candidate wins exact ties


def image_bootstrap_weights(image_keys, repeats, seed):
    """Every sampled image contributes ALL its expressions, layers and seeds.

    Columns are expressions, not unique images; later head-level metrics index
    these same columns instead of independently resampling heads or random seeds.
    """
    _require(repeats >= 1, 'bootstrap_repeats must be positive')
    groups = sorted(set(image_keys), key=lambda value: (str(type(value)), str(value)))
    _require(bool(groups), 'No evaluation images')
    lookup = {key: i for i, key in enumerate(groups)}
    inverse = np.asarray([lookup[key] for key in image_keys], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(repeats, len(groups)))
    counts = np.zeros((repeats, len(groups)), dtype=np.int64)
    for index, draw in enumerate(draws):
        counts[index] = np.bincount(draw, minlength=len(groups))
    return counts[:, inverse]


def _weighted_mean(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(values) & (weights > 0)
    total = weights[valid].sum()
    return float(np.dot(values[valid], weights[valid]) / total) if total else math.nan


def _bootstrap_means(values, weights):
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    denominator = weights[:, valid].sum(axis=1)
    numerator = weights[:, valid] @ values[valid]
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan),
                     where=denominator > 0)


def _interval(point, replicates, n_images, minimum_fraction=.8):
    """Conservative missing-bootstrap policy; never convert NA to a null effect."""
    values = np.asarray(replicates, float)
    valid = values[np.isfinite(values)]
    result = dict(value=_number(point), ci95=None, valid_bootstrap=int(len(valid)),
                  bootstrap_repeats=int(len(values)), n_images=int(n_images))
    if not _finite(point):
        result['reason'] = 'Point estimate is not identifiable'
    elif n_images < 2:
        result['reason'] = 'Fewer than two independent images'
    elif len(valid) < max(2, math.ceil(minimum_fraction * len(values))):
        result['reason'] = 'Too few estimable image-bootstrap replicates'
    else:
        result['ci95'] = [float(v) for v in np.quantile(valid, [.025, .975])]
        result['reason'] = None
    return result


def _summary_value(values, weights, images):
    values = np.asarray(values, float)
    keep = np.isfinite(values)
    result = _interval(_weighted_mean(values, np.ones(len(values))),
                       _bootstrap_means(values, weights),
                       len({images[i] for i in np.flatnonzero(keep)}))
    result['n_expressions'] = int(keep.sum())
    return result


def _ranks(values, weights):
    """Ranks of the expanded observations, without materializing repeated images."""
    order = np.argsort(values, kind='stable')
    sorted_values, sorted_weights = values[order], weights[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    counts = np.add.reduceat(sorted_weights, starts)
    rank_values = np.cumsum(counts) - counts + (counts + 1) / 2
    group_sizes = np.diff(np.r_[starts, len(values)])
    out = np.empty(len(values), dtype=float)
    out[order] = np.repeat(rank_values, group_sizes)
    return out


def weighted_spearman(scores, benefit, weights=None):
    x, y = np.asarray(scores, float), np.asarray(benefit, float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    keep = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[keep], y[keep], w[keep]
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan
    a, b = _ranks(x, w), _ranks(y, w)
    a -= _weighted_mean(a, w)
    b -= _weighted_mean(b, w)
    denom = math.sqrt(float(np.dot(w, a * a) * np.dot(w, b * b)))
    return float(np.dot(w, a * b) / denom) if denom else math.nan


def weighted_auc(scores, harmful, weights=None):
    """Tie-correct AUROC for B > eps; constant scores are explicitly NA."""
    x, y = np.asarray(scores, float), np.asarray(harmful, bool)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    keep = np.isfinite(x) & (w > 0)
    x, y, w = x[keep], y[keep], w[keep]
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan
    order = np.argsort(x, kind='stable')
    x, y, w = x[order], y[order], w[order]
    starts = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1]
    pos = np.add.reduceat(w * y, starts)
    neg = np.add.reduceat(w * ~y, starts)
    concordance = np.dot(pos, np.cumsum(neg) - neg + .5 * neg)
    return float(concordance / (pos.sum() * neg.sum()))


def _prediction_metric(records, sample_indices, weights, images, eps):
    """Return serializable metrics plus paired replicates for macro inference."""
    valid = [r for r in records if r.get('stat_valid') and _finite(r.get('score'))]
    indices = np.asarray([sample_indices[r['id']] for r in valid], dtype=int)
    scores = np.asarray([r['score'] for r in valid], float)
    benefit = np.asarray([r['benefit'] for r in valid], float)
    harmful = benefit > eps
    n_images = len({images[i] for i in indices})
    boot_weights = weights[:, indices]
    point_auc = weighted_auc(scores, harmful)
    point_rho = weighted_spearman(scores, benefit) if np.unique(harmful).size == 2 else math.nan
    auc_boot = np.asarray([weighted_auc(scores, harmful, w) for w in boot_weights])
    rho_boot = np.asarray([weighted_spearman(scores, benefit, w)
                           if np.unique(harmful[w > 0]).size == 2 else math.nan
                           for w in boot_weights])
    result = dict(n_observations=len(valid), n_expressions=len(set(indices.tolist())),
                  n_images=n_images, dropped_invalid=len(records) - len(valid),
                  harmful=int(harmful.sum()), non_harmful=int((~harmful).sum()),
                  single_effect_class_policy='Primary AUROC and Spearman are NA when B>eps has one class',
                  auroc=_interval(point_auc, auc_boot, n_images),
                  spearman=_interval(point_rho, rho_boot, n_images), harmful_rates={})
    for category in ('sinkS', 'non_sinkS'):
        chosen = np.asarray([r['category'] == 'sinkS' for r in valid], dtype=bool)
        if category == 'non_sinkS':
            chosen = ~chosen
        vals = np.where(chosen, harmful.astype(float), np.nan)
        rate = _interval(_weighted_mean(vals, np.ones(len(vals))),
                         _bootstrap_means(vals, boot_weights),
                         len({images[i] for i in indices[chosen]}))
        rate['n_observations'] = int(chosen.sum())
        result['harmful_rates'][category] = rate
    return result, dict(auroc=auc_boot, spearman=rho_boot)


def _effect_summary(delta_sink, delta_random, eligible, weights, images, eps):
    ds, dr = np.asarray(delta_sink), np.asarray(delta_random)
    c = (-ds > eps).astype(float) - (-dr > eps).mean(axis=-1)
    d = ds - dr.mean(axis=-1)
    eligible = np.asarray(eligible, bool)
    # A no-op is an observed zero effect, never a missing case in ITT.
    _require(np.all(c[~eligible] == 0) and np.all(d[~eligible] == 0),
             'A no-op arm changed an outcome')
    result = {}
    for label, mask in (('eligible', eligible), ('itt', np.ones_like(eligible))):
        if ds.ndim == 2:
            denominator = mask.sum(axis=1)
            cv = np.divide((c * mask).sum(axis=1), denominator,
                           out=np.full(len(c), np.nan), where=denominator > 0)
            dv = np.divide((d * mask).sum(axis=1), denominator,
                           out=np.full(len(d), np.nan), where=denominator > 0)
        else:
            cv, dv = np.where(mask, c, np.nan), np.where(mask, d, np.nan)
        result[label] = dict(C=_summary_value(cv, weights, images),
                             D=_summary_value(dv, weights, images),
                             n_sample_layers=int(mask.sum()))
    return result


def _resource_summary(resources, predictions):
    """Trust only a unique physical-forward ledger, never duplicated arm timings."""
    forwards, events = {}, []
    for record in resources:
        pid = record.get('physical_forward_id')
        if pid is None:
            events.append(record)
            continue
        _require(pid not in forwards, f'Duplicate physical resource record: {pid}')
        _require(type(record.get('collector')) is bool, 'Missing/invalid collector resource scope')
        _require(isinstance(record.get('phase'), str) and bool(record['phase']), 'Missing physical resource phase')
        for field in ('forward_ms', 'request_ms', 'peak_allocated_bytes', 'peak_reserved_bytes'):
            value = record.get(field)
            _require(isinstance(value, (int, float, np.number)) and
                     not isinstance(value, (bool, np.bool_)) and _finite(value) and value >= 0,
                     f'Missing/invalid actual resource measurement: {field}')
        forwards[pid] = record
    used = {r['physical_forward_id'] for r in predictions}
    _require(used <= set(forwards), 'Predictions reference missing physical resource records')
    for prediction in predictions:
        physical = forwards[prediction['physical_forward_id']]
        if 'backend' in prediction:
            _require(prediction['backend'] == physical.get('backend'), 'Prediction/resource backend mismatch')
    groups = defaultdict(list)
    for r in forwards.values():
        groups['collector' if r.get('collector', False) else 'plain_or_gate'].append(r)
    def summarize(rows):
        summary = dict(physical_forwards=len(rows))
        for field in ('forward_ms', 'request_ms', 'peak_allocated_bytes', 'peak_reserved_bytes'):
            values = [float(r[field]) for r in rows if _finite(r.get(field))]
            _require(all(v >= 0 for v in values), f'Negative resource metric: {field}')
            summary[field] = (dict(count=len(values), missing=len(rows)-len(values),
                                   mean=float(np.mean(values)), p50=float(np.median(values)),
                                   p95=float(np.quantile(values, .95)), maximum=max(values),
                                   total=float(np.sum(values)) if field.endswith('_ms') else None)
                              if values else dict(count=0, missing=len(rows), mean=None,
                                                  p50=None, p95=None, maximum=None, total=None))
        return summary
    stages = defaultdict(list)
    for r in forwards.values():
        stages[str(r.get('phase', 'unspecified'))].append(r)
    command_events = [r for r in events if str(r.get('phase', '')).startswith('command_')]
    for event in command_events:
        _require(_finite(event.get('wall_seconds')) and event['wall_seconds'] >= 0,
                 'Missing/invalid command wall time')
        _require(type(event.get('succeeded')) is bool, 'Missing/invalid command success flag')
    return dict(logical_prediction_rows=len(predictions),
                unique_prediction_forwards=len(used),
                referenced_reuses=len(predictions)-len(used),
                total_physical_forwards=len(forwards),
                additional_check_warmup_calibration_forwards=len(set(forwards)-used),
                scopes={name: summarize(rows) for name, rows in groups.items()},
                stages={name: summarize(rows) for name, rows in stages.items()},
                non_forward_events=events,
                job_stage_wall_seconds_sum=sum(r['wall_seconds'] for r in command_events) if command_events else None,
                recorded_command_invocations=len(command_events),
                successful_command_invocations=sum(r['succeeded'] for r in command_events),
                failed_command_invocations=sum(not r['succeeded'] for r in command_events),
                report_compute_wall_seconds=None,
                wall_time_note='Sum of recorded command wall times, including failed attempts and model loading/hashing. '
                               'Excludes waiting between commands; do not add forward/request times to this sum. '
                               'Report compute is measured separately before publishing outputs.',
                note='Collector timings are not plain-baseline timings. Head zeroing does not imply acceleration.')


def _validate_and_evaluate(inputs, predictions, selections, targets, config):
    # Pure stdlib validation shared with the runner: exact label-free fields,
    # candidate geometry/order hash and image grouping. It performs no file I/O.
    from ref_sads_io import validate_inputs
    validate_inputs(inputs, config)
    eval_rows = [r for r in inputs if r['split'] == 'evaluation']
    _require(bool(eval_rows), 'No evaluation inputs')
    _require(len({r['id'] for r in inputs}) == len(inputs), 'Duplicate input IDs')
    _require(not any('answer_boxes' in r or 'gt' in r for r in inputs), 'GT found in inputs')
    by_id = {r['id']: r for r in eval_rows}
    target_map = {r['id']: r for r in targets}
    _require(all(set(row) == {'id', 'answer_boxes'} for row in targets), 'Unexpected evaluation target fields')
    _require(len(target_map) == len(targets) and set(target_map) == set(by_id),
             'Evaluation GT IDs must match exactly; calibration GT is forbidden')
    layers, seeds, gates = list(config['layers']), list(config['random_seeds']), list(config['gates'])
    _require(bool(layers) and bool(seeds), 'Empty preregistered layer/seed lists')
    _require(len(set(layers)) == len(layers) and len(set(seeds)) == len(seeds), 'Duplicate layer/seed')
    _require(set(gates) == {0, .5}, 'Expected preregistered hard and soft gates')
    _require(all(1 <= layer <= 36 for layer in layers), 'Layer IDs must be 1-based')
    expected = {('baseline', None, 1., None)}
    expected |= {('sinkS', layer, float(gate), None) for layer in layers for gate in gates}
    expected |= {('random', layer, float(gate), seed) for layer in layers for gate in gates for seed in seeds}
    selection_map = {(r['id'], r['layer']): r for r in selections}
    _require(len(selection_map) == len(selections) and
             set(selection_map) == {(sid, layer) for sid in by_id for layer in layers},
             'Selection coverage mismatch')
    for (sid, layer), selection in selection_map.items():
        _require(selection['image_key'] == by_id[sid]['image_key'], 'Selection image mismatch')
        eligible = selection['eligible_heads']
        _require(len(set(eligible)) == len(eligible) and all(1 <= h < 32 for h in eligible),
                 'Eligible heads must be distinct, valid, and exclude shared head 0')
        _require(selection['k'] == int(bool(eligible)), 'Selection k/eligible mismatch')
        _require(selection['sink_head'] in eligible if eligible else selection['sink_head'] is None,
                 'Selected sinkS head is not eligible')
        stats = {h['head']: h for h in selection['head_statistics']}
        _require(len(stats) == len(selection['head_statistics']), 'Duplicate head statistics')
        for head in eligible:
            _require(head in stats and stats[head].get('valid') and stats[head]['category'] == 'sinkS',
                     'Eligible head lacks valid sinkS statistics')
        _require(set(selection['random_heads']) == {str(seed) for seed in seeds}, 'Random seed coverage mismatch')
        for head in selection['random_heads'].values():
            _require(1 <= head < 32 if eligible else head is None,
                     'Random control must match k and preserve shared head 0')
    groups = defaultdict(list)
    physical_identity = {}
    for row in predictions:
        _require(row['id'] in by_id, 'Predictions contain unknown/non-evaluation ID')
        _require(row['image_key'] == by_id[row['id']]['image_key'], 'Prediction image mismatch')
        if row['kind'] == 'reference':
            # An optional original-backend comparison is explicitly outside the 25-arm matrix.
            groups[row['id']].append(row)
        else:
            _require((row['kind'], row.get('layer'), float(row['gate']), row.get('seed')) in expected,
                     'Unexpected experimental arm')
            groups[row['id']].append(row)
        if row['kind'] == 'baseline' or (row['kind'] in ('sinkS', 'random') and row['k'] == 0):
            operation = ('baseline',)
        elif row['kind'] == 'reference':
            operation = ('reference', row['arm_id'])
        else:
            operation = (row['layer'], row['head'], float(row['gate']))
        signature = (row['id'], row['input_sha256'], row.get('backend'), operation, tuple(row['logits']))
        pid = row['physical_forward_id']
        _require(bool(pid) and bool(row['input_sha256']), 'Missing physical/input identity')
        if pid in physical_identity:
            _require(physical_identity[pid] == signature,
                     'One physical forward has inconsistent intervention identity/outputs/inputs')
        else:
            physical_identity[pid] = signature
    arm_protocols, records = {}, []
    for sid, inp in by_id.items():
        rows = groups[sid]
        _require(len({r['arm_id'] for r in rows}) == len(rows), 'Duplicate logical prediction arm')
        main_rows = [r for r in rows if r['kind'] != 'reference']
        actual = [(r['kind'], r.get('layer'), float(r['gate']), r.get('seed')) for r in main_rows]
        _require(len(actual) == len(expected) and set(actual) == expected, 'Incomplete logical arm matrix')
        base = next(r for r in main_rows if r['kind'] == 'baseline')
        _require(base['arm_id'] == 'baseline' and base.get('head') is None and base['k'] == 0,
                 'Malformed baseline arm')
        overlaps = candidate_ious(inp['candidate_boxes'], target_map[sid]['answer_boxes'], inp['width'], inp['height'])
        labels = np.where(overlaps > .5, overlaps, 0).astype(np.float32)
        baseline_pick = raw_top1(base['logits'])
        base_loss = focal_loss_fp32(base['logits'], labels)
        base_correct = bool(overlaps[baseline_pick] >= .5)
        intervention_identity = {}
        for row in rows:
            _require(len(row['logits']) == len(overlaps), 'Candidate/logit count mismatch')
            _require(row['input_sha256'] == base['input_sha256'], 'Inputs changed between arms')
            pick = raw_top1(row['logits'])
            _require(pick == row['prediction_index'], 'Stored prediction disagrees with raw-logit first-tie argmax')
            head_stat = None
            if row['kind'] in ('sinkS', 'random'):
                sel = selection_map[(sid, row['layer'])]
                head = sel['sink_head'] if row['kind'] == 'sinkS' else sel['random_heads'][str(row['seed'])]
                _require(row['head'] == head and row['k'] == sel['k'], 'Prediction/selection head mismatch')
                if sel['k'] == 0:
                    _require(row['logits'] == base['logits'] and
                             row['physical_forward_id'] == base['physical_forward_id'],
                             'No-op must reference the baseline physical forward')
                else:
                    head_stat = next((h for h in sel['head_statistics'] if h['head'] == head), None)
                    _require(head_stat is not None, 'Selected head statistics missing')
                    identity = (row['layer'], head, row['gate'])
                    if identity in intervention_identity:
                        _require(row['logits'] == intervention_identity[identity],
                                 'Duplicate identical intervention has inconsistent logits')
                    intervention_identity[identity] = row['logits']
            descriptor = (row['kind'], row.get('layer'), float(row['gate']), row.get('seed'))
            if row['arm_id'] in arm_protocols:
                _require(arm_protocols[row['arm_id']] == descriptor, 'Arm ID changes meaning across samples')
            arm_protocols[row['arm_id']] = descriptor
            loss = focal_loss_fp32(row['logits'], labels)
            correct = bool(overlaps[pick] >= .5)
            record = dict(row)
            record.update(prediction_box=clipped_boxes(inp['candidate_boxes'], inp['width'], inp['height'])[pick].tolist(),
                          selected_iou=float(overlaps[pick]), correct=correct,
                          baseline_correct=base_correct, baseline_prediction_index=baseline_pick,
                          prediction_unchanged=pick == baseline_pick,
                          candidate_covers_gt=bool((overlaps >= .5).any()),
                          task_loss=loss, baseline_loss=base_loss, delta_loss=loss-base_loss,
                          benefit=base_loss-loss, F=int(not base_correct and correct),
                          H=int(base_correct and not correct), head_statistics=head_stat,
                          eligible=bool(row['k']) if row['kind'] != 'reference' else False,
                          stat_valid=bool(head_stat and head_stat.get('valid') and
                                          head_stat.get('category') in ('vision', 'sinkG', 'sinkS')),
                          category=head_stat.get('category') if head_stat else None,
                          score=head_stat.get('score') if head_stat else None)
            records.append(record)
    _require(set(groups) == set(by_id), 'Prediction coverage mismatch')
    # No arm may disappear in another expression, including optional backend references.
    _require(all(len([r for r in records if r['arm_id'] == arm]) == len(by_id)
                 for arm in arm_protocols), 'An arm has incomplete evaluation coverage')
    return eval_rows, records, selection_map, arm_protocols


def _conclusions(effects, predictivity):
    hard = effects['0']['aggregate']['eligible']
    c, d = hard['C']['ci95'], hard['D']['ci95']
    frequency = c is not None and c[0] > 0
    intensity = d is not None and d[1] < 0
    frequency_reverse = c is not None and c[1] < 0
    intensity_reverse = d is not None and d[0] > 0
    if hard['C']['n_expressions'] == 0 or c is None or d is None:
        category = 'unidentifiable'
    elif frequency and intensity:
        category = 'supported'
    elif (frequency or intensity) and not (frequency_reverse or intensity_reverse):
        category = 'partial_support'
    else:
        category = 'not_supported_or_reverse'
    auc = predictivity['0']['macro']['auroc']
    continuous = ('supported' if auc['ci95'] and auc['ci95'][0] > .5 else
                  'unidentifiable' if auc['value'] is None or auc['ci95'] is None else 'uncertain_or_reverse')
    return dict(category_comparison=category, more_often_harmful_supported=frequency,
                loss_intensity_supported=intensity, more_often_harmful_reverse=frequency_reverse,
                loss_intensity_reverse=intensity_reverse, continuous_score_prediction=continuous,
                notes=['A negative scientific result is not an engineering failure.',
                       'Inference is conditional on eligible sample/layer contexts.',
                       'No result establishes re-sinking, training-gradient causality, or actual acceleration.'])


def analyze_run(inputs, predictions, selections, targets, config, consistency, resources):
    """Analyze complete frozen inputs and logical arms; return report plus records."""
    _require(consistency.get('passed') is True, 'Consistency gate has not passed')
    eps = float(consistency['eps'])
    _require(math.isfinite(eps) and eps >= 1e-6, 'Invalid frozen label-free noise threshold')
    eval_rows, records, sel_map, protocols = _validate_and_evaluate(
        inputs, predictions, selections, targets, config)
    n = len(eval_rows)
    expected_counts = config.get('counts', {})
    _require(n == expected_counts.get('evaluation', n), 'Evaluation count differs from frozen config')
    calibration = [r for r in inputs if r['split'] == 'calibration']
    _require(len(calibration) == expected_counts.get('calibration', len(calibration)),
             'Calibration count differs from frozen config')
    _require(not ({r['image_key'] for r in calibration} & {r['image_key'] for r in eval_rows}),
             'Calibration/evaluation image leakage')
    ids, images = [r['id'] for r in eval_rows], [r['image_key'] for r in eval_rows]
    index = {sid: i for i, sid in enumerate(ids)}
    repeats = int(config.get('analysis', {}).get('bootstrap_repeats', 2000))
    bootstrap_seed = int(config.get('analysis', {}).get('bootstrap_seed', 20260925))
    weights = image_bootstrap_weights(images, repeats, bootstrap_seed)
    layers, seeds, gates = config['layers'], config['random_seeds'], config['gates']
    by_arm, by_protocol = {}, {}
    for arm, descriptor in protocols.items():
        rows = sorted((r for r in records if r['arm_id'] == arm), key=lambda r: index[r['id']])
        correct = np.asarray([r['correct'] for r in rows], float)
        changed = correct - np.asarray([r['baseline_correct'] for r in rows], float)
        f, h = sum(r['F'] for r in rows), sum(r['H'] for r in rows)
        _require(int(changed.sum()) == f-h, 'F-H/Top1 identity failed')
        coverage = sum(r['candidate_covers_gt'] for r in rows)
        by_arm[arm] = dict(kind=descriptor[0], layer=descriptor[1], gate=descriptor[2], seed=descriptor[3],
                           N=n, correct=int(correct.sum()), top1=float(correct.mean()),
                           F=f, H=h, net_repairs=f-h, delta_top1=float(changed.mean()),
                           mean_loss=float(np.mean([r['task_loss'] for r in rows])),
                           mean_delta_loss=float(np.mean([r['delta_loss'] for r in rows])),
                           candidate_coverage=coverage/n, covered_expressions=coverage,
                           eligible_expressions=sum(r['eligible'] for r in rows),
                           top1_interval=_summary_value(correct, weights, images),
                           delta_top1_interval=_summary_value(changed, weights, images),
                           delta_loss_interval=_summary_value([r['delta_loss'] for r in rows], weights, images))
        by_protocol[descriptor] = rows
    random_summary, effects, predictivity = {}, {}, {}
    eligible = np.asarray([[sel_map[(sid, layer)]['k'] > 0 for layer in layers] for sid in ids])
    for gate in gates:
        key = _gate_key(gate)
        ds = np.asarray([[r['delta_loss'] for r in by_protocol[('sinkS', layer, float(gate), None)]]
                         for layer in layers]).T
        dr = np.asarray([[[r['delta_loss'] for r in by_protocol[('random', layer, float(gate), seed)]]
                          for seed in seeds] for layer in layers]).transpose(2, 0, 1)
        effects[key] = dict(layers={}, aggregate=_effect_summary(ds, dr, eligible, weights, images, eps))
        pred_layers, pred_boot = {}, {}
        for li, layer in enumerate(layers):
            effects[key]['layers'][str(layer)] = _effect_summary(ds[:, li], dr[:, li], eligible[:, li],
                                                                 weights, images, eps)
            arm_ids = [next(arm for arm, desc in protocols.items()
                           if desc == ('random', layer, float(gate), seed)) for seed in seeds]
            fields = ('top1', 'F', 'H', 'net_repairs', 'mean_loss', 'mean_delta_loss')
            random_summary[f'layer{layer}_gate{key}'] = dict(
                seeds=list(seeds), arms=arm_ids,
                metrics={field: dict(values=[by_arm[arm][field] for arm in arm_ids],
                                     mean=float(np.mean([by_arm[arm][field] for arm in arm_ids])),
                                     minimum=min(by_arm[arm][field] for arm in arm_ids),
                                     maximum=max(by_arm[arm][field] for arm in arm_ids)) for field in fields})
            random_records = [r for seed in seeds for r in by_protocol[('random', layer, float(gate), seed)]
                              if r['eligible']]
            pred_layers[str(layer)], pred_boot[str(layer)] = _prediction_metric(
                random_records, index, weights, images, eps)
        macro = {}
        for metric in ('auroc', 'spearman'):
            point = [pred_layers[str(layer)][metric]['value'] for layer in layers]
            valid = [v for v in point if v is not None]
            boot = np.asarray([pred_boot[str(layer)][metric] for layer in layers])
            # np.mean deliberately propagates a missing preregistered layer.
            values = np.mean(boot, axis=0)
            macro[metric] = _interval(np.mean(valid) if len(valid) == len(layers) else math.nan,
                                      values, len(set(images)))
            macro[metric].update(preregistered_layers=len(layers), available_layers=len(valid),
                                 available_layers_descriptive_mean=float(np.mean(valid)) if valid else None)
        predictivity[key] = dict(layers=pred_layers, macro=macro,
                                population='Random single-head arms in eligible sample/layer contexts; seeds retained, images clustered')
    # Sample-balanced task comparison across the same preregistered layer set.
    task_aggregate = {}
    baseline_rows = by_protocol[('baseline', None, 1., None)]
    base_correct = np.asarray([r['correct'] for r in baseline_rows], float)
    for gate in gates:
        for kind in ('sinkS', 'random'):
            arms = [rows for desc, rows in by_protocol.items() if desc[0] == kind and desc[2] == float(gate)]
            correctness = np.asarray([[r['correct'] for r in rows] for rows in arms], float).mean(axis=0)
            deltas = np.asarray([[r['delta_loss'] for r in rows] for rows in arms]).mean(axis=0)
            fixes = float(np.mean([sum(r['F'] for r in rows) for rows in arms]))
            harms = float(np.mean([sum(r['H'] for r in rows) for rows in arms]))
            task_aggregate[f'{kind}_gate{_gate_key(gate)}'] = dict(
                top1=float(correctness.mean()), F_mean_across_arms=fixes, H_mean_across_arms=harms,
                net_repairs_mean_across_arms=fixes-harms,
                delta_top1=_summary_value(correctness-base_correct, weights, images),
                delta_loss=_summary_value(deltas, weights, images))
    report = dict(status='PASSED', protocol='SADS-inspired frozen full36 single-head pilot',
                  noise_eps=eps, counts=dict(calibration=len(calibration), evaluation=n,
                                            evaluation_images=len(set(images)),
                                            logical_arms=len(protocols), eligible_sample_layers=int(eligible.sum()),
                                            eligible_expressions=int(eligible.any(axis=1).sum())),
                  inference=dict(unit='image', bootstrap_seed=bootstrap_seed, bootstrap_repeats=repeats,
                                 minimum_ci_images=2, minimum_estimable_bootstrap_fraction=.8,
                                 aggregate_weighting='Equal layers within expression, then equal expressions; random seeds equally weighted',
                                 macro_missing_layer_policy='Primary macro NA if any preregistered layer is unavailable; available-layer mean descriptive only'),
                  arms=by_arm, random_seed_summary=random_summary, task_aggregate=task_aggregate,
                  effects=effects, predictivity=predictivity,
                  resources=_resource_summary(resources, predictions),
                  eval_predictions=records)
    report['conclusions'] = _conclusions(effects, predictivity)
    report['conclusions']['sinkS_hard_top1_net_benefit'] = task_aggregate['sinkS_gate0']['net_repairs_mean_across_arms'] > 0
    return report


def _fmt(value, digits=5):
    return 'NA' if value is None else f'{value:.{digits}g}'


def _fmt_estimate(record):
    point = _fmt(record['value'])
    ci = record.get('ci95')
    return f'{point} [{_fmt(ci[0])}, {_fmt(ci[1])}]' if ci else f'{point} [CI: NA]'


def render_markdown(report):
    """Render all arms and the preregistered effects, including negative findings."""
    counts, evidence = report['counts'], report['conclusions']
    lines = ['# SADS-inspired 冻结单头干预报告', '',
             f"工程状态：{report['status']}。效果集 {counts['evaluation']} 条 / {counts['evaluation_images']} 张图；"
             f"eligible {counts['eligible_expressions']} 条表达、{counts['eligible_sample_layers']} 个 sample×layer。", '',
             f"硬抑制类别比较：**{evidence['category_comparison']}**；连续分数预测：**{evidence['continuous_score_prediction']}**。",
             f"冻结无标签噪声门槛 eps={report['noise_eps']:.9g}；B=原损失−干预损失，B>eps 才计有害 head。", '',
             '以下所有 Top-1 都按 raw-logit 首个最大值，IoU≥0.5；全部样本/no-op/未覆盖样本保留。', '',
             '| 臂 | N | Top-1 | F | H | F−H | Δloss | eligible |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, arm in report['arms'].items():
        lines.append(f"| {name} | {arm['N']} | {arm['top1']:.2%} | {arm['F']} | {arm['H']} | "
                     f"{arm['net_repairs']} | {arm['mean_delta_loss']:.6g} | {arm['eligible_expressions']} |")
    lines += ['', '随机种子全部保留，以下为三 seed 的均值 [最小, 最大]，不选最佳 seed。', '',
              '| 层/gate | Top-1 | F−H | Δloss |', '|---|---:|---:|---:|']
    for name, row in report['random_seed_summary'].items():
        values = []
        for metric in ('top1', 'net_repairs', 'mean_delta_loss'):
            v = row['metrics'][metric]
            values.append(f"{_fmt(v['mean'])} [{_fmt(v['minimum'])}, {_fmt(v['maximum'])}]")
        lines.append(f"| {name} | {' | '.join(values)} |")
    lines += ['', 'C>0表示 sinkS 更常有害，D<0表示其抑制收益更大；括号是按 image 配对的95%区间。', '',
              '| gate | 范围 | 分母 | C | D |', '|---|---|---|---:|---:|']
    for gate, result in report['effects'].items():
        for name, group in [('aggregate', result['aggregate']), *result['layers'].items()]:
            for population, row in group.items():
                lines.append(f"| {gate} | {name}/{population} | {row['C']['n_expressions']} 表达 | "
                             f"{_fmt_estimate(row['C'])} | {_fmt_estimate(row['D'])} |")
    lines += ['', '只用随机单头臂检验预测性；主结果为 hard gate=0，soft=.5是固定辅助结果。', '',
              '| gate | 层 | AUROC | Spearman |', '|---|---|---:|---:|']
    for gate, result in report['predictivity'].items():
        for name, row in [*result['layers'].items(), ('macro', result['macro'])]:
            lines.append(f"| {gate} | {name} | {_fmt_estimate(row['auroc'])} | {_fmt_estimate(row['spearman'])} |")
    r = report['resources']
    lines += ['', f"资源：{r['logical_prediction_rows']} 条逻辑结果引用 {r['unique_prediction_forwards']} 次实际预测前向；"
              f"资源账本共 {r['total_physical_forwards']} 次实际前向。重复引用不重复计时。", '',
              '| 范围 | 物理前向 | forward ms均值/p95 | request ms均值/p95 | peak allocated/reserved bytes |',
              '|---|---:|---:|---:|---:|']
    for name, row in r['scopes'].items():
        fw, req = row['forward_ms'], row['request_ms']
        lines.append(f"| {name} | {row['physical_forwards']} | {_fmt(fw['mean'])}/{_fmt(fw['p95'])} | "
                     f"{_fmt(req['mean'])}/{_fmt(req['p95'])} | {_fmt(row['peak_allocated_bytes']['maximum'])}/"
                     f"{_fmt(row['peak_reserved_bytes']['maximum'])} |")
    lines += ['', f"已记录命令的墙钟耗时之和：{_fmt(r.get('job_stage_wall_seconds_sum'))} 秒"
              f"（{r.get('recorded_command_invocations', 0)} 次调用，其中失败 {r.get('failed_command_invocations', 0)} 次）；"
              f"报告计算：{_fmt(r.get('report_compute_wall_seconds'))} 秒，单列。",
              '命令墙钟包含模型加载、哈希及该命令内的前向/读写开销，不能再与forward/request耗时相加；'
              '这是已记录命令耗时的总和，不含命令之间的用户等待。报告计时截至分析完成、发布产物之前；'
              '续跑复用首次完成的计算记录，不把新测量时间覆盖旧记录。']
    if not evidence['sinkS_hard_top1_net_benefit']:
        lines += ['', '**硬抑制 sinkS 未获得候选 Top-1 的净收益**，即使损失指标方向有利也不能掩盖此项。']
    lines += ['', '未知/无稳定双峰/无有效区间均保守报告无证据，不把阴性结果当工程失败。',
              '跨层先在每条表达内等权平均，再对表达等权；bootstrap携带同图全部表达/层/seed，单层结果不作未经多重比较修正的显著性宣称。',
              '至少两张独立图且至少80% bootstrap可估计才给CI；跨层主预测指标缺任一预注册层即NA。',
              '保守实现假设：B>eps仅有单一效应类别时，主AUROC和主Spearman均记NA；有效重复数见metrics。',
              '历史开发池结果不代表独立测试泛化；本实验不能证明 re-sinking、训练梯度机制或实际加速。',
              'collector与普通前向分开计时；head输出置零仍执行QKV、attention和o_proj。', '']
    return '\n'.join(lines)


def _timing_events(run):
    """Read a frozen snapshot of actual command invocations, never infer durations."""
    from ref_sads_io import read_json, safe_child
    files, events = [], []
    for path in sorted((run / 'timings').glob('*.json')):
        name = path.relative_to(run).as_posix()
        record = read_json(safe_child(run, name))
        _require(isinstance(record.get('command'), str) and bool(record['command']), 'Invalid timing command')
        _require(type(record.get('succeeded')) is bool, 'Invalid timing success flag')
        _require(isinstance(record.get('wall_seconds'), (int, float)) and
                 not isinstance(record['wall_seconds'], bool) and _finite(record['wall_seconds']) and
                 record['wall_seconds'] >= 0, 'Invalid measured command wall time')
        files.append(name)
        events.append(dict(record, phase='command_'+record['command'], source_file=name))
    return files, events


def _promote_writing(path, validate):
    """Recover an already complete exclusive atomic write after an interruption.

    A truncated/corrupt .writing file is retained and rejected for inspection.
    Only a validated complete byte stream is published, without replacing files.
    """
    pending = path.with_name(path.name+'.writing')
    if pending.exists():
        validate(pending)
        if path.exists():
            _require(path.read_bytes() == pending.read_bytes(), f'Pending/final content mismatch: {path.name}')
        else:
            os.link(pending, path)
        pending.unlink()


def _publish_frozen(path, value, kind, resume):
    """Exclusive creation or explicit, content-identical resume, never overwrite."""
    from ref_sads_io import read_json, read_jsonl, write_json, write_jsonl, _write
    def validate(existing):
        if kind == 'json':
            content = read_json(existing)
        elif kind == 'jsonl':
            content = read_jsonl(existing)
        else:
            content = existing.read_text(encoding='utf-8')
        _require(content == value, f'Report resume content mismatch: {path.name}')
    if resume:
        _promote_writing(path, validate)
    if path.exists():
        _require(resume, f'Report outputs already exist; refusing overwrite: {path.name}')
        validate(path)
    elif kind == 'json':
        write_json(path, value)
    elif kind == 'jsonl':
        write_jsonl(path, value)
    else:
        _write(path, value.encode('utf-8'))


def _finish_complete(run, report, resume):
    from ref_sads_io import digest
    value = dict(status='PASSED', report_stage_sha256=digest(run / 'stages/report.json'),
                 scientific_conclusion=report['conclusions'])
    _publish_frozen(run / 'COMPLETE.json', value, 'json', resume)


def report_run(run, resume=False):
    """Publish or resume one report using a frozen, checksummed computation.

    ``prepared.json`` is saved before any final report output. Its measured
    compute duration and timing-file snapshot are reused exactly on resume.
    This prevents a retry's new duration from conflicting with existing metrics.
    """
    from ref_sads_io import (digest, load_run, marker_path, part_read, part_write,
                             read_json, read_jsonl, require_stage, safe_child,
                             stage_lock, write_stage)
    started = time.perf_counter()
    run = Path(run).resolve()
    loaded = load_run(run)
    require_stage(run, 'intervene')
    marker = marker_path(run, 'report')
    if marker.exists():
        _require(resume, 'Report already exists; refusing overwrite; use explicit --resume to verify/recover COMPLETE')
        require_stage(run, 'report')
        report = read_json(run / 'metrics.json')
        _finish_complete(run, report, True)
        return report
    _require(not (run / 'COMPLETE.json').exists(), 'COMPLETE exists without a verified report stage')
    output_names = ['eval_predictions.jsonl', 'metrics.json', 'report.md']
    consumed_names = ['inputs.jsonl', 'predictions.jsonl', 'selections.jsonl', 'eval_targets.jsonl',
                      'config.json', 'checks/consistency.json', 'resources.jsonl']
    manifest_sha = digest(run / 'manifest.json')
    with stage_lock(run, 'report', resume) as partial:
        prepared_path = partial / 'prepared.json'
        def validate_prepared(path):
            frozen = part_read(path, manifest_sha)
            _require(set(frozen['input_sha256']) == set(consumed_names + frozen['timing_files']),
                     'Frozen report input list mismatch')
            for name, sha in frozen['input_sha256'].items():
                _require(digest(safe_child(run, name)) == sha, f'Frozen report input changed: {name}')
            return frozen
        if resume:
            _promote_writing(prepared_path, validate_prepared)
        if prepared_path.exists():
            _require(resume, 'Partial report already exists; use explicit --resume')
            frozen = validate_prepared(prepared_path)
        else:
            _require(not any((run / name).exists() for name in output_names),
                     'Report output exists without a frozen computation; refusing overwrite')
            timing_files, events = _timing_events(run)
            input_hashes = {name: digest(safe_child(run, name)) for name in consumed_names + timing_files}
            report = analyze_run(loaded['inputs'], read_jsonl(run / 'predictions.jsonl'),
                                 read_jsonl(run / 'selections.jsonl'), read_jsonl(run / 'eval_targets.jsonl'),
                                 loaded['config'], read_json(run / 'checks/consistency.json'),
                                 read_jsonl(run / 'resources.jsonl') + events)
            records = report.pop('eval_predictions')
            report['resources']['report_compute_wall_seconds'] = time.perf_counter() - started
            report['resources']['timing_snapshot_files'] = timing_files
            frozen = dict(report=report, records=records, markdown=render_markdown(report),
                          timing_files=timing_files, input_sha256=input_hashes)
            part_write(prepared_path, frozen, manifest_sha)
            # Recheck after analysis so input mutation cannot be sealed into a report.
            validate_prepared(prepared_path)
        report = frozen['report']
        for name, value, kind in ((output_names[0], frozen['records'], 'jsonl'),
                                   (output_names[1], report, 'json'),
                                   (output_names[2], frozen['markdown'], 'text')):
            _publish_frozen(run / name, value, kind, resume)
        files = output_names + ['parts/report/prepared.json'] + frozen['timing_files']
        # The stage marker itself can also have been fully written before a crash.
        if resume:
            def validate_pending_marker(path):
                value = read_json(path)
                _require(value.get('stage') == 'report' and value.get('status') == 'PASSED' and
                         value.get('manifest_sha256') == manifest_sha, 'Invalid pending report marker')
                _require(value.get('parents') == {'intervene': digest(marker_path(run, 'intervene'))},
                         'Pending report parent mismatch')
                _require(value.get('files') == {name: digest(safe_child(run, name)) for name in files},
                         'Pending report artifacts mismatch')
            _promote_writing(marker, validate_pending_marker)
        if not marker.exists():
            write_stage(run, 'report', files, metadata={'status': report['status']}, parents=['intervene'])
        require_stage(run, 'report')
        _finish_complete(run, report, resume)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--resume', action='store_true',
                        help='Verify frozen partial outputs and resume without overwriting or recomputing recorded timing')
    args = parser.parse_args()
    report = report_run(args.run, args.resume)
    print(f"SADS-inspired report: {report['status']} ({report['conclusions']['category_comparison']})")


if __name__ == '__main__':
    main()
