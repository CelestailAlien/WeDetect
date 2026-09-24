"""Label-free SADS-inspired calibration and matched, reproducible head selection.

Pure NumPy/stdlib: no model loading, file I/O, or evaluation targets. Records must
explicitly identify unmodified forward statistics with ``stats_source=baseline``.
All indices except the 1-based layer number are zero based.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Any

import numpy as np


DEFAULT_GMM = dict(init_seeds=[0, 1, 2, 3, 4], max_iter=200, tol=1e-4,
                   reg_covar=1e-6, min_observations=200, min_images=20,
                   min_bic_gain=10., min_weight=.1, bootstrap_repeats=20,
                   bootstrap_min_success=16, bootstrap_iqr_fraction=.1,
                   bootstrap_seed=20260924)
_FORBIDDEN = {"gt", "label", "labels", "target", "targets", "iou", "ious",
              "correct", "correctness", "loss", "losses", "accuracy", "answer",
              "gold", "groundtruth"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def _stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(_canonical(parts).encode("utf-8")).digest()[:16], "big")


def _reject_targets(value: Any, path: str = "row") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = re.sub(r"IoU|IOU", "_iou_", str(key))
            snake_key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", normalized_key)
            words = set(re.findall(r"[a-z0-9]+", snake_key.lower()))
            if words & _FORBIDDEN or {"ground", "truth"} <= words:
                raise ValueError(f"GT/evaluation field forbidden in label-free input: {path}.{key}")
            _reject_targets(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_targets(nested, f"{path}[{index}]")


def _spec(config: dict) -> dict:
    gmm = dict(DEFAULT_GMM, **config.get("gmm", {}))
    spec = dict(layers=list(config.get("layers", [28, 32, 36])),
                num_heads=config.get("num_heads", 32), shared_head=config.get("shared_head", 0),
                selection_seed=config.get("selection_seed", 20260923),
                random_seeds=list(config.get("random_seeds", [11, 29, 47])), gmm=gmm)
    if (not spec["layers"] or any(type(x) is not int or x < 1 for x in spec["layers"])
            or len(set(spec["layers"])) != len(spec["layers"])):
        raise ValueError("layers must contain distinct positive 1-based integers")
    if type(spec["num_heads"]) is not int or spec["num_heads"] < 2:
        raise ValueError("num_heads must be an integer >=2")
    if type(spec["shared_head"]) is not int or not 0 <= spec["shared_head"] < spec["num_heads"]:
        raise ValueError("shared_head must be a valid head index")
    for key in ("random_seeds",):
        seeds = spec[key]
        if not seeds or any(type(s) is not int for s in seeds) or len(seeds) != len(set(seeds)):
            raise ValueError(f"{key} must contain distinct integer seeds")
    if type(spec["selection_seed"]) is not int:
        raise ValueError("selection_seed must be an integer")
    seeds = gmm["init_seeds"]
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("GMM init_seeds must be distinct nonnegative integers")
    for key in ("max_iter", "min_observations", "min_images", "bootstrap_repeats", "bootstrap_min_success"):
        if type(gmm[key]) is not int or gmm[key] < 1:
            raise ValueError(f"gmm.{key} must be a positive integer")
    if gmm["bootstrap_min_success"] > gmm["bootstrap_repeats"]:
        raise ValueError("bootstrap_min_success exceeds bootstrap_repeats")
    for key in ("tol", "reg_covar", "bootstrap_iqr_fraction"):
        if not math.isfinite(float(gmm[key])) or gmm[key] <= 0:
            raise ValueError(f"gmm.{key} must be positive and finite")
    if (not 0 < gmm["min_weight"] <= .5 or not math.isfinite(float(gmm["min_bic_gain"]))
            or gmm["min_bic_gain"] < 0):
        raise ValueError("invalid min_weight or min_bic_gain")
    if type(gmm["bootstrap_seed"]) is not int or gmm["bootstrap_seed"] < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    return spec


def _signature(spec: dict) -> str:
    return hashlib.sha256(_canonical(spec).encode("utf-8")).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and bool(np.isfinite(value))


def _x_valid(row: dict) -> bool:
    return bool(row.get("x_valid", row["valid"])) and _finite(row.get("x")) and 0 <= row["x"] <= 1 + 1e-9


def _entropy_valid(row: dict) -> bool:
    return (bool(row.get("entropy_valid", row["valid"])) and _finite(row.get("H"))
            and row["H"] >= 0 and _finite(row.get("e")) and 0 <= row["e"] <= 1 + 1e-9)


def _validate_rows(rows: list[dict], spec: dict, split: str) -> list[dict]:
    rows = list(rows)
    if not rows:
        raise ValueError("head statistics must not be empty")
    groups: dict[tuple, dict] = {}
    sample_layers: dict[str, set] = defaultdict(set)
    sample_images: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each head record must be a dict")
        _reject_targets(row)
        for key in ("id", "image_key", "split", "layer", "head", "x", "H", "e", "valid", "stats_source"):
            if key not in row:
                raise ValueError(f"missing required head field: {key}")
        if row["split"] != split:
            raise ValueError(f"expected {split} records exclusively")
        if row["stats_source"] != "baseline":
            raise ValueError("only stats_source='baseline' can be used for calibration/selection")
        for key in ("gate", "head_gate", "gates"):
            if key in row:
                gate_values = row[key] if isinstance(row[key], (list, tuple)) else [row[key]]
                if any(not _finite(v) or v != 1 for v in gate_values):
                    raise ValueError("intervened statistics cannot be used for head selection")
        if row.get("intervened", False) or row.get("selected_heads"):
            raise ValueError("intervened statistics cannot be used for head selection")
        if "arm" in row and row["arm"] not in ("baseline", "B0", "no_intervention"):
            raise ValueError("nonbaseline arm cannot be used for head selection")
        for key in ("id", "image_key"):
            if type(row[key]) not in (str, int) or row[key] == "":
                raise ValueError(f"{key} must be a nonempty string or integer")
        if type(row["layer"]) is not int or row["layer"] not in spec["layers"]:
            raise ValueError("unexpected layer")
        if type(row["head"]) is not int or not 0 <= row["head"] < spec["num_heads"]:
            raise ValueError("head index out of range")
        for key in ("valid", "x_valid", "entropy_valid"):
            if key in row and type(row[key]) is not bool:
                raise ValueError(f"{key} must be boolean")
        sample = _canonical(row["id"])
        image = _canonical(row["image_key"])
        if sample in sample_images and sample_images[sample] != image:
            raise ValueError("one expression ID cannot refer to multiple images")
        sample_images[sample] = image
        sample_layers[sample].add(row["layer"])
        group = groups.setdefault((sample, row["layer"]), {})
        if row["head"] in group:
            raise ValueError("duplicate sample/layer/head record")
        group[row["head"]] = row
    if any(set(group) != set(range(spec["num_heads"])) for group in groups.values()):
        raise ValueError("each sample/layer must have the complete head set")
    if any(layers != set(spec["layers"]) for layers in sample_layers.values()):
        raise ValueError("each sample must include every configured layer")
    return sorted(rows, key=lambda r: (_canonical(r["id"]), r["layer"], r["head"]))


def _log_components(x: np.ndarray, weights: np.ndarray, means: np.ndarray,
                    variances: np.ndarray) -> np.ndarray:
    return (np.log(weights)[None, :] - .5 * np.log(2 * np.pi * variances)[None, :]
            - .5 * (x[:, None] - means[None, :]) ** 2 / variances[None, :])


def _log_density(x: np.ndarray, weights: np.ndarray, means: np.ndarray,
                 variances: np.ndarray) -> np.ndarray:
    logp = _log_components(x, weights, means, variances)
    maximum = logp.max(axis=1)
    return maximum + np.log(np.exp(logp - maximum[:, None]).sum(axis=1))


def _em(x: np.ndarray, components: int, seed: int, g: dict) -> dict:
    if components == 1:
        means = np.array([x.mean()])
        variances = np.array([x.var() + g["reg_covar"]])
        weights = np.array([1.])
        converged, iterations = True, 1
    else:
        rng = np.random.default_rng(seed)
        means = np.quantile(x, [.25, .75]) + rng.normal(0, .05, 2)
        variances = np.full(2, x.var() + g["reg_covar"])
        weights = np.full(2, .5)
        previous = None
        converged = False
        for iterations in range(1, g["max_iter"] + 1):
            logp = _log_components(x, weights, means, variances)
            maxlog = logp.max(axis=1, keepdims=True)
            responsibility = np.exp(logp - maxlog)
            responsibility /= responsibility.sum(axis=1, keepdims=True)
            count = responsibility.sum(axis=0)
            if np.any(count <= np.finfo(float).eps):
                return dict(seed=seed, converged=False, iterations=iterations, reason="empty_component")
            weights = count / len(x)
            means = (responsibility * x[:, None]).sum(axis=0) / count
            variances = ((responsibility * (x[:, None] - means) ** 2).sum(axis=0)
                         / count + g["reg_covar"])
            likelihood = float(_log_density(x, weights, means, variances).sum())
            if not math.isfinite(likelihood):
                return dict(seed=seed, converged=False, iterations=iterations, reason="nonfinite_likelihood")
            if previous is not None and abs(likelihood - previous) / len(x) < g["tol"]:
                converged = True
                break
            previous = likelihood
    order = np.argsort(means, kind="stable")
    weights, means, variances = weights[order], means[order], variances[order]
    likelihood = float(_log_density(x, weights, means, variances).sum())
    parameters = 3 * components - 1
    return dict(seed=seed, converged=converged, iterations=iterations,
                reason="converged" if converged else "max_iter", log_likelihood=likelihood,
                bic=float(parameters * np.log(len(x)) - 2 * likelihood),
                weights=weights.tolist(), means=means.tolist(), variances=variances.tolist())


def _stationary_points(fit: dict) -> dict:
    """Find density modes/valley, not Gaussian intersections or tail minima."""
    weights = np.asarray(fit["weights"])
    means = np.asarray(fit["means"])
    variances = np.asarray(fit["variances"])
    if not means[0] < means[1]:
        return dict(valid=False, reason="coincident_means", points=[])
    # Include tails so an extremely separated mode rounded exactly to its
    # component mean still has both derivative signs in the search domain.
    lo = float(means[0] - 8 * np.sqrt(variances[0]))
    hi = float(means[1] + 8 * np.sqrt(variances[1]))
    grid = [np.linspace(lo, hi, 4097)]
    for mu, var in zip(means, variances):
        grid.append(np.clip(np.linspace(mu - 8 * np.sqrt(var), mu + 8 * np.sqrt(var), 1025), lo, hi))
    grid = np.unique(np.concatenate(grid))

    def derivative(values: np.ndarray) -> np.ndarray:
        logp = _log_components(values, weights, means, variances)
        scaled = np.exp(logp - logp.max(axis=1, keepdims=True))
        return (scaled * (means[None, :] - values[:, None]) / variances).sum(axis=1)

    signs = np.sign(derivative(grid))
    nonzero = np.flatnonzero(signs)
    points = []
    for left, right in zip(nonzero[:-1], nonzero[1:]):
        if signs[left] == signs[right]:
            continue
        a, b, sign_a = float(grid[left]), float(grid[right]), signs[left]
        for _ in range(70):
            mid = (a + b) / 2
            sign_mid = float(np.sign(derivative(np.array([mid]))[0]))
            if sign_mid == 0:
                a = b = mid
                break
            if sign_mid == sign_a:
                a = mid
            else:
                b = mid
        position = (a + b) / 2
        logp = float(_log_density(np.array([position]), weights, means, variances)[0])
        points.append(dict(position=position, kind="mode" if sign_a > 0 else "valley", log_density=logp))
    valid = len(points) == 3 and [p["kind"] for p in points] == ["mode", "valley", "mode"]
    if valid:
        valid = (means[0] < points[1]["position"] < means[1]
                 and points[1]["log_density"] < min(points[0]["log_density"], points[2]["log_density"]))
    return dict(valid=valid, reason="bimodal" if valid else "no_two_modes_and_internal_valley", points=points,
                valley=points[1]["position"] if valid else None)


def _fit_threshold(values: list[float], images: list[Any], g: dict, original: bool) -> dict:
    x = np.asarray(values, dtype=np.float64)
    count_images = len({_canonical(i) for i in images})
    result = dict(valid=False, threshold=None, n=len(x), n_images=count_images,
                  mean=None, sigma=None, iqr=None, fits={})
    if len(x) < (g["min_observations"] if original else 2):
        return dict(result, reason="insufficient_observations")
    if original and count_images < g["min_images"]:
        return dict(result, reason="insufficient_images")
    mean, sigma = float(x.mean()), float(x.std())
    iqr = float(np.quantile(x, .75) - np.quantile(x, .25))
    result.update(mean=mean, sigma=sigma, iqr=iqr)
    if not all(math.isfinite(v) for v in (mean, sigma, iqr)) or sigma <= 0 or iqr <= 0:
        return dict(result, reason="degenerate_scale")
    z = (x - mean) / sigma
    for components in (1, 2):
        initializations = [_em(z, components, seed, g) for seed in g["init_seeds"]]
        converged = [fit for fit in initializations if fit["converged"]]
        best = max(converged, key=lambda f: f["log_likelihood"]) if converged else None
        result["fits"][str(components)] = dict(initializations=initializations, selected=best)
    one, two = (result["fits"][str(k)]["selected"] for k in (1, 2))
    if one is None or two is None:
        return dict(result, reason="no_converged_fit")
    result["bic_gain"] = one["bic"] - two["bic"]
    shape = _stationary_points(two)
    # Preserve the standardized audit and original-unit locations for inspection.
    shape["points_original"] = [dict(p, position=mean + sigma * p["position"],
                                     log_density=p["log_density"] - math.log(sigma))
                                for p in shape["points"]]
    result["shape"] = shape
    if result["bic_gain"] < g["min_bic_gain"]:
        return dict(result, reason="bic_gain_too_small")
    if min(two["weights"]) < g["min_weight"]:
        return dict(result, reason="small_component_weight")
    if not shape["valid"]:
        return dict(result, reason=shape["reason"])
    result.update(valid=True, threshold=float(mean + sigma * shape["valley"]), reason="structurally_bimodal")
    return result


def _two_stage(rows: list[dict], g: dict, original: bool) -> tuple[dict, dict]:
    usable_x = [r for r in rows if _x_valid(r)]
    alpha = _fit_threshold([float(r["x"]) for r in usable_x], [r["image_key"] for r in usable_x], g, original)
    if not alpha["valid"]:
        return alpha, dict(valid=False, threshold=None, reason="alpha_invalid", n=0, n_images=0,
                           mean=None, sigma=None, iqr=None, fits={})
    sinks = [r for r in usable_x if r["x"] < alpha["threshold"] and _entropy_valid(r)]
    beta = _fit_threshold([float(r["e"]) for r in sinks], [r["image_key"] for r in sinks], g, original)
    return alpha, beta


def _stability(fit: dict, fits: list[dict], g: dict) -> dict:
    valleys = [f["threshold"] for f in fits if f["valid"]]
    iqr = float(np.quantile(valleys, .75) - np.quantile(valleys, .25)) if valleys else None
    maximum = g["bootstrap_iqr_fraction"] * fit["iqr"] if fit.get("iqr") is not None else None
    success = (fit["valid"] and len(valleys) >= g["bootstrap_min_success"]
               and iqr is not None and maximum is not None and iqr <= maximum)
    return dict(valid=bool(success), successes=len(valleys), required=g["bootstrap_min_success"],
                threshold_iqr=iqr, max_threshold_iqr=maximum, thresholds=valleys,
                reason="stable" if success else "bootstrap_unstable")


def calibrate_heads(rows: list[dict], config: dict) -> dict:
    """Fit two-stage per-layer thresholds only on disjoint, label-free calibration."""
    spec = _spec(config)
    rows = _validate_rows(rows, spec, "calibration")
    g = spec["gmm"]
    output = dict(schema_version=1, stats_source="baseline", split="calibration", config=spec,
                  config_signature=_signature(spec), layers={},
                  calibration_ids=sorted({r["id"] for r in rows}, key=_canonical),
                  calibration_image_keys=sorted({r["image_key"] for r in rows}, key=_canonical))
    for layer in spec["layers"]:
        layer_rows = [r for r in rows if r["layer"] == layer]
        alpha, beta = _two_stage(layer_rows, g, original=True)
        by_image: dict[str, list] = defaultdict(list)
        for row in layer_rows:
            by_image[_canonical(row["image_key"])].append(row)
        image_keys = sorted(by_image)
        rng = np.random.default_rng(_stable_seed(g["bootstrap_seed"], "image_bootstrap", layer))
        bootstrap = []
        # Alpha can be retained when beta is invalid; the two stages are always
        # refitted together in each resample rather than freezing alpha there.
        if alpha["valid"]:
            for repeat in range(g["bootstrap_repeats"]):
                picked = rng.integers(0, len(image_keys), size=len(image_keys))
                sampled = [row for index in picked for row in by_image[image_keys[int(index)]]]
                a, b = _two_stage(sampled, g, original=False)
                bootstrap.append(dict(repeat=repeat, sampled_image_keys=[json.loads(image_keys[int(i)]) for i in picked],
                                      alpha=a, beta=b))
        stable_a = _stability(alpha, [b["alpha"] for b in bootstrap], g)
        stable_b = _stability(beta, [b["beta"] for b in bootstrap], g)
        alpha_ok, beta_ok = stable_a["valid"], stable_a["valid"] and stable_b["valid"]
        layer_output = dict(layer=layer, status="stable" if beta_ok else "beta_unstable" if alpha_ok else "alpha_unstable",
                            alpha=alpha["threshold"] if alpha_ok else None,
                            beta=beta["threshold"] if beta_ok else None,
                            sigma_x=alpha.get("sigma"), sigma_e=beta.get("sigma"),
                            n_observations=len(layer_rows), n_images=len(image_keys),
                            alpha_fit=alpha, beta_fit=beta,
                            alpha_stability=stable_a, beta_stability=stable_b, bootstrap=bootstrap)
        categories: dict[str, int] = defaultdict(int)
        for row in layer_rows:
            categories[classify_head(row, layer_output)["category"]] += 1
        layer_output["category_counts"] = dict(categories)
        output["layers"][str(layer)] = layer_output
    return output


def classify_head(row: dict, calibration_layer: dict) -> dict:
    """Classify one head, preserving boundary ties and uncertainty conservatively."""
    _reject_targets(row)
    if row.get("stats_source") != "baseline":
        raise ValueError("classification requires explicit baseline provenance")
    if "valid" not in row or type(row["valid"]) is not bool:
        raise ValueError("classification requires a boolean valid field")
    alpha, beta = calibration_layer.get("alpha"), calibration_layer.get("beta")
    x_ok, e_ok = _x_valid(row), _entropy_valid(row)
    result = dict(category="unknown", score=None, x_valid=x_ok, entropy_valid=e_ok,
                  reason="alpha_unstable")
    if not _finite(alpha):
        return result
    if not x_ok:
        return dict(result, reason="invalid_visual_statistic")
    if row["x"] >= alpha:
        result.update(category="vision", reason="x_at_or_above_alpha")
    elif not _finite(beta):
        result.update(category="sink_unknown", reason="beta_unstable")
    elif not e_ok:
        result.update(reason="invalid_entropy")
    else:
        result.update(category="sinkG" if row["e"] >= beta else "sinkS", reason="two_stage_thresholds")
    sx, se = calibration_layer.get("sigma_x"), calibration_layer.get("sigma_e")
    if (e_ok and _finite(beta) and _finite(sx) and _finite(se) and sx > 0 and se > 0):
        result["score"] = float(min((alpha - row["x"]) / sx, (beta - row["e"]) / se))
    return result


def select_heads(rows: list[dict], calibration: dict, config: dict) -> list[dict]:
    """Choose one sinkS and seeded matched random heads; unknowns are no-ops."""
    spec = _spec(config)
    rows = _validate_rows(rows, spec, "evaluation")
    _reject_targets(calibration, "calibration")
    if calibration.get("stats_source") != "baseline" or calibration.get("split") != "calibration":
        raise ValueError("selection requires baseline calibration provenance")
    if calibration.get("config_signature") != _signature(spec):
        raise ValueError("calibration/selection configuration mismatch")
    cal_ids = {_canonical(i) for i in calibration["calibration_ids"]}
    cal_images = {_canonical(i) for i in calibration["calibration_image_keys"]}
    if any(_canonical(r["id"]) in cal_ids or _canonical(r["image_key"]) in cal_images for r in rows):
        raise ValueError("calibration and evaluation must have disjoint expressions and images")
    if set(calibration["layers"]) != {str(layer) for layer in spec["layers"]}:
        raise ValueError("calibration layer set mismatch")
    groups: dict[tuple, list] = defaultdict(list)
    for row in rows:
        groups[(_canonical(row["id"]), row["layer"])].append(row)
    output = []
    random_pool = [h for h in range(spec["num_heads"]) if h != spec["shared_head"]]
    for (_, layer), group in groups.items():
        sample, image = group[0]["id"], group[0]["image_key"]
        stats = [dict(row, **classify_head(row, calibration["layers"][str(layer)])) for row in group]
        eligible = [r["head"] for r in stats if r["category"] == "sinkS" and r["head"] != spec["shared_head"]]
        sink, random_heads = None, {str(seed): None for seed in spec["random_seeds"]}
        if eligible:
            rng = np.random.default_rng(_stable_seed(spec["selection_seed"], "sinkS", sample, layer))
            sink = int(rng.choice(eligible))
            for seed in spec["random_seeds"]:
                rng = np.random.default_rng(_stable_seed(seed, "random", sample, layer))
                random_heads[str(seed)] = int(rng.choice(random_pool))
        status = calibration["layers"][str(layer)].get("status", "unknown")
        output.append(dict(id=sample, image_key=image, layer=layer, shared_head=spec["shared_head"],
                           eligible_heads=eligible, k=int(bool(eligible)), sink_head=sink,
                           random_heads=random_heads, head_statistics=stats,
                           reason="selected" if eligible else f"no_eligible_sinkS:{status}"))
    return output
