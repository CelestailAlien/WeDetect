"""Offline tests for the label-free GMM and single-head intervention schedule."""
import copy
import json
import math
import unittest

import numpy as np

from ref_sads_stats import (DEFAULT_GMM, _em, _spec, _signature, _stationary_points,
                            calibrate_heads, classify_head, select_heads)


def configuration():
    return dict(layers=[28], num_heads=32, shared_head=0, selection_seed=20260923,
                random_seeds=[11, 29, 47],
                gmm=dict(init_seeds=[0, 1], min_observations=80, min_images=8,
                         bootstrap_repeats=4, bootstrap_min_success=3,
                         bootstrap_iqr_fraction=.25))


def fixture(split="calibration", images=12, seed=3):
    rng = np.random.default_rng(seed)
    records = []
    for image in range(images):
        for head in range(32):
            sink = head < 16
            x = (.02 if sink else .16) + rng.normal(0, .002)
            e = (.15 if head < 8 else .80) + rng.normal(0, .015)
            records.append(dict(id=f"{split}-{image}", image_key=f"{split}-image-{image}",
                                split=split, layer=28, head=head, x=x,
                                H=e * math.log(20), e=e, valid=True, stats_source="baseline"))
    return records


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = configuration()
        cls.records = fixture()
        cls.calibration = calibrate_heads(cls.records, cls.config)

    def test_separated_two_stage_fit_and_complete_audit(self):
        result = self.calibration["layers"]["28"]
        self.assertEqual(result["status"], "stable")
        self.assertTrue(.03 < result["alpha"] < .15)
        self.assertTrue(.20 < result["beta"] < .75)
        self.assertEqual(result["n_images"], 12)
        self.assertEqual(len(result["bootstrap"]), 4)
        self.assertEqual(result["alpha_stability"]["successes"], 4)
        self.assertEqual(result["beta_stability"]["successes"], 4)
        self.assertEqual(result["category_counts"], dict(sinkS=96, sinkG=96, vision=192))
        for bootstrap in result["bootstrap"]:
            self.assertEqual(len(bootstrap["sampled_image_keys"]), 12)
            self.assertIn("alpha", bootstrap)
            self.assertIn("beta", bootstrap)
        json.dumps(self.calibration, allow_nan=False)

    def test_row_order_does_not_change_fitting(self):
        reverse = calibrate_heads(list(reversed(self.records)), self.config)
        self.assertEqual(reverse, self.calibration)

    def test_bootstrap_does_not_reapply_unique_image_minimum(self):
        config = copy.deepcopy(self.config)
        config["gmm"]["min_images"] = 12
        layer = calibrate_heads(self.records, config)["layers"]["28"]
        self.assertEqual(layer["status"], "stable")
        self.assertTrue(all(b["alpha"]["n_images"] < 12 for b in layer["bootstrap"]))
        self.assertEqual(layer["alpha_stability"]["successes"], 4)

    def test_single_gaussian_and_two_component_without_two_modes(self):
        x = np.random.default_rng(8).normal(size=1000)
        from ref_sads_stats import _fit_threshold
        result = _fit_threshold(x.tolist(), [f"image-{i // 25}" for i in range(1000)],
                                dict(DEFAULT_GMM), True)
        self.assertFalse(result["valid"])
        shape = _stationary_points(dict(weights=[.5, .5], means=[-.1, .1], variances=[1., 1.]))
        self.assertFalse(shape["valid"])

    def test_extremely_separated_modes_are_not_lost_to_underflow(self):
        shape = _stationary_points(dict(weights=[.5, .5], means=[-1., 1.], variances=[1e-6, 1e-6]))
        self.assertTrue(shape["valid"])
        self.assertAlmostEqual(shape["valley"], 0.)

    def test_constant_x_falls_back_without_synthetic_quantiles(self):
        rows = copy.deepcopy(self.records)
        for row in rows:
            row["x"] = .1
        layer = calibrate_heads(rows, self.config)["layers"]["28"]
        self.assertEqual(layer["status"], "alpha_unstable")
        self.assertIsNone(layer["alpha"])
        self.assertIsNone(layer["beta"])
        self.assertEqual(layer["bootstrap"], [])

    def test_invalid_entropy_does_not_force_sinkS(self):
        rows = copy.deepcopy(self.records)
        for row in rows:
            row.update(H=None, e=None, entropy_valid=False, x_valid=True)
        layer = calibrate_heads(rows, self.config)["layers"]["28"]
        self.assertEqual(layer["status"], "beta_unstable")
        self.assertIsNotNone(layer["alpha"])
        self.assertIsNone(layer["beta"])
        self.assertEqual(layer["category_counts"], dict(sink_unknown=192, vision=192))

    def test_nonconvergence_is_not_accepted(self):
        fit = _em(np.linspace(-1, 1, 100), 2, 0, dict(DEFAULT_GMM, max_iter=1))
        self.assertFalse(fit["converged"])

    def test_gt_and_provenance_rejected(self):
        for change in (dict(gt_box=[1, 2, 3, 4]), dict(extra={"answer_boxes": []}),
                       dict(groundTruth=[1, 2, 3, 4]), dict(candidateIoU=.8),
                       dict(stats_source="intervention"), dict(gate=.5), dict(arm="sink_hard"),
                       dict(split="evaluation")):
            rows = copy.deepcopy(self.records)
            rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                calibrate_heads(rows, self.config)
        rows = copy.deepcopy(self.records)
        del rows[0]["stats_source"]
        with self.assertRaises(ValueError):
            calibrate_heads(rows, self.config)

    def test_duplicate_missing_and_wrong_heads_rejected(self):
        for rows in (self.records[:-1], self.records + [self.records[0]],
                     [dict(r, head=32) if i == 0 else r for i, r in enumerate(self.records)]):
            with self.assertRaises(ValueError):
                calibrate_heads(rows, self.config)
        changed = dict(self.config, layers=[28, 32])
        with self.assertRaises(ValueError):
            calibrate_heads(self.records, changed)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration()
        self.layer = dict(status="stable", alpha=.08, beta=.45, sigma_x=.07, sigma_e=.30)
        self.calibration = dict(stats_source="baseline", split="calibration", layers={"28": self.layer},
                                config_signature=_signature(_spec(self.config)),
                                calibration_ids=["cal-0"], calibration_image_keys=["cal-image-0"])
        self.rows = fixture("evaluation", images=8)

    def test_boundary_ties_are_retained_and_score_agrees(self):
        row = self.rows[0]
        self.assertEqual(classify_head(dict(row, x=.08), self.layer)["category"], "vision")
        self.assertEqual(classify_head(dict(row, e=.45), self.layer)["category"], "sinkG")
        self.assertGreater(classify_head(row, self.layer)["score"], 0)
        self.assertLessEqual(classify_head(dict(row, x=.08), self.layer)["score"], 0)
        self.assertEqual(classify_head(dict(row, valid=False), self.layer)["category"], "unknown")
        self.assertIsNone(classify_head(dict(row, e=None), self.layer)["score"])

    def test_reproducible_uniform_pool_matching_and_shared_head(self):
        selected = select_heads(self.rows, self.calibration, self.config)
        reversed_selected = select_heads(list(reversed(self.rows)), self.calibration, self.config)
        self.assertEqual(selected, reversed_selected)
        for entry in selected:
            self.assertEqual(entry["eligible_heads"], list(range(1, 8)))
            self.assertEqual(entry["k"], 1)
            self.assertIn(entry["sink_head"], entry["eligible_heads"])
            self.assertEqual(len(entry["head_statistics"]), 32)
            for head in entry["random_heads"].values():
                self.assertIn(head, range(1, 32))
        json.dumps(selected, allow_nan=False)

    def test_no_eligible_sinkS_keeps_every_arm_noop(self):
        for row in self.rows:
            row["e"] = .8
        # Only h0 is sinkS, and must remain untouched.
        self.rows[0]["e"] = .1
        for selected in select_heads(self.rows, self.calibration, self.config):
            self.assertEqual(selected["k"], 0)
            self.assertIsNone(selected["sink_head"])
            self.assertTrue(all(x is None for x in selected["random_heads"].values()))

    def test_random_may_equal_selected_sink_or_other_random_seeds(self):
        # With two heads, excluding h0 leaves one legal head. Rejection sampling
        # or excluding selected sinkS would make this correctly configured case fail.
        config = dict(self.config, num_heads=2)
        calibration = copy.deepcopy(self.calibration)
        calibration["config_signature"] = _signature(_spec(config))
        rows = [r for r in self.rows if r["head"] < 2]
        for entry in select_heads(rows, calibration, config):
            self.assertEqual(entry["sink_head"], 1)
            self.assertEqual(entry["random_heads"], {"11": 1, "29": 1, "47": 1})

    def test_unknown_layer_is_all_noop(self):
        self.layer.update(alpha=None, beta=None, status="alpha_unstable")
        selected = select_heads(self.rows, self.calibration, self.config)
        self.assertTrue(all(r["k"] == 0 for r in selected))
        self.assertTrue(all(h["category"] == "unknown" for r in selected for h in r["head_statistics"]))

    def test_data_or_config_leakage_rejected(self):
        for key, value in (("id", "cal-0"), ("image_key", "cal-image-0")):
            rows = [dict(r, **{key: value}) if r["id"] == "evaluation-0" else dict(r) for r in self.rows]
            with self.subTest(key=key), self.assertRaises(ValueError):
                select_heads(rows, self.calibration, self.config)
        with self.assertRaises(ValueError):
            select_heads(self.rows, self.calibration, dict(self.config, selection_seed=4))
        with self.assertRaises(ValueError):
            select_heads(fixture("calibration"), self.calibration, self.config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
