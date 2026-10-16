import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.svm_experiment import save_sweep_results, sweep_svm
from src.svm import PegasosSVM, RandomFourierFeatures


class SVMSweepTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        self.y = np.array([1, 0, 0, 1])
        self.X_val = self.X * 0.8

    def run_sweep(self, metric, **kwargs):
        return sweep_svm(self.X, self.y, self.X_val, self.y, metric=metric,
                         lambdas=(0.01, 0.1), gammas=(0.5, 1.0), n_epochs=3, **kwargs)

    def test_grid_and_validation_selection(self):
        observed = []
        scores = iter([0.2, 0.9, 0.5, 0.1])

        def metric(y, predictions):
            np.testing.assert_array_equal(y, self.y)
            self.assertTrue(np.isin(predictions, [0, 1]).all())
            observed.append(predictions.copy())
            return next(scores)

        sweep = self.run_sweep(metric)
        self.assertEqual(len(observed), 4)
        self.assertEqual([(r["lambda_"], r["gamma"]) for r in sweep["results"]],
                         [(0.01, 0.5), (0.1, 0.5), (0.01, 1.0), (0.1, 1.0)])
        self.assertEqual(sweep["best_score"], 0.9)
        self.assertEqual(sweep["best_params"]["lambda_"], 0.1)
        self.assertEqual(sweep["best_params"]["gamma"], 0.5)
        np.testing.assert_array_equal(sweep["best_model"].predict(self.X_val), observed[1])
        for row in sweep["results"]:
            self.assertEqual(row["history"][0]["step"], 0)
            self.assertEqual(row["history"][-1]["step"], 12)
            self.assertEqual(row["objective"], row["history"][-1]["objective"])

    def test_lower_is_better_and_stable_ties(self):
        scores = iter([0.2, 0.1, 0.1, 0.5])
        sweep = self.run_sweep(lambda y, predictions: next(scores), greater_is_better=False)
        self.assertEqual(sweep["best_score"], 0.1)
        self.assertEqual(sweep["best_params"]["lambda_"], 0.1)
        self.assertEqual(sweep["best_params"]["gamma"], 0.5)

    def test_metric_can_receive_decision_scores(self):
        observed = []

        def metric(y, scores):
            self.assertEqual(scores.dtype.kind, "f")
            observed.append(scores.copy())
            return float(np.mean((scores >= 0) == y))

        sweep = self.run_sweep(metric, score_input="scores", metric_name="toy_accuracy")
        np.testing.assert_array_equal(sweep["best_model"].decision_function(self.X_val),
                                      observed[next(i for i, r in enumerate(sweep["results"])
                                                    if r["validation_score"] == sweep["best_score"])])
        self.assertEqual(sweep["positive_class"], 1)

    def test_feature_mapping_is_reused_across_regularization_values(self):
        original = RandomFourierFeatures.transform
        calls = []

        def transform(mapping, X):
            calls.append((mapping.gamma, len(X)))
            return original(mapping, X)

        with patch.object(RandomFourierFeatures, "transform", transform):
            sweep = self.run_sweep(lambda y, predictions: float(np.mean(y == predictions)))
        self.assertEqual(calls, [(0.5, 4), (0.5, 4), (1.0, 4), (1.0, 4)])
        standalone = PegasosSVM(**sweep["best_params"]).fit(self.X, self.y)
        np.testing.assert_array_equal(sweep["best_model"].w, standalone.w)
        self.assertEqual(sweep["best_model"].b, standalone.b)
        self.assertEqual(sweep["best_model"].history_, standalone.history_)

    def test_exports_preserve_grid_and_histories(self):
        sweep = self.run_sweep(lambda y, predictions: float(np.mean(y == predictions)))
        with tempfile.TemporaryDirectory() as folder:
            save_sweep_results(sweep, folder)
            with (Path(folder) / "validation.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)
            payload = json.loads((Path(folder) / "sweep.json").read_text())
            self.assertEqual(payload["results"], sweep["results"])
            self.assertEqual(payload["best_params"], sweep["best_params"])
            self.assertNotIn("best_model", payload)

    def test_invalid_inputs_fail_before_metric_evaluation(self):
        def metric(y, predictions):
            self.fail("Invalid inputs should be rejected before scoring")

        for overrides in ({"lambdas": ()}, {"gammas": [0]}, {"score_input": "unknown"},
                          {"n_epochs": None}, {"n_epochs": 0}, {"metric_name": ""},
                          {"y_val": [0, 1, 2, 1]}, {"X_val": [[1.0]]}):
            params = dict(X_train=self.X, y_train=self.y, X_val=self.X_val,
                          y_val=self.y, metric=metric, n_epochs=1)
            params.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                sweep_svm(**params)

    def test_nonfinite_or_nonscalar_metric_is_rejected(self):
        for result in (np.nan, np.inf, (0.2, 0.3), "invalid"):
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.run_sweep(lambda y, predictions: result)


if __name__ == "__main__":
    unittest.main()
