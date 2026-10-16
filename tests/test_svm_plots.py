import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.svm_experiment import save_sweep_results, sweep_svm
from src.svm_plots import plot_sweep_results


class SVMPlotTests(unittest.TestCase):
    def setUp(self):
        X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        y = np.array([1, 0, 0, 1])
        self.sweep = sweep_svm(X, y, X * 0.8, y, lambdas=(0.01, 0.1), gammas=(0.5, 1.0),
                               n_epochs=4, history_every=4, n_components=30,
                               metric=lambda truth, prediction: float(np.mean(truth == prediction)),
                               metric_name="accuracy")

    def test_exports_from_saved_results_without_a_model(self):
        with tempfile.TemporaryDirectory() as folder:
            save_sweep_results(self.sweep, folder)
            saved = json.loads((Path(folder) / "sweep.json").read_text())
            paths = plot_sweep_results(saved, folder, data_label="Synthetic test fixture",
                                       score_as_percent=True)
            self.assertEqual(set(paths), {"svm_convergence", "svm_validation_sweeps", "svm_margin_validation"})
            for formats in paths.values():
                self.assertTrue(formats["png"].read_bytes().startswith(b"\x89PNG"))
                self.assertTrue(formats["pdf"].read_bytes().startswith(b"%PDF"))

    def test_plotted_values_match_the_sweep(self):
        from matplotlib.figure import Figure

        captured = {}
        original = Figure.savefig

        def save(fig, path, **kwargs):
            if Path(path).suffix == ".png":
                captured[Path(path).stem] = {
                    "lines": [[(line.get_xdata().copy(), line.get_ydata().copy())
                               for line in ax.lines] for ax in fig.axes],
                    "points": [[collection.get_offsets().copy() for collection in ax.collections]
                                for ax in fig.axes],
                }
            return original(fig, path, **kwargs)

        with tempfile.TemporaryDirectory() as folder, patch.object(Figure, "savefig", save):
            plot_sweep_results(self.sweep, folder, data_label="Synthetic test fixture")
        selected = next(row for row in self.sweep["results"]
                        if row["lambda_"] == self.sweep["best_params"]["lambda_"]
                        and row["gamma"] == self.sweep["best_params"]["gamma"])
        trace = captured["svm_convergence"]["lines"][0][0]
        np.testing.assert_array_equal(trace[0], [row["step"] for row in selected["history"]])
        np.testing.assert_array_equal(trace[1], [row["objective"] for row in selected["history"]])
        regularization_lines = captured["svm_validation_sweeps"]["lines"][0]
        self.assertEqual(len(regularization_lines), 2)
        for gamma, line in zip((0.5, 1.0), regularization_lines):
            expected = sorted((row for row in self.sweep["results"] if row["gamma"] == gamma),
                              key=lambda row: row["lambda_"])
            np.testing.assert_array_equal(line[0], [row["lambda_"] for row in expected])
            np.testing.assert_array_equal(line[1], [row["validation_score"] for row in expected])
        scatter = captured["svm_margin_validation"]["points"][0]
        for row, point in zip(self.sweep["results"], scatter):
            np.testing.assert_allclose(point, [[row["margin_width"], row["validation_score"]]])

    def test_infinite_widths_and_single_configuration_are_supported(self):
        sweep = {key: value for key, value in self.sweep.items() if key != "best_model"}
        sweep = copy.deepcopy(sweep)
        selected = next(row for row in sweep["results"]
                        if row["lambda_"] == sweep["best_params"]["lambda_"]
                        and row["gamma"] == sweep["best_params"]["gamma"])
        sweep["results"] = [selected]
        selected["margin_width"] = float("inf")
        with tempfile.TemporaryDirectory() as folder:
            paths = plot_sweep_results(sweep, folder, data_label="Synthetic edge case")
            self.assertEqual(len(paths), 3)

    def test_invalid_plot_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                plot_sweep_results(self.sweep, folder, data_label="")
            for change in ("empty", "nan_score", "negative_lambda", "duplicate"):
                sweep = {key: value for key, value in self.sweep.items() if key != "best_model"}
                sweep = copy.deepcopy(sweep)
                if change == "empty":
                    sweep["results"] = []
                elif change == "nan_score":
                    sweep["results"][0]["validation_score"] = float("nan")
                elif change == "negative_lambda":
                    sweep["results"][0]["lambda_"] = -1
                else:
                    sweep["results"].append(sweep["results"][0])
                with self.subTest(change=change), self.assertRaises(ValueError):
                    plot_sweep_results(sweep, folder, data_label="Invalid fixture")


if __name__ == "__main__":
    unittest.main()
