import unittest

import numpy as np

from src.svm import PegasosSVM


class LinearPegasosTests(unittest.TestCase):
    def test_first_update_projects_weights_and_keeps_bias_unregularized(self):
        X = np.array([[3.0, 4.0], [-3.0, -4.0]])
        model = PegasosSVM(lambda_=1.0, n_iters=1, random_state=42)

        self.assertIs(model.fit(X, [0, 1]), model)
        np.testing.assert_allclose(model.w, [-0.6, -0.8])
        self.assertEqual(model.b, -1.0)
        self.assertAlmostEqual(model.weight_norm_, 1.0)

    def test_margin_equal_to_one_only_shrinks_weights(self):
        model = PegasosSVM(lambda_=1.0, n_iters=2, random_state=42)
        model.fit([[-2.0], [2.0]], [0, 1])

        np.testing.assert_allclose(model.w, [0.5])
        self.assertEqual(model.b, -1.0)

    def test_second_violation_updates_bias_with_decaying_learning_rate(self):
        model = PegasosSVM(lambda_=1.0, n_iters=2, random_state=42)
        model.fit([[0.0], [0.0]], [0, 1])

        np.testing.assert_array_equal(model.w, [0.0])
        self.assertEqual(model.b, -0.5)

    def test_decision_scores_and_original_labels(self):
        for labels in ([0, 1], [-1, 1], ["premium", "standard"]):
            with self.subTest(labels=labels):
                model = PegasosSVM(lambda_=1.0, n_iters=1, random_state=42)
                model.fit([[-2.0], [2.0]], labels)

                scores = model.decision_function([[-1.0], [3.0]])
                np.testing.assert_allclose(scores, [-2.0, 2.0])
                np.testing.assert_array_equal(model.predict([[-1.0], [3.0]]), labels)
                self.assertEqual(scores.shape, (2,))
                self.assertEqual(model.predict([[-1.0]]).dtype, np.asarray(labels).dtype)

    def test_zero_score_selects_positive_class(self):
        model = PegasosSVM(lambda_=1.0, n_iters=1, random_state=42)
        model.fit([[-2.0], [2.0]], [0, 1])

        np.testing.assert_allclose(model.decision_function([[1.0]]), [0.0])
        np.testing.assert_array_equal(model.predict([[1.0]]), [1])

    def test_objective_checkpoints_include_initial_and_final_values(self):
        model = PegasosSVM(lambda_=1.0, n_iters=2, history_every=1, random_state=42)
        model.fit([[-2.0], [2.0]], [0, 1])

        self.assertEqual([row["step"] for row in model.history_], [0, 1, 2])
        np.testing.assert_allclose([row["hinge_loss"] for row in model.history_], [1.0, 0.0, 0.5])
        np.testing.assert_allclose([row["objective"] for row in model.history_], [1.0, 0.5, 0.625])

        model.history_every = 3
        model.fit([[-2.0], [2.0]], [0, 1])
        self.assertEqual([row["step"] for row in model.history_], [0, 2])

    def test_margin_width_is_distance_between_margin_planes(self):
        X = np.array([[-3.0, -1.0], [-2.0, 0.0], [2.0, 0.0], [3.0, 1.0]])
        model = PegasosSVM(lambda_=0.1, n_epochs=20).fit(X, [0, 0, 1, 1])
        norm = np.linalg.norm(model.w)
        center = -model.b * model.w / (model.w @ model.w)
        displacement = model.w / (norm * norm)
        boundary_points = np.array([center - displacement, center + displacement])

        self.assertAlmostEqual(model.weight_norm_, norm)
        self.assertAlmostEqual(model.margin_width_, np.linalg.norm(2 * displacement))
        np.testing.assert_allclose(model.decision_function(boundary_points), [-1.0, 1.0])
        self.assertLessEqual(norm, 1 / np.sqrt(model.lambda_) + 1e-10)

    def test_support_points_include_boundary_and_violations(self):
        X = np.array([[-3.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
        model = PegasosSVM(lambda_=1.0, n_iters=1, random_state=42)
        model.fit(X, [0, 0, 1, 1, 1, 1])

        np.testing.assert_array_equal(model.support_, [2, 3, 4])
        np.testing.assert_array_equal(model.support_vectors_, [[0.0], [1.0], [2.0]])
        X[:] = 99.0
        np.testing.assert_array_equal(model.support_vectors_, [[0.0], [1.0], [2.0]])

    def test_zero_weight_margin_and_refit_diagnostics(self):
        model = PegasosSVM(lambda_=1.0, n_iters=1, random_state=42)
        model.fit([[0.0], [0.0]], [0, 1])

        self.assertEqual(model.weight_norm_, 0.0)
        self.assertTrue(np.isinf(model.margin_width_))
        np.testing.assert_array_equal(model.support_, [0, 1])

        model.fit([[-2.0], [2.0]], [0, 1])
        self.assertAlmostEqual(model.weight_norm_, 1.0)
        self.assertAlmostEqual(model.margin_width_, 2.0)
        np.testing.assert_array_equal(model.support_, [1])

    def test_epochs_override_update_count(self):
        model = PegasosSVM(n_iters=1, n_epochs=3)
        model.fit([[-2.0], [2.0]], [0, 1])

        self.assertEqual(model.n_steps_, 6)
        self.assertEqual(model.history_[-1]["step"], 6)

    def test_learns_linear_boundary_on_held_out_points(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(400, 2))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        model = PegasosSVM(lambda_=0.05, n_epochs=60).fit(X[:240], y[:240])

        self.assertGreater(np.mean(model.predict(X[240:]) == y[240:]), 0.95)
        self.assertLess(model.history_[-1]["objective"], model.history_[0]["objective"])

    def test_prediction_rejects_unfitted_models_and_invalid_features(self):
        for method in ("predict", "decision_function"):
            with self.subTest(method=method), self.assertRaises(RuntimeError):
                getattr(PegasosSVM(), method)([[1.0]])

        model = PegasosSVM(n_iters=1).fit([[-2.0], [2.0]], [0, 1])
        for X in ([[1.0, 2.0]], [[np.nan]], [[np.inf]], [1.0], []):
            for method in ("predict", "decision_function"):
                with self.subTest(X=X, method=method), self.assertRaises(ValueError):
                    getattr(model, method)(X)


if __name__ == "__main__":
    unittest.main()
