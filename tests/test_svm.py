import unittest

import numpy as np

from src.svm import PegasosSVM, RandomFourierFeatures


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


class RFFPegasosTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        self.y = np.array([1, 0, 0, 1])

    def test_random_features_approximate_rbf_similarities(self):
        Z = np.array([[0.0, 0.0], [2.0, -1.0]])
        for gamma in (0.1, 1.0, 3.0):
            with self.subTest(gamma=gamma):
                mapping = RandomFourierFeatures(gamma=gamma, n_components=20000).fit(self.X)
                expected = np.exp(-gamma * ((self.X[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
                actual = mapping.transform(self.X) @ mapping.transform(Z).T
                np.testing.assert_allclose(actual, expected, atol=0.035)

    def test_transform_matches_formula_and_batch_sizes(self):
        mapping = RandomFourierFeatures(n_components=40, batch_size=2).fit(self.X)
        expected = np.sqrt(2.0 / 40) * np.cos(self.X @ mapping.random_weights_ + mapping.random_offset_)
        np.testing.assert_allclose(mapping.transform(self.X), expected)
        mapping.batch_size = 100
        np.testing.assert_allclose(mapping.transform(self.X), expected)
        self.assertEqual(expected.shape, (4, 40))

    def test_rff_updates_match_linear_training_on_same_features(self):
        for epochs in (None, 5):
            for power in (1.0, 0.75):
                with self.subTest(epochs=epochs, power=power):
                    params = dict(lambda_=0.2, n_iters=31, n_epochs=epochs,
                                  learning_rate_power=power, history_every=7)
                    model = PegasosSVM(kernel="rff", gamma=0.7, n_components=50, **params).fit(self.X, self.y)
                    features = model.feature_map_.transform(self.X)
                    linear = PegasosSVM(**params).fit(features, self.y)
                    np.testing.assert_array_equal(model.w, linear.w)
                    np.testing.assert_allclose(model.decision_function(self.X),
                                               linear.decision_function(features))
                    self.assertEqual(model.history_, linear.history_)
                    self.assertAlmostEqual(model.weight_norm_, linear.weight_norm_)
                    self.assertAlmostEqual(model.margin_width_, linear.margin_width_)
                    self.assertLessEqual(model.weight_norm_, 1 / np.sqrt(0.2) + 1e-10)
                    self.assertEqual(model.n_steps_, 31 if epochs is None else 20)

    def test_rff_objective_and_support_diagnostics(self):
        model = PegasosSVM(kernel="rff", lambda_=0.1, n_epochs=20).fit(self.X, self.y)
        features = model.feature_map_.transform(self.X)
        scores = features @ model.w + model.b
        margins = (2 * self.y - 1) * scores
        hinge = np.maximum(0.0, 1.0 - margins).mean()
        np.testing.assert_allclose(model.decision_function(self.X), scores)
        self.assertAlmostEqual(model.history_[-1]["hinge_loss"], hinge)
        self.assertAlmostEqual(model.history_[-1]["objective"], 0.05 * (model.w @ model.w) + hinge)
        np.testing.assert_array_equal(model.support_, np.flatnonzero(margins <= 1 + 1e-10))
        np.testing.assert_array_equal(model.support_vectors_, self.X[model.support_])
        self.assertEqual(model.w.shape, (500,))
        self.assertFalse(hasattr(model, "alpha_"))
        self.assertFalse(hasattr(model, "X_fit_"))

    def test_rff_predictions_preserve_string_labels(self):
        labels = np.where(self.y == 1, "premium", "standard")
        model = PegasosSVM(kernel="rff", lambda_=0.01, n_epochs=40).fit(self.X, labels)
        np.testing.assert_array_equal(model.predict(self.X), labels)
        self.assertEqual(model.predict(self.X).dtype, labels.dtype)

    def test_rff_learns_xor_on_held_out_points(self):
        rng = np.random.default_rng(4)
        X = rng.uniform(-2.0, 2.0, (200, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        model = PegasosSVM(kernel="rff", lambda_=0.01, n_epochs=50).fit(X[:100], y[:100])
        self.assertGreaterEqual(np.mean(model.predict(X[100:]) == y[100:]), 0.9)
        self.assertLess(model.history_[-1]["objective"], model.history_[0]["objective"])

    def test_batched_prediction_reuses_map_and_ignores_input_mutations(self):
        X = self.X.copy()
        model = PegasosSVM(kernel="rff", lambda_=0.1, n_epochs=10, batch_size=2).fit(X, self.y)
        before = model.decision_function(self.X).copy()
        supports = model.support_vectors_.copy()
        weights = model.feature_map_.random_weights_.copy()
        X[:] = 99.0
        np.testing.assert_array_equal(model.support_vectors_, supports)
        np.testing.assert_allclose(model.decision_function(self.X), before)
        model.batch_size = 1
        np.testing.assert_allclose(model.decision_function(self.X), before)
        np.testing.assert_array_equal(model.feature_map_.random_weights_, weights)

    def test_mapping_and_model_validation(self):
        for params in ({"gamma": 0}, {"gamma": np.nan}, {"n_components": 0},
                       {"n_components": True}, {"batch_size": 0}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                PegasosSVM(kernel="rff", **params).fit(self.X, self.y)
            with self.subTest(mapping=params), self.assertRaises(ValueError):
                RandomFourierFeatures(**params).fit(self.X)
        for mode in ("rbf", "unknown"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                PegasosSVM(kernel=mode).fit(self.X, self.y)
        with self.assertRaises(RuntimeError):
            RandomFourierFeatures().transform(self.X)
        for method in ("decision_function", "predict"):
            with self.subTest(method=method), self.assertRaises(RuntimeError):
                getattr(PegasosSVM(kernel="rff"), method)(self.X)
        mapping = RandomFourierFeatures().fit(self.X)
        model = PegasosSVM(kernel="rff", n_iters=1).fit(self.X, self.y)
        for X in ([[1.0]], [[np.nan, 0.0]], [], [1.0, 2.0]):
            with self.subTest(X=X), self.assertRaises(ValueError):
                mapping.transform(X)
            with self.subTest(prediction=X), self.assertRaises(ValueError):
                model.predict(X)


class ReproducibilityTests(unittest.TestCase):
    def test_seeded_fits_and_refits_match_without_changing_global_rng(self):
        X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        y = np.array([1, 0, 0, 1])
        for kernel in ("linear", "rff"):
            for epochs in (None, 7):
                with self.subTest(kernel=kernel, epochs=epochs):
                    params = dict(kernel=kernel, n_iters=29, n_epochs=epochs, lambda_=0.1,
                                  random_state=17, history_every=5)
                    state_before = np.random.get_state()
                    model = PegasosSVM(**params).fit(X, y)
                    state_after = np.random.get_state()
                    for before, after in zip(state_before, state_after):
                        np.testing.assert_array_equal(before, after)

                    scores = model.decision_function(X).copy()
                    history = [dict(row) for row in model.history_]
                    norm, width = model.weight_norm_, model.margin_width_
                    for fitted in (PegasosSVM(**params).fit(X, y), model.fit(X, y)):
                        np.testing.assert_array_equal(fitted.decision_function(X), scores)
                        self.assertEqual(fitted.history_, history)
                        self.assertEqual(fitted.weight_norm_, norm)
                        self.assertEqual(fitted.margin_width_, width)

    def test_refitting_between_modes_resets_feature_mapping(self):
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = [0, 0, 1, 1]
        model = PegasosSVM(kernel="rff", lambda_=0.1, n_epochs=10).fit(X, y)
        kernel_scores = model.decision_function(X).copy()
        model.kernel = "linear"
        model.fit(X, y)

        self.assertIsNone(model.feature_map_)
        np.testing.assert_allclose(model.decision_function(X), X @ model.w + model.b)
        model.kernel = "rff"
        model.fit(X, y)
        self.assertIsNotNone(model.feature_map_)
        np.testing.assert_array_equal(model.decision_function(X), kernel_scores)


if __name__ == "__main__":
    unittest.main()
