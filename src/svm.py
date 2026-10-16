from __future__ import annotations

import numpy as np


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be a positive integer")


def _positive_number(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive number")


def _feature_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a nonempty two-dimensional feature matrix")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    return X


class RandomFourierFeatures:
    def __init__(self, gamma=1.0, n_components=500, random_state=42, batch_size=2048):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.batch_size = batch_size

    def fit(self, X):
        _positive_number(self.gamma, "gamma")
        _positive_integer(self.n_components, "n_components")
        _positive_integer(self.batch_size, "batch_size")
        X = _feature_matrix(X)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.random_weights_ = rng.normal(0.0, np.sqrt(2.0 * self.gamma),
                                          (X.shape[1], self.n_components))
        self.random_offset_ = rng.uniform(0.0, 2.0 * np.pi, self.n_components)
        return self

    def transform(self, X):
        if not hasattr(self, "random_weights_"):
            raise RuntimeError("Call fit before transforming features")
        X = _feature_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features as training data")
        Z = np.empty((len(X), self.n_components), dtype=float)
        for start in range(0, len(X), self.batch_size):
            block = Z[start:start + self.batch_size]
            np.matmul(X[start:start + self.batch_size], self.random_weights_, out=block)
            block += self.random_offset_
            np.cos(block, out=block)
            block *= np.sqrt(2.0 / self.n_components)
        return Z


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,
        n_iters: int = 20_000,
        random_state: int = 42,
        *,
        n_epochs: int | None = None,
        learning_rate_power: float = 1.0,
        history_every: int = 1000,
        gamma: float = 1.0,
        kernel: str = "linear",
        n_components: int = 500,
        batch_size: int = 2048,
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.learning_rate_power = learning_rate_power
        self.history_every = history_every
        self.gamma = gamma
        self.kernel = kernel
        self.n_components = n_components
        self.batch_size = batch_size
        self.w = None
        self.b = 0.0

    def _validate_params(self):
        if self.kernel not in ("linear", "rff"):
            raise ValueError("kernel must be 'linear' or 'rff'")
        _positive_number(self.lambda_, "lambda_")
        _positive_number(self.gamma, "gamma")
        for name in ("n_iters", "history_every", "n_components", "batch_size"):
            _positive_integer(getattr(self, name), name)
        if self.n_epochs is not None:
            _positive_integer(self.n_epochs, "n_epochs")
        _positive_number(self.learning_rate_power, "learning_rate_power")
        if not 0.5 < self.learning_rate_power <= 1.0:
            raise ValueError("learning_rate_power must be in (0.5, 1.0]")

    _validate_X = staticmethod(_feature_matrix)

    def _to_pm1(self, y):
        y = np.asarray(y)
        if y.ndim != 1 or y.size == 0:
            raise ValueError("y must be a nonempty one-dimensional label array")
        if y.dtype.kind not in "biufUS":
            raise ValueError("y must contain numeric or string labels")
        if y.dtype.kind in "f" and not np.isfinite(y).all():
            raise ValueError("y must contain only finite labels")
        classes, inverse = np.unique(y, return_inverse=True)
        if classes.size != 2:
            raise ValueError("y must contain exactly two classes")
        self.classes_ = classes
        return 2.0 * inverse - 1.0

    def _record_objective(self, X, y_pm1, step):
        margins = y_pm1 * (X @ self.w + self.b)
        hinge_loss = np.maximum(0.0, 1.0 - margins).mean()
        regularization = self.lambda_ / 2.0 * (self.w @ self.w)
        self.history_.append({
            "step": step,
            "hinge_loss": float(hinge_loss),
            "objective": float(regularization + hinge_loss),
        })

    def fit(self, X, y):
        self._validate_params()
        X = self._validate_X(X)
        if self.kernel == "rff":
            mapping = RandomFourierFeatures(self.gamma, self.n_components,
                                            self.random_state, self.batch_size).fit(X)
            Z = mapping.transform(X)
        else:
            mapping, Z = None, X
        return self._fit_features(X, y, Z, mapping)

    def _fit_features(self, X, y, Z, mapping=None):
        self._validate_params()
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("X and y must have matching sample counts")
        y_pm1 = self._to_pm1(y)
        self.n_features_in_ = X.shape[1]
        self.feature_map_ = mapping
        self.w = np.zeros(Z.shape[1], dtype=float)
        self.b = 0.0
        self.history_ = []
        self._record_objective(Z, y_pm1, 0)
        rng = np.random.default_rng(self.random_state)
        radius = 1.0 / np.sqrt(self.lambda_)
        n_steps = self.n_iters if self.n_epochs is None else self.n_epochs * len(Z)

        for t in range(1, n_steps + 1):
            if self.n_epochs is None:
                i = rng.integers(len(Z))
            else:
                offset = (t - 1) % len(Z)
                if offset == 0:
                    order = rng.permutation(len(Z))
                i = order[offset]
            margin = y_pm1[i] * (Z[i] @ self.w + self.b)
            eta = 1.0 / (self.lambda_ * t ** self.learning_rate_power)
            self.w *= 1.0 - eta * self.lambda_

            if margin < 1.0:
                self.w += eta * y_pm1[i] * Z[i]
                self.b += eta * y_pm1[i]

            norm = np.linalg.norm(self.w)
            if norm > radius:
                self.w *= radius / norm

            if t % self.history_every == 0 or t == n_steps:
                self._record_objective(Z, y_pm1, t)

        self.n_steps_ = n_steps
        self.weight_norm_ = float(np.linalg.norm(self.w))
        self.margin_width_ = 2.0 / self.weight_norm_ if self.weight_norm_ > 0.0 else float("inf")
        margins = y_pm1 * (Z @ self.w + self.b)
        self.support_ = np.flatnonzero(margins <= 1.0 + 1e-10)
        self.support_vectors_ = X[self.support_].copy()
        return self

    def decision_function(self, X):
        if self.w is None:
            raise RuntimeError("Call fit before prediction")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features as training data")
        if self.feature_map_ is None:
            return X @ self.w + self.b
        scores = np.empty(len(X), dtype=float)
        for start in range(0, len(X), self.batch_size):
            block = X[start:start + self.batch_size]
            scores[start:start + len(block)] = self.feature_map_.transform(block) @ self.w + self.b
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0.0).astype(int)]
