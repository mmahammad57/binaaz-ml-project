from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,
        n_iters: int = 20_000,
        random_state: int = 42,
        *,
        n_epochs: int | None = None,
        learning_rate_power: float = 1.0,
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.learning_rate_power = learning_rate_power
        self.w = None
        self.b = 0.0

    def _validate_params(self):
        if (isinstance(self.lambda_, (bool, np.bool_))
                or not isinstance(self.lambda_, (int, float, np.integer, np.floating))
                or not np.isfinite(self.lambda_) or self.lambda_ <= 0):
            raise ValueError("lambda_ must be a finite positive number")
        if (isinstance(self.n_iters, (bool, np.bool_))
                or not isinstance(self.n_iters, (int, np.integer))
                or self.n_iters < 1):
            raise ValueError("n_iters must be a positive integer")
        if self.n_epochs is not None:
            if (isinstance(self.n_epochs, (bool, np.bool_))
                    or not isinstance(self.n_epochs, (int, np.integer))
                    or self.n_epochs < 1):
                raise ValueError("n_epochs must be a positive integer or None")
        if (isinstance(self.learning_rate_power, (bool, np.bool_))
                or not isinstance(self.learning_rate_power, (int, float, np.integer, np.floating))
                or not np.isfinite(self.learning_rate_power)
                or not 0.5 < self.learning_rate_power <= 1.0):
            raise ValueError("learning_rate_power must be in (0.5, 1.0]")

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty two-dimensional feature matrix")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        return X

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

    def fit(self, X, y):
        self._validate_params()
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("X and y must have matching sample counts")
        y_pm1 = self._to_pm1(y)
        self.n_features_in_ = X.shape[1]
        self.w = np.zeros(self.n_features_in_, dtype=float)
        self.b = 0.0
        rng = np.random.default_rng(self.random_state)
        radius = 1.0 / np.sqrt(self.lambda_)
        n_steps = self.n_iters if self.n_epochs is None else self.n_epochs * X.shape[0]

        for t in range(1, n_steps + 1):
            if self.n_epochs is None:
                i = rng.integers(X.shape[0])
            else:
                offset = (t - 1) % X.shape[0]
                if offset == 0:
                    order = rng.permutation(X.shape[0])
                i = order[offset]
            margin = y_pm1[i] * (X[i] @ self.w + self.b)
            eta = 1.0 / (self.lambda_ * t ** self.learning_rate_power)
            self.w *= 1.0 - eta * self.lambda_

            if margin < 1.0:
                self.w += eta * y_pm1[i] * X[i]
                self.b += eta * y_pm1[i]

            norm = np.linalg.norm(self.w)
            if norm > radius:
                self.w *= radius / norm

        self.n_steps_ = n_steps
        return self

    def decision_function(self, X):
        if self.w is None:
            raise RuntimeError("Call fit before prediction")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features as training data")
        return X @ self.w + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0.0).astype(int)]
