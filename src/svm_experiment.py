from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from .svm import PegasosSVM, RandomFourierFeatures


def sweep_svm(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    metric,
    lambdas=(0.001, 0.01, 0.1),
    gammas=(0.1, 1.0, 10.0),
    n_epochs=20,
    learning_rate_power=1.0,
    history_every=50_000,
    n_components=500,
    batch_size=2048,
    random_state=42,
    metric_name="validation_score",
    score_input="labels",
    greater_is_better=True,
):
    if not callable(metric):
        raise ValueError("metric must be callable and return a scalar")
    if score_input not in ("labels", "scores"):
        raise ValueError("score_input must be 'labels' or 'scores'")
    if not isinstance(metric_name, str) or not metric_name.strip():
        raise ValueError("metric_name must be a nonempty string")
    if not isinstance(greater_is_better, (bool, np.bool_)):
        raise ValueError("greater_is_better must be a boolean")
    if n_epochs is None:
        raise ValueError("n_epochs must be a positive integer")

    X_train = PegasosSVM._validate_X(X_train)
    X_val = PegasosSVM._validate_X(X_val)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Training and validation features must have matching columns")
    if y_train.ndim != 1 or len(y_train) != len(X_train):
        raise ValueError("Training features and labels must have matching rows")
    if y_val.ndim != 1 or len(y_val) != len(X_val):
        raise ValueError("Validation features and labels must have matching rows")
    encoder = PegasosSVM()
    encoder._to_pm1(y_train)
    if not np.isin(y_val, encoder.classes_).all():
        raise ValueError("Validation labels must belong to the training classes")

    lambdas, gammas = tuple(lambdas), tuple(gammas)
    if not lambdas or not gammas:
        raise ValueError("lambdas and gammas must be nonempty")
    configs = []
    for gamma, lambda_ in product(gammas, lambdas):
        params = dict(kernel="rff", lambda_=lambda_, gamma=gamma, n_epochs=n_epochs,
                      learning_rate_power=learning_rate_power,
                      history_every=history_every, random_state=random_state,
                      n_components=n_components, batch_size=batch_size)
        PegasosSVM(**params)._validate_params()
        configs.append(params)

    results = []
    best_model = None
    best_params = None
    best_score = None
    mapping = None
    Z_train = Z_val = None
    for params in configs:
        if mapping is None or mapping.gamma != params["gamma"]:
            Z_train = Z_val = None
            mapping = RandomFourierFeatures(params["gamma"], n_components,
                                            random_state, batch_size).fit(X_train)
            Z_train = mapping.transform(X_train)
            Z_val = mapping.transform(X_val)
        model = PegasosSVM(**params)._fit_features(X_train, y_train, Z_train, mapping)
        scores = Z_val @ model.w + model.b
        output = model.classes_[(scores >= 0.0).astype(int)] if score_input == "labels" else scores
        score = metric(y_val.copy(), output)
        if not np.isscalar(score) or not isinstance(score, (int, float, np.integer, np.floating)):
            raise ValueError("metric must return a finite scalar number")
        score = float(score)
        if not np.isfinite(score):
            raise ValueError("metric must return a finite scalar number")
        results.append({
            "lambda_": float(params["lambda_"]),
            "gamma": float(params["gamma"]),
            "n_components": int(n_components),
            "validation_score": score,
            "weight_norm": model.weight_norm_,
            "margin_width": model.margin_width_,
            "support_count": len(model.support_),
            "objective": model.history_[-1]["objective"],
            "history": [dict(row) for row in model.history_],
        })
        if (best_model is None
                or (greater_is_better and score > best_score)
                or (not greater_is_better and score < best_score)):
            best_model = model
            best_params = dict(params)
            best_score = score

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_score": best_score,
        "metric_name": metric_name,
        "score_input": score_input,
        "positive_class": encoder.classes_[1].item(),
        "greater_is_better": bool(greater_is_better),
        "n_train": len(X_train),
        "n_validation": len(X_val),
        "results": results,
    }


def save_sweep_results(sweep, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [{key: value for key, value in row.items() if key != "history"}
            for row in sweep["results"]]
    with (output_dir / "validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {key: value for key, value in sweep.items() if key != "best_model"}
    with (output_dir / "sweep.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=lambda value: value.item())
        handle.write("\n")
