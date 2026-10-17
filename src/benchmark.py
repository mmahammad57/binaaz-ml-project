"""
benchmark.py — Problem 5: our models vs scikit-learn, on identical data.

Run from the repo root:
    python -m src.benchmark            # full run
    python -m src.benchmark --quick    # smoke test: 5k-row CV, 3 folds, 5k-row test fits

Stages
  1. data         load -> clean -> make_features -> 6-array train/val/test split
  2. CV           stratified 20k-row subsample of train+val, 5-fold StratifiedKFold;
                  tier threshold and standardization refit inside every fold
  3. test         every model fit once on the training split, scored once on test
  4. trees        our trees vs sklearn's trees on test (structure, agreement, importances)
  5. scaling      fit time at n = 2k..16k, log-log figure
  6. save         tables (CSV + LaTeX), test predictions, metadata

Test data is never used to choose anything: thresholds, scalers, subsamples
and hyperparameters all come from training rows or from FINAL_CONFIGS.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import data_prep
from . import evaluate as ev
from .data_prep import SEED
from .decision_tree import DecisionTree
from .svm import PegasosSVM

# --- Final configurations ---------------------------------------------------
# No tuned values exist in the repo yet (checked 2026-10-17): docs/SVM.md says
# "final SVM settings have not been established yet", and B's
# src/tree_experiment.py (origin/feature/decision-tree) only runs on synthetic
# data. Values below are placeholders unless a source is cited.
FINAL_CONFIGS = {
    # Task A: regression on log(price).
    "tree_task_a": {
        "max_depth": 10,            # TODO: confirm with B
        "min_samples_split": 10,    # TODO: confirm with B
        "min_samples_leaf": 5,      # TODO: confirm with B
    },
    # Task B: price tier. Both gini and entropy are benchmarked with these.
    "tree_task_b": {
        "max_depth": 10,            # TODO: confirm with B
        "min_samples_split": 10,    # TODO: confirm with B
        "min_samples_leaf": 5,      # TODO: confirm with B
    },
    "svm": {
        # docs/SVM.md: "The final nonlinear model uses random Fourier features
        # (RFF) with linear Pegasos".
        "kernel": "rff",
        # docs/SVM.md: PegasosSVM default and the documented sweep setting.
        "n_components": 500,
        # docs/SVM.md fit/sweep examples use n_epochs=20.
        "n_epochs": 20,
        "lambda_": 1e-3,            # TODO: confirm with C (lowest value of C's sweep grid)
        # "scale" = 1 / n_features: sklearn's default RBF width for
        # standardized features, so it uses no data beyond the column count.
        "gamma": "scale",           # TODO: confirm with C
    },
}
MAX_TREE_DEPTH = 12        # hard cap on max_depth (our tree is O(n^1.6) per level)
SVC_MAX_ROWS = 10_000      # exact-kernel SVC is fit on at most this many rows

ROOT = Path(__file__).resolve().parents[1]


# --- Small helpers ----------------------------------------------------------
@contextlib.contextmanager
def _stage(title):
    print(f"\n=== {title}")
    start = time.perf_counter()
    yield
    print(f"--- {title}: {time.perf_counter() - start:.1f}s")


def _quiet():
    """Silence data_prep's per-call prints (make_tier_label runs once per fold)."""
    return contextlib.redirect_stdout(io.StringIO())


def _check_finite(X, names, where):
    """Run before every standardize call; names the offending columns."""
    bad = ~np.isfinite(X).all(axis=0)
    if bad.any():
        cols = [names[j] for j in np.flatnonzero(bad)]
        raise ValueError(f"{where}: NaN/inf in {len(cols)} column(s): {cols}")


def _tiers(price_fit, *price_other):
    """
    Tier labels with the threshold = median of price_fit, i.e. of the rows the
    model is trained on. Returns (threshold, labels_fit, *labels_other).
    """
    with _quiet():
        y_fit, threshold = data_prep.make_tier_label(price_fit)
        others = [data_prep.make_tier_label(p, threshold)[0] for p in price_other]
    return (threshold, y_fit, *others)


def _stratified_sample(labels, size):
    """Indices of a stratified random subset of `size` rows (all rows if fewer)."""
    idx = np.arange(len(labels))
    if size >= len(labels):
        return idx
    subset, _ = train_test_split(idx, train_size=size, stratify=labels, random_state=SEED)
    return np.sort(subset)


def _tree_params(key):
    params = dict(FINAL_CONFIGS[key])
    params["max_depth"] = min(params["max_depth"], MAX_TREE_DEPTH)
    return params


def _svm_params(n_features):
    params = dict(FINAL_CONFIGS["svm"])
    if params["gamma"] == "scale":
        params["gamma"] = 1.0 / n_features
    return params


def _slug(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


# --- SVM adapters (common fit / predict / decision_function interface) -----
class _StandardizedPegasos:
    """
    Our PegasosSVM on features scaled by data_prep.standardize using the
    training rows' mean/std. standardize() returns no scaler object, so the
    training matrix is kept and the same statistics are recomputed for new
    rows (deterministic, so identical to scaling everything in one call).
    """

    def __init__(self, names, **params):
        self.names = names
        self.params = params

    def fit(self, X, y):
        _check_finite(X, self.names, "Pegasos fit")
        self._X_fit = X
        # One objective checkpoint per epoch (the default, every 1000 steps,
        # costs a full pass over the data each time).
        self.model_ = PegasosSVM(random_state=SEED, history_every=len(X), **self.params)
        self.model_.fit(data_prep.standardize(X), y)
        return self

    def _scale(self, X):
        _check_finite(X, self.names, "Pegasos predict")
        return data_prep.standardize(self._X_fit, X)[1]

    def decision_function(self, X):
        return self.model_.decision_function(self._scale(X))

    def predict(self, X):
        return self.model_.predict(self._scale(X))


def _sgd_pipeline(svm):
    """
    sklearn's SGD on the same objective as Pegasos. SGDClassifier minimizes
    mean(hinge) + alpha * ||w||^2 / 2, which is Pegasos' objective with
    alpha = lambda_; its default 'optimal' step size 1/(alpha * (t + t0)) is
    Pegasos' 1/(lambda_ * t) schedule, shifted. For the RFF kernel the same
    random-Fourier construction (RBFSampler: W ~ N(0, 2*gamma), b ~ U(0, 2*pi))
    is inserted, so the comparison is optimizer vs optimizer on equivalent
    features rather than linear vs kernel.
    """
    steps = [StandardScaler()]
    if svm["kernel"] == "rff":
        steps.append(RBFSampler(gamma=svm["gamma"], n_components=svm["n_components"],
                                random_state=SEED))
    steps.append(SGDClassifier(loss="hinge", penalty="l2", alpha=svm["lambda_"],
                               max_iter=svm["n_epochs"], tol=None, random_state=SEED))
    return make_pipeline(*steps)


class _SubsampledSVC:
    """
    Exact-kernel sklearn SVC in the same kernel family as ours (RFF
    approximates exactly this RBF kernel), fit on at most `cap` stratified
    training rows because kernel SVC scales roughly quadratically in n.

    C = 1 / (n * lambda_): SVC minimizes ||w||^2 / 2 + C * sum(hinge); dividing
    by C * n gives ||w||^2 / (2 * C * n) + mean(hinge), which is Pegasos'
    objective when lambda_ = 1 / (C * n). n is the number of rows SVC sees.
    """

    def __init__(self, svm, cap):
        self.svm = svm
        self.cap = cap

    def fit(self, X, y):
        idx = _stratified_sample(y, self.cap)
        self.n_fit_ = len(idx)
        self.C_ = 1.0 / (self.n_fit_ * self.svm["lambda_"])
        kernel = {"kernel": "rbf", "gamma": self.svm["gamma"]} if self.svm["kernel"] == "rff" \
            else {"kernel": "linear"}
        self.model_ = make_pipeline(StandardScaler(),
                                    SVC(C=self.C_, cache_size=1000, random_state=SEED, **kernel))
        self.model_.fit(X[idx], y[idx])
        return self

    def decision_function(self, X):
        return self.model_.decision_function(X)

    def predict(self, X):
        return self.model_.predict(X)


# --- Model sets -------------------------------------------------------------
def _task_a_models():
    params = _tree_params("tree_task_a")
    return {
        "Tree (ours)": DecisionTree(task="regression", criterion="mse",
                                    random_state=SEED, **params),
        "Tree (sklearn)": DecisionTreeRegressor(criterion="squared_error",
                                                random_state=SEED, **params),
    }


def _task_b_models(names):
    """name -> (model, score kind). Trees score with P(premium), SVMs with margins."""
    params = _tree_params("tree_task_b")
    svm = _svm_params(len(names))
    svm_args = {k: svm[k] for k in ("lambda_", "kernel", "gamma", "n_components", "n_epochs")}
    models = {}
    for criterion in ("gini", "entropy"):
        models[f"Tree {criterion} (ours)"] = (
            DecisionTree(task="classification", criterion=criterion, random_state=SEED, **params),
            "proba")
        models[f"Tree {criterion} (sklearn)"] = (
            DecisionTreeClassifier(criterion=criterion, random_state=SEED, **params), "proba")
    models["Pegasos SVM (ours)"] = (_StandardizedPegasos(names, **svm_args), "margin")
    models["SGD hinge (sklearn)"] = (_sgd_pipeline(svm), "margin")
    models[f"SVC (sklearn, max {SVC_MAX_ROWS // 1000}k rows)"] = (
        _SubsampledSVC(svm, SVC_MAX_ROWS), "margin")
    return models


def _fit_and_score(model, X_fit, y_fit, X_eval, kind):
    """Fit, then produce every output needed for the metrics, timing both."""
    start = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit_s = time.perf_counter() - start

    start = time.perf_counter()
    if kind == "regression":
        out = {"pred": model.predict(X_eval)}
    else:
        out = {"label": model.predict(X_eval)}
        if kind == "proba":
            if list(model.classes_) != [0, 1]:
                raise ValueError(f"expected classes [0, 1], got {model.classes_}")
            out["score"] = model.predict_proba(X_eval)[:, 1]
        else:
            out["score"] = model.decision_function(X_eval)
    out["predict_s"] = time.perf_counter() - start
    out["fit_s"] = fit_s
    return out


# --- Metrics ----------------------------------------------------------------
def _regression_metrics(price_true, log_pred):
    # exp() of a log-scale prediction estimates the conditional median price,
    # not the mean, so the price-scale errors include that retransformation
    # bias. Both scales are reported for that reason.
    log_true = np.log(price_true)
    price_pred = np.exp(log_pred)
    return {
        "RMSE log": ev.rmse(log_true, log_pred),
        "MAE log": ev.mae(log_true, log_pred),
        "R2 log": ev.r2(log_true, log_pred),
        "RMSE AZN": ev.rmse(price_true, price_pred),
        "MAE AZN": ev.mae(price_true, price_pred),
        "R2 AZN": ev.r2(price_true, price_pred),
    }


def _classification_metrics(y_true, label, score):
    precision, recall, f1 = ev.precision_recall_f1(y_true, label)
    return {"Precision": precision, "Recall": recall, "F1": f1,
            "ROC-AUC": ev.roc_auc(y_true, score)}


def _with_times(metrics, out):
    return {**metrics, "Fit s": out["fit_s"], "Predict s": out["predict_s"]}


def _mean_std(per_fold):
    """{model: [ {metric: value} per fold ]} -> {model: {metric: (mean, std)}}."""
    # std uses ddof=1: the folds are a sample of possible splits.
    return {
        model: {metric: (float(np.mean([f[metric] for f in folds])),
                         float(np.std([f[metric] for f in folds], ddof=1)))
                for metric in folds[0]}
        for model, folds in per_fold.items()
    }


# --- Stage 2: cross-validation ---------------------------------------------
def cross_validate(X, price, names, n_max, n_folds):
    """
    X, price: the CV pool (train + validation rows; test is never touched).
    Stratification labels use the pool's own median; every fold then
    recomputes its tier threshold, and the SVM scaler, from its fit rows only.
    """
    _, strata = _tiers(price)
    subset = _stratified_sample(strata, n_max)
    print(f"CV sample: {len(subset):,} of {len(price):,} train+val rows (stratified), {n_folds} folds")
    X, price, strata = X[subset], price[subset], strata[subset]

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    per_fold = {"A": defaultdict(list), "B": defaultdict(list)}
    svc_rows = set()
    for k, (fit, held) in enumerate(folds.split(X, strata), 1):
        start = time.perf_counter()
        for name, model in _task_a_models().items():
            out = _fit_and_score(model, X[fit], np.log(price[fit]), X[held], "regression")
            per_fold["A"][name].append(_with_times(_regression_metrics(price[held], out["pred"]), out))

        threshold, y_fit, y_held = _tiers(price[fit], price[held])
        for name, (model, kind) in _task_b_models(names).items():
            out = _fit_and_score(model, X[fit], y_fit, X[held], kind)
            per_fold["B"][name].append(_with_times(_classification_metrics(y_held, out["label"], out["score"]), out))
            if isinstance(model, _SubsampledSVC):
                svc_rows.add(model.n_fit_)
        print(f"  fold {k}/{n_folds}: fit {len(fit):,} / eval {len(held):,} rows, "
              f"tier threshold {threshold:,.0f} AZN, {time.perf_counter() - start:.1f}s")

    print(f"  SVC was fit on {sorted(svc_rows)} rows per fold (cap {SVC_MAX_ROWS:,}).")
    return {"A": _mean_std(per_fold["A"]), "B": _mean_std(per_fold["B"])}, {
        "n_rows": int(len(subset)), "n_folds": n_folds, "svc_rows": sorted(svc_rows)}


# --- Stage 3: held-out test ------------------------------------------------
def held_out_test(X_tr, price_tr, X_te, price_te, names, max_train=None):
    """Fit each model once on the training split, score once on test."""
    if max_train is not None:
        _, strata = _tiers(price_tr)
        keep = _stratified_sample(strata, max_train)
        X_tr, price_tr = X_tr[keep], price_tr[keep]
        print(f"QUICK: test-stage models trained on {len(keep):,} stratified training rows")

    threshold, y_tr, y_te = _tiers(price_tr, price_te)
    print(f"Training rows {len(price_tr):,}, test rows {len(price_te):,}, "
          f"tier threshold (train median) {threshold:,.0f} AZN")

    tables, outputs, fitted = {"A": {}, "B": {}}, {"A": {}, "B": {}}, {}
    for name, model in _task_a_models().items():
        out = _fit_and_score(model, X_tr, np.log(price_tr), X_te, "regression")
        tables["A"][name] = _with_times(_regression_metrics(price_te, out["pred"]), out)
        outputs["A"][name], fitted[name] = out, model
        print(f"  {name:<30} fit {out['fit_s']:7.1f}s")
    for name, (model, kind) in _task_b_models(names).items():
        out = _fit_and_score(model, X_tr, y_tr, X_te, kind)
        tables["B"][name] = _with_times(_classification_metrics(y_te, out["label"], out["score"]), out)
        outputs["B"][name], fitted[name] = out, model
        extra = f" (fit on {model.n_fit_:,} rows, C={model.C_:.3g})" if isinstance(model, _SubsampledSVC) else ""
        print(f"  {name:<30} fit {out['fit_s']:7.1f}s{extra}")
    return tables, outputs, fitted, {"threshold": threshold, "y_test": y_te, "n_train": len(price_tr)}


# --- Stage 4: our trees vs sklearn's trees ---------------------------------
def _our_depth_and_leaves(node):
    if node.is_leaf():
        return 0, 1
    depth_l, leaves_l = _our_depth_and_leaves(node.left)
    depth_r, leaves_r = _our_depth_and_leaves(node.right)
    return 1 + max(depth_l, depth_r), leaves_l + leaves_r


def compare_trees(fitted, X_te, names):
    """
    Expected (benign) gaps between the two implementations:
      - thresholds: ours splits at an observed training value (x <= v), sklearn
        at the midpoint between neighbouring values. Training partitions are
        identical, but test rows falling between the two values go different
        ways, so agreement on test can be below 100% even for the same tree;
      - tie-breaking: when two splits give the same impurity decrease, ours
        keeps the first feature scanned, sklearn visits features in a random
        (seeded) order, so the trees can diverge from that node down;
      - min_impurity_decrease: ours compares the raw decrease at the node,
        sklearn weights it by the node's share of training rows, so the same
        value prunes differently (both are 0.0 here, which sidesteps it).
    """
    pairs = [("Task A tree", "Tree (ours)", "Tree (sklearn)", "regression"),
             ("Task B gini", "Tree gini (ours)", "Tree gini (sklearn)", "classification"),
             ("Task B entropy", "Tree entropy (ours)", "Tree entropy (sklearn)", "classification")]
    rows, importances = {}, {"feature": names}
    for label, ours_name, sk_name, task in pairs:
        ours, sk = fitted[ours_name], fitted[sk_name]
        p_ours, p_sk = ours.predict(X_te), sk.predict(X_te)
        agree = np.isclose(p_ours, p_sk, rtol=1e-9, atol=0) if task == "regression" else p_ours == p_sk
        imp_ours = ev.tree_feature_importances(ours, len(names))
        imp_sk = sk.feature_importances_
        corr = float(np.corrcoef(imp_ours, imp_sk)[0, 1]) if imp_ours.std() and imp_sk.std() else float("nan")
        depth, leaves = _our_depth_and_leaves(ours.root)
        rows[label] = {
            "depth ours": depth, "depth sklearn": sk.get_depth(),
            "leaves ours": leaves, "leaves sklearn": sk.get_n_leaves(),
            "root ours": f"{names[ours.root.feature]} <= {ours.root.threshold:.6g}",
            "root sklearn": f"{names[sk.tree_.feature[0]]} <= {sk.tree_.threshold[0]:.6g}",
            "test agreement %": 100.0 * agree.mean(),
            "importance corr": corr,
        }
        importances[f"{_slug(label)}_ours"] = imp_ours
        importances[f"{_slug(label)}_sklearn"] = imp_sk
    return pd.DataFrame.from_dict(rows, orient="index"), pd.DataFrame(importances)


# --- Stage 5: scaling ------------------------------------------------------
def scaling(X_tr, price_tr, names, sizes):
    """Fit time vs n for both trees (Task B, gini) and both linear-hinge SVMs."""
    _, y = _tiers(price_tr)
    params = _tree_params("tree_task_b")
    svm = _svm_params(len(names))
    svm_args = {k: svm[k] for k in ("lambda_", "kernel", "gamma", "n_components", "n_epochs")}
    makers = {
        "Tree gini (ours)": lambda: DecisionTree(task="classification", criterion="gini",
                                                 random_state=SEED, **params),
        "Tree gini (sklearn)": lambda: DecisionTreeClassifier(criterion="gini",
                                                              random_state=SEED, **params),
        "Pegasos SVM (ours)": lambda: _StandardizedPegasos(names, **svm_args),
        "SGD hinge (sklearn)": lambda: _sgd_pipeline(svm),
    }
    times = defaultdict(list)
    for n in sizes:
        idx = _stratified_sample(y, n)
        for name, make in makers.items():
            start = time.perf_counter()
            make().fit(X_tr[idx], y[idx])
            times[name].append(time.perf_counter() - start)
        print(f"  n={n:>6,}: " + "  ".join(f"{name} {t[-1]:.2f}s" for name, t in times.items()))
    table = pd.DataFrame(times, index=pd.Index(sizes, name="n_train"))
    # Empirical exponent k in time ~ n^k: slope of the log-log line.
    slopes = {name: float(np.polyfit(np.log(sizes), np.log(table[name]), 1)[0]) for name in table}
    print("  log-log slopes: " + ", ".join(f"{name} {k:.2f}" for name, k in slopes.items()))
    return table, slopes


def _plot_scaling(table, slopes, stem):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FixedLocator, NullLocator

    # Categorical slots 1-4 of the dataviz reference palette, in fixed order.
    # Ours = solid, sklearn = dashed, plus distinct markers and direct labels,
    # so identity never rests on colour alone.
    style = {
        "Tree gini (ours)": ("#2a78d6", "-", "o"),
        "Tree gini (sklearn)": ("#eb6834", "--", "s"),
        "Pegasos SVM (ours)": ("#1baf7a", "-", "^"),
        "SGD hinge (sklearn)": ("#eda100", "--", "D"),
    }
    ink, muted, grid = "#1f1f1f", "#5f5f5f", "#e3e3e3"
    sizes = table.index.to_numpy()

    fig = Figure(figsize=(7.5, 4.6))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for name, (color, dash, marker) in style.items():
        ax.plot(sizes, table[name], color=color, linestyle=dash, linewidth=2,
                marker=marker, markersize=7, label=f"{name}  (slope {slopes[name]:.2f})")

    # Direct labels at the right end, nudged apart in log space so they never overlap.
    ends = sorted((np.log10(table[name].iloc[-1]), name) for name in style)
    placed = []
    for y, name in ends:
        if placed and y - placed[-1][0] < 0.13:
            y = placed[-1][0] + 0.13
        placed.append((y, name))
    for y, name in placed:
        ax.annotate(name, xy=(sizes[-1], table[name].iloc[-1]), xytext=(sizes[-1] * 1.12, 10 ** y),
                    va="center", fontsize=9, color=ink)

    ax.set(xscale="log", yscale="log", xlim=(sizes[0] / 1.25, sizes[-1] * 3.2),
           xlabel="Training rows (log scale)", ylabel="Fit time, seconds (log scale)")
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{n // 1000}k" for n in sizes])
    ax.grid(True, which="major", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted, labelsize=9)
    ax.xaxis.label.set_color(ink)
    ax.yaxis.label.set_color(ink)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=ink)
    ax.set_title("Fit time vs training size (Task B, one fit per point)", fontsize=11,
                 color=ink, loc="left")
    fig.tight_layout()
    paths = []
    for ext in ("png", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
        paths.append(path)
    return paths


# --- Entry point ------------------------------------------------------------
def run_benchmark(quick: bool = False) -> dict:
    n_max, n_folds = (5_000, 3) if quick else (20_000, 5)
    sizes = (2_000, 4_000, 8_000, 16_000)
    results_dir = ROOT / ("results/quick" if quick else "results")
    tables_dir = results_dir if quick else ROOT / "report/tables"
    figures_dir = results_dir if quick else ROOT / "report/figures"
    for d in (results_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)
    stage_s = {}
    run_start = time.perf_counter()

    print(f"Benchmark ({'QUICK' if quick else 'FULL'} run), SEED={SEED}")
    print("Configs:", json.dumps(FINAL_CONFIGS))

    # 1. data ---------------------------------------------------------------
    start = time.perf_counter()
    with _stage("1. data"):
        df = data_prep.clean(data_prep.load_raw(data_prep.DATA_PATH))
        X, price, names = data_prep.make_features(df)
        _check_finite(X, names, "make_features output")
        # data_prep's split returns no row indices. Its shuffle depends only on
        # the row count and seed, so a row-id column appended to X travels with
        # the rows without changing the split, and is stripped afterwards.
        row_id = np.arange(len(price), dtype=float)
        X_tr, price_tr, X_val, price_val, X_te, price_te = data_prep.train_val_test_split(
            np.column_stack([X, row_id]), price)
        id_tr, id_te = X_tr[:, -1].astype(int), X_te[:, -1].astype(int)
        X_tr, X_val, X_te = X_tr[:, :-1], X_val[:, :-1], X_te[:, :-1]
        assert np.array_equal(price_tr, price[id_tr]) and np.array_equal(price_te, price[id_te])
        print(f"X {X.shape}, train/val/test {len(price_tr):,}/{len(price_val):,}/{len(price_te):,}")
    stage_s["data"] = time.perf_counter() - start

    # 2. cross-validation (train + val only) --------------------------------
    start = time.perf_counter()
    with _stage("2. cross-validation"):
        cv, cv_info = cross_validate(np.vstack([X_tr, X_val]), np.concatenate([price_tr, price_val]),
                                     names, n_max, n_folds)
        for task in ("A", "B"):
            print(f"\nCV, Task {task} (mean ± std over {n_folds} folds):")
            ev.compare(cv[task])
    stage_s["cv"] = time.perf_counter() - start

    # 3. held-out test --------------------------------------------------------
    start = time.perf_counter()
    with _stage("3. held-out test"):
        test, outputs, fitted, test_info = held_out_test(
            X_tr, price_tr, X_te, price_te, names, max_train=5_000 if quick else None)
        for task in ("A", "B"):
            print(f"\nTest, Task {task}:")
            ev.compare(test[task])
    stage_s["test"] = time.perf_counter() - start

    # 4. tree comparison ------------------------------------------------------
    start = time.perf_counter()
    with _stage("4. tree comparison (test set)"):
        tree_table, importances = compare_trees(fitted, X_te, names)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(tree_table.to_string(float_format=lambda v: f"{v:.3f}"))
    stage_s["trees"] = time.perf_counter() - start

    # 5. scaling --------------------------------------------------------------
    start = time.perf_counter()
    with _stage("5. scaling"):
        scale_table, slopes = scaling(X_tr, price_tr, names, sizes)
        figure_paths = _plot_scaling(scale_table, slopes, figures_dir / "benchmark_scaling")
    stage_s["scaling"] = time.perf_counter() - start

    # 6. save -----------------------------------------------------------------
    start = time.perf_counter()
    with _stage("6. save outputs"):
        written = []
        for kind, tables in (("cv", cv), ("test", test)):
            for task in ("A", "B"):
                stem = f"benchmark_{kind}_task_{task.lower()}"
                ev.save_table(tables[task], results_dir / f"{stem}.csv", tables_dir / f"{stem}.tex")
                written += [results_dir / f"{stem}.csv", tables_dir / f"{stem}.tex"]

        # Test-set predictions for error analysis. clean_index is the row label
        # in data_prep.clean(load_raw()), so listings can be joined back.
        preds = pd.DataFrame({"clean_index": df.index[id_te], "price": price_te,
                              "log_price": np.log(price_te), "tier": test_info["y_test"]})
        for name, out in outputs["A"].items():
            preds[f"a_{_slug(name)}_log_pred"] = out["pred"]
            preds[f"a_{_slug(name)}_price_pred"] = np.exp(out["pred"])
        for name, out in outputs["B"].items():
            preds[f"b_{_slug(name)}_label"] = out["label"]
            preds[f"b_{_slug(name)}_score"] = out["score"]
        # (frame, keep_index): the comparison and scaling tables are indexed by
        # tree / n_train; the other two carry their keys as columns.
        extra = {
            "test_predictions.csv": (preds, False),
            "tree_comparison.csv": (tree_table, True),
            "tree_importances.csv": (importances, False),
            "benchmark_scaling.csv": (scale_table, True),
        }
        for filename, (frame, keep_index) in extra.items():
            frame.to_csv(results_dir / filename, index=keep_index, encoding="utf-8")
            written.append(results_dir / filename)

        stage_s["save"] = time.perf_counter() - start
        meta = {
            "quick": quick, "seed": SEED, "data_path": data_prep.DATA_PATH,
            "configs": FINAL_CONFIGS, "max_tree_depth": MAX_TREE_DEPTH,
            "svm_gamma_resolved": _svm_params(len(names))["gamma"],
            "n_features": len(names),
            "split_sizes": {"train": len(price_tr), "val": len(price_val), "test": len(price_te)},
            "cv": cv_info,
            "test": {"tier_threshold_train_median": test_info["threshold"],
                     "n_train_used": test_info["n_train"]},
            "scaling_slopes": slopes, "stage_seconds": stage_s,
            "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                         "scikit-learn": sklearn.__version__},
        }
        (results_dir / "benchmark_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        written += [results_dir / "benchmark_meta.json", *figure_paths]
        for path in written:
            print(f"  wrote {path.relative_to(ROOT)}")

    total = time.perf_counter() - run_start
    print("\nStage times: " + ", ".join(f"{k} {v:.1f}s" for k, v in stage_s.items())
          + f" | total {total:.1f}s")
    return {"cv": cv, "test": test, "trees": tree_table, "scaling": scale_table, "meta": meta}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark our models against scikit-learn.")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 5k-row CV with 3 folds and 5k-row test-stage fits")
    run_benchmark(parser.parse_args().quick)
