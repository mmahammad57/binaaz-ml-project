"""
svm_tuning.py — P4b: tune our PegasosSVM on the real data (train -> validation).

Run from the repo root:
    python -m src.svm_tuning            # full training split
    python -m src.svm_tuning --quick    # 5k-row stratified training subsample

The test split is never touched: tier labels come from the training median,
standardization from training statistics, and selection from validation F1.

Grid. docs/SVM.md gives no binding grid: its lambdas (0.001, 0.01, 0.1) and
gammas (0.1, 1, 10) appear only in a usage example / as sweep_svm defaults,
and its handoff asks to "adjust the search using validation evidence" and
"record the actual search space". The search space used here:
    lambda_ in {1e-7, ..., 1e-1} (powers of ten), linear and RFF kernels
      (1e-7 and 1e-6 were added after the best RFF lambda sat on the grid's
      lower edge, 1e-5, in the first sweep);
    RFF gamma in {0.1, 1, 10} x 'scale', where scale = 1 / n_features (the RBF
    width sklearn uses for standardized inputs), with n_components = 500.
The docs' lambdas are a subset; its absolute gammas of 1 and 10 would be
~171x and ~1710x 'scale' for our 171 standardized features.

Selection, per kernel: among configs within F1_TOLERANCE (0.005) validation
F1 of that kernel's best, take the one with the LARGEST lambda (see
_select_stable for why). Both kernels' chosen configs are kept (the benchmark
runs both); the "final" model is the chosen config with the higher validation
F1 (tie -> linear).

Outputs
    results/svm_tuning/{rff,linear}/validation.csv, sweep.json   (save_sweep_results)
    results/svm_tuning/sweep_table.csv
    report/figures/svm_rff_{convergence,validation_sweeps,margin_validation}.{png,pdf}
        (svm_plots.plot_sweep_results; its star marks the best RFF config)
    report/figures/svm_linear_validation.{png,pdf}     F1 and ROC-AUC vs lambda
    report/figures/svm_linear_margin.{png,pdf}         margin width vs lambda
    report/figures/svm_<kernel>_final_convergence.{png,pdf}   final model only
    report/figures/svm_validation_f1.{png,pdf}         F1 vs lambda, linear and RFF per gamma
    report/figures/slide_svm_validation_f1.png         the same figure at slide size (SLIDE_STYLE)
    results/svm_tuning/stability_check.csv             CV F1 of the max-F1 and chosen configs
    results/svm_final_config.json                      (read by src/benchmark.py)
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import data_prep
from . import evaluate as ev
from .benchmark import _StandardizedPegasos, _stratified_sample, _tiers
from .data_prep import SEED
from .splits import get_splits
from .svm import PegasosSVM
from .svm_experiment import save_sweep_results, sweep_svm
from .svm_plots import plot_sweep_results

LAMBDAS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
GAMMA_MULTIPLIERS = (0.1, 1.0, 10.0)       # times 'scale' = 1 / n_features
N_COMPONENTS = 500                          # docs/SVM.md default
N_EPOCHS = 20                               # docs/SVM.md fit and sweep examples
QUICK_TRAIN_ROWS = 5_000
F1_TOLERANCE = 0.005                        # "near-best" band for _select_stable
KERNEL_LABEL = {"linear": "linear", "rff": "RFF"}
SELECTION_RULE = (
    f"per kernel: among configs within {F1_TOLERANCE} validation F1 of the best, the largest "
    "lambda (Pegasos needs T >> 1/lambda updates; see svm_tuning/stability_check.csv); "
    "final = chosen config with the higher validation F1, tie -> linear"
)
# Stability check: the benchmark's cross-validation design (same sample, folds
# and per-fold thresholds/scaling), applied to the max-F1 and chosen configs.
STABILITY_ROWS, STABILITY_FOLDS = 20_000, 5
QUICK_STABILITY_ROWS, QUICK_STABILITY_FOLDS = 5_000, 3

ROOT = Path(__file__).resolve().parents[1]


def _f1_recording_auc(aucs):
    """
    Sweep metric: F1 of the predicted labels, with ROC-AUC of the same scores
    appended to `aucs` as a side record. sweep_svm accepts one metric, so it is
    called with score_input="scores"; labels are recovered exactly as
    PegasosSVM.predict does it (score >= 0 -> classes_[1] = 1, premium).
    sweep_svm calls the metric once per configuration, in grid order.
    """
    def metric(y_true, scores):
        aucs.append(ev.roc_auc(y_true, scores))
        return ev.precision_recall_f1(y_true, (scores >= 0.0).astype(int))[2]
    return metric


def sweep_linear(X_train, y_train, X_val, y_val, *, metric, lambdas, n_epochs, history_every):
    """
    Linear-kernel counterpart of svm_experiment.sweep_svm (which only does RFF).
    Returns the same dict layout, so save_sweep_results can store it.
    """
    results, best = [], None
    for lambda_ in lambdas:
        model = PegasosSVM(kernel="linear", lambda_=lambda_, n_epochs=n_epochs,
                           history_every=history_every, random_state=SEED).fit(X_train, y_train)
        score = float(metric(y_val.copy(), model.decision_function(X_val)))
        results.append({
            "lambda_": float(lambda_), "gamma": None, "n_components": None,
            "validation_score": score, "weight_norm": model.weight_norm_,
            "margin_width": model.margin_width_, "support_count": len(model.support_),
            "objective": model.history_[-1]["objective"],
            "history": [dict(row) for row in model.history_],
        })
        if best is None or score > best[2]:
            best = (model, {"kernel": "linear", "lambda_": float(lambda_), "n_epochs": n_epochs}, score)
    return {
        "best_model": best[0], "best_params": best[1], "best_score": best[2],
        "metric_name": "F1", "score_input": "scores", "positive_class": 1,
        "greater_is_better": True, "n_train": len(X_train), "n_validation": len(X_val),
        "results": results,
    }


def _select_stable(sweep, tolerance=F1_TOLERANCE):
    """
    Among configs within `tolerance` validation F1 of the sweep's best, pick
    the one with the largest lambda (ties in lambda: higher F1).

    Why: Pegasos' step size is 1/(lambda * t) and its guarantee is
    O(1/(lambda * T)) after T updates, so it needs T >> 1/lambda. A small
    lambda can win on the full training split yet be under-trained on
    smaller training sets (e.g. 1/lambda = 1e6 updates for lambda = 1e-6,
    against 16,000 rows x 20 epochs = 320k updates in a CV fold). When F1 is
    near-tied, the larger lambda converges in far fewer updates and is the
    more regularized model. stability_check() measures both configs in CV.

    The sweep's own "selected" fields are pointed at the chosen row, so the
    saved sweep.json and svm_plots' star refer to it; the raw F1 arg-max is
    kept as best_f1_params / best_f1_score.
    """
    rows = sweep["results"]
    top = max(r["validation_score"] for r in rows)
    near = [r for r in rows if r["validation_score"] >= top - tolerance]
    chosen = max(near, key=lambda r: (r["lambda_"], r["validation_score"]))

    sweep["best_f1_params"], sweep["best_f1_score"] = dict(sweep["best_params"]), sweep["best_score"]
    sweep["best_params"] = {**sweep["best_params"], "lambda_": chosen["lambda_"]}
    if chosen["gamma"] is not None:
        sweep["best_params"]["gamma"] = chosen["gamma"]
    sweep["best_score"] = chosen["validation_score"]
    sweep["best_model"] = None      # that fitted model was the raw arg-max; not reused
    sweep["selection_rule"] = SELECTION_RULE
    return chosen


def _best_row(sweep):
    """The results row of the sweep's selected configuration."""
    params = sweep["best_params"]
    return next(r for r in sweep["results"] if r["lambda_"] == params["lambda_"]
                and (r["gamma"] is None or r["gamma"] == params["gamma"]))


def _table(sweeps, scale):
    rows = []
    for kernel, sweep in sweeps.items():
        for r in sweep["results"]:
            rows.append({
                "kernel": kernel, "lambda_": r["lambda_"],
                "gamma_x_scale": None if r["gamma"] is None else round(r["gamma"] / scale, 6),
                "gamma": r["gamma"], "val_F1": r["validation_score"], "val_ROC_AUC": r["roc_auc"],
                "weight_norm": r["weight_norm"], "margin_width": r["margin_width"],
                "support_count": r["support_count"], "objective": r["objective"],
            })
    return pd.DataFrame(rows)


# --- Figures for the linear sweep and the final model ------------------------
# Static report figures following the same conventions as benchmark.py's
# scaling plot: dataviz reference palette slots in fixed order, visible
# markers, recessive grid, text in ink colours, white background. Each is
# drawn at its printed size (one report column, 3.5 in wide) with 8.5-9 pt
# fonts; titles and data notes go in the report captions.
_INK, _MUTED, _GRID = "#1f1f1f", "#5f5f5f", "#e3e3e3"
_SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
_WIDTH, _FONT, _SMALL = 3.5, 9, 8.5
# Sizes for plot_validation_f1: the report column, and the slide version used
# by presentation/slides.tex (6 x 3.5 in, 16 pt text, thicker marks, legend
# beside the axes so the plot keeps its height; PNG only).
# tick_step labels every k-th grid lambda (None: matplotlib's choice); at 16 pt
# all seven labels would overlap on the narrower slide axis.
REPORT_STYLE = dict(width=_WIDTH, height=3.3, font=_FONT, small=_SMALL, line=1.5, marker=4, star=130,
                    legend=dict(loc="outside lower center", ncols=2), tick_step=None, exts=("png", "pdf"))
SLIDE_STYLE = dict(width=6.0, height=3.5, font=16, small=16, line=2.5, marker=7, star=360,
                   legend=dict(loc="outside right center", ncols=1), tick_step=2, exts=("png",))


def _figure(height=2.6, width=_WIDTH, small=_SMALL):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.grid(True, which="major", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=small)
    return fig, ax


def _finish(fig, ax, stem, xlabel, ylabel, font=_FONT, exts=("png", "pdf")):
    ax.set_xlabel(xlabel, color=_INK, fontsize=font)
    ax.set_ylabel(ylabel, color=_INK, fontsize=font)
    paths = {}
    for ext in exts:
        paths[ext] = stem.with_suffix(f".{ext}")
        fig.savefig(paths[ext], dpi=300, facecolor="white")
    return paths


def _lambda_axis(ax):
    """Log lambda axis labelled 1e-7 ... 1e-1: a mathtext 10^k exponent would print below 8 pt."""
    from matplotlib.ticker import FuncFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{round(np.log10(v))}"))


def _legend(ax, loc, **kwargs):
    ax.legend(loc=loc, frameon=False, fontsize=_SMALL, labelcolor=_INK, handlelength=1.5, **kwargs)


def _marker_label(kernel, final_kernel):
    """The star is 'final model' only when this kernel's best config won overall."""
    if kernel == final_kernel:
        return "Selected = final model"
    return f"Chosen {KERNEL_LABEL[kernel]} config"


def plot_linear_sweep(sweep, final_kernel, out_dir):
    rows = sorted(sweep["results"], key=lambda r: r["lambda_"])
    lam = np.array([r["lambda_"] for r in rows])
    best = _best_row(sweep)
    star = _marker_label("linear", final_kernel)
    paths = {}

    # Validation F1 and ROC-AUC share one axis: both are fractions in [0, 1].
    fig, ax = _figure()
    series = (("Validation F1", "validation_score", "o"), ("Validation ROC-AUC", "roc_auc", "s"))
    for (label, key, marker), color in zip(series, _SLOTS):
        ax.plot(lam, [r[key] for r in rows], color=color, linewidth=1.5, marker=marker, markersize=4,
                label=label)
    ax.scatter([best["lambda_"]], [best["validation_score"]], marker="*", s=120,
               facecolors="none", edgecolors=_INK, linewidths=1, zorder=5, label=star)
    _lambda_axis(ax)
    _legend(ax, "lower center")
    paths["svm_linear_validation"] = _finish(fig, ax, out_dir / "svm_linear_validation",
                                             "Regularization λ", "Validation score")

    # Margin width 2/||w|| in the standardized feature space.
    fig, ax = _figure()
    ax.plot(lam, [r["margin_width"] for r in rows], color=_SLOTS[0], linewidth=1.5, marker="o", markersize=4)
    ax.scatter([best["lambda_"]], [best["margin_width"]], marker="*", s=120,
               facecolors="none", edgecolors=_INK, linewidths=1, zorder=5, label=star)
    from matplotlib.ticker import FuncFormatter
    _lambda_axis(ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    _legend(ax, "upper left")
    paths["svm_linear_margin"] = _finish(fig, ax, out_dir / "svm_linear_margin",
                                         "Regularization λ", "Margin width 2 / ||w||")
    return paths


def plot_validation_f1(table, chosen, final_kernel, out_dir, stem="svm_validation_f1", style=REPORT_STYLE):
    """
    Validation F1 vs lambda for every swept configuration: the linear kernel
    and RFF at each gamma, from the sweep table. Stars mark the configuration
    chosen per kernel (solid star: the final model). `style` sets only sizes
    (REPORT_STYLE or SLIDE_STYLE); the data and marks are the same.
    """
    s = style
    fig, ax = _figure(s["height"], s["width"], s["small"])
    groups = [("Linear", table[table.kernel == "linear"])]
    groups += [(f"RFF, γ = {m:g}·scale", table[(table.kernel == "rff") & (table.gamma_x_scale == m)])
               for m in GAMMA_MULTIPLIERS]
    for (label, rows), color, marker in zip(groups, _SLOTS, "os^D"):
        rows = rows.sort_values("lambda_")
        ax.plot(rows["lambda_"], rows["val_F1"], color=color, linewidth=s["line"], marker=marker,
                markersize=s["marker"], linestyle="-" if label == "Linear" else "--", label=label)
    for kernel, row in chosen.items():
        final = kernel == final_kernel
        ax.scatter([row["lambda_"]], [row["validation_score"]], marker="*", s=s["star"], zorder=5, clip_on=False,
                   facecolors=_INK if final else "none", edgecolors=_INK, linewidths=1,
                   label="Final model" if final else f"Chosen {KERNEL_LABEL[kernel]}")
    _lambda_axis(ax)
    if s["tick_step"]:
        ax.set_xticks(np.sort(table["lambda_"].unique())[::s["tick_step"]])
    # Outside the axes (below in the report, beside on the slide): inside, it would cover sweep points.
    fig.legend(**s["legend"], frameon=False, fontsize=s["small"], labelcolor=_INK,
               handlelength=1.5, columnspacing=1.0)
    return {stem: _finish(fig, ax, out_dir / stem, "Regularization λ", "Validation F1",
                          font=s["font"], exts=s["exts"])}


def plot_final_convergence(row, kernel, out_dir):
    history = row["history"]
    steps = [h["step"] for h in history]
    fig, ax = _figure()
    ax.plot(steps, [h["objective"] for h in history], color=_SLOTS[0], linewidth=1.5,
            marker="o", markersize=4, label="Objective λ/2·||w||² + mean hinge")
    ax.plot(steps, [h["hinge_loss"] for h in history], color=_SLOTS[1], linewidth=1.5,
            linestyle="--", label="Mean hinge loss")
    ax.set_ylim(0, None)
    _legend(ax, "upper right")
    return {f"svm_{kernel}_final_convergence": _finish(
        fig, ax, out_dir / f"svm_{kernel}_final_convergence",
        "Training updates (one checkpoint per epoch)", "Loss on training rows")}


def _plot_rff_with_kernel_names(sweep, out_dir, data_label):
    """
    svm_plots.plot_sweep_results always writes svm_convergence /
    svm_validation_sweeps / svm_margin_validation; render into a temporary
    folder and store them as svm_rff_* so every filename names its kernel
    (the titles already say "RFF + Pegasos").
    """
    paths = {}
    with tempfile.TemporaryDirectory() as tmp:
        written = plot_sweep_results(sweep, tmp, data_label=data_label, score_as_percent=True)
        for stem, by_ext in written.items():
            new_stem = stem.replace("svm_", "svm_rff_", 1)
            paths[new_stem] = {}
            for ext, path in by_ext.items():
                target = out_dir / f"{new_stem}.{ext}"
                shutil.move(path, target)
                paths[new_stem][ext] = target
    return paths


# --- Stability check -----------------------------------------------------------
def stability_check(X_pool, price_pool, names, sweeps, scale, n_max, n_folds):
    """
    k-fold CV F1 (mean ± std) of each kernel's highest-validation-F1 config
    and of its chosen config, using the benchmark's CV design: a stratified
    n_max-row sample of train + validation, StratifiedKFold(seed), and in
    every fold a tier threshold and scaler from the fold's training rows only
    (benchmark._tiers, benchmark._StandardizedPegasos). The test split is
    not part of X_pool.
    """
    _, strata = _tiers(price_pool)
    subset = _stratified_sample(strata, n_max)
    X, price, strata = X_pool[subset], price_pool[subset], strata[subset]
    folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED).split(X, strata))
    rows, cache = [], {}
    for kernel, sweep in sweeps.items():
        for role, params in (("max validation F1", sweep["best_f1_params"]), ("chosen", sweep["best_params"])):
            gamma = params.get("gamma") if kernel == "rff" else None
            key = (kernel, params["lambda_"], gamma)
            if key not in cache:                    # max-F1 and chosen configs may coincide
                f1 = []
                for fit, held in folds:
                    _, y_fit, y_held = _tiers(price[fit], price[held])
                    model = _StandardizedPegasos(names, kernel=kernel, lambda_=params["lambda_"],
                                                 gamma=gamma if gamma is not None else scale,
                                                 n_components=N_COMPONENTS, n_epochs=N_EPOCHS)
                    model.fit(X[fit], y_fit)
                    f1.append(ev.precision_recall_f1(y_held, model.predict(X[held]))[2])
                cache[key] = (float(np.mean(f1)), float(np.std(f1, ddof=1)))
            row = next(r for r in sweep["results"] if r["lambda_"] == params["lambda_"]
                       and (gamma is None or r["gamma"] == gamma))
            rows.append({
                "kernel": kernel, "config": role, "lambda_": params["lambda_"],
                "gamma_x_scale": None if gamma is None else round(gamma / scale, 6),
                "val_F1": row["validation_score"], "cv_F1_mean": cache[key][0], "cv_F1_std": cache[key][1],
                "f1_tolerance": F1_TOLERANCE, "n_epochs": N_EPOCHS,
                "cv_rows": len(subset), "cv_folds": n_folds, "fit_rows_per_fold": len(folds[0][0]),
                "updates_per_fit": len(folds[0][0]) * N_EPOCHS, "one_over_lambda": 1.0 / params["lambda_"],
            })
    return pd.DataFrame(rows)


# --- Entry point --------------------------------------------------------------
def run_tuning(quick: bool = False) -> dict:
    start_all = time.perf_counter()
    results_dir = ROOT / ("results/quick" if quick else "results")
    figures_dir = results_dir / "svm_tuning" if quick else ROOT / "report/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    timings = {}

    # Data: tier labels and scaling from the training rows only.
    start = time.perf_counter()
    df = data_prep.clean(data_prep.load_raw(data_prep.DATA_PATH))
    X, price, names = data_prep.make_features(df)
    split = get_splits(X, price)            # the shared stratified split (src/splits.py)
    X_tr, X_val = X[split.train], X[split.val]
    y_tr, y_val, threshold = split.y_train, split.y_val, split.threshold
    # Unscaled train + validation rows, in the benchmark's order, for the CV stability check.
    X_pool = np.vstack([X_tr, X_val])
    price_pool = np.concatenate([price[split.train], price[split.val]])
    if quick:
        keep, _ = train_test_split(np.arange(len(y_tr)), train_size=QUICK_TRAIN_ROWS,
                                   stratify=y_tr, random_state=SEED)
        X_tr, y_tr = X_tr[np.sort(keep)], y_tr[np.sort(keep)]
        print(f"QUICK: training on a stratified subsample of {len(y_tr):,} rows")
    for where, M in (("train", X_tr), ("validation", X_val)):
        bad = ~np.isfinite(M).all(axis=0)
        if bad.any():
            raise ValueError(f"{where}: NaN/inf in columns {[names[j] for j in np.flatnonzero(bad)]}")
    X_tr, X_val = data_prep.standardize(X_tr, X_val)
    scale = 1.0 / X_tr.shape[1]
    gammas = tuple(m * scale for m in GAMMA_MULTIPLIERS)
    history_every = len(X_tr)        # one objective checkpoint per epoch (docs/SVM.md)
    print(f"train {len(y_tr):,} rows ({y_tr.mean():.1%} premium), validation {len(y_val):,} rows, "
          f"{X_tr.shape[1]} features, tier threshold {threshold:,.0f} AZN, "
          f"gamma 'scale' = 1/{X_tr.shape[1]} = {scale:.6f}")
    timings["data"] = time.perf_counter() - start

    # Sweeps (validation only).
    sweeps = {}
    start = time.perf_counter()
    aucs = []
    sweeps["linear"] = sweep_linear(X_tr, y_tr, X_val, y_val, metric=_f1_recording_auc(aucs),
                                    lambdas=LAMBDAS, n_epochs=N_EPOCHS, history_every=history_every)
    for row, auc in zip(sweeps["linear"]["results"], aucs, strict=True):
        row["roc_auc"] = auc
    timings["linear sweep"] = time.perf_counter() - start
    print(f"linear sweep: {len(LAMBDAS)} configs in {timings['linear sweep']:.1f}s")

    start = time.perf_counter()
    aucs = []
    sweeps["rff"] = sweep_svm(X_tr, y_tr, X_val, y_val, metric=_f1_recording_auc(aucs),
                              metric_name="F1", score_input="scores", greater_is_better=True,
                              lambdas=LAMBDAS, gammas=gammas, n_epochs=N_EPOCHS,
                              history_every=history_every, n_components=N_COMPONENTS,
                              random_state=SEED)
    for row, auc in zip(sweeps["rff"]["results"], aucs, strict=True):
        row["roc_auc"] = auc
    timings["rff sweep"] = time.perf_counter() - start
    print(f"RFF sweep: {len(LAMBDAS) * len(gammas)} configs in {timings['rff sweep']:.1f}s")

    # Selection: per kernel, the largest lambda within F1_TOLERANCE of the best
    # (_select_stable explains why); the final model is the chosen config with
    # the higher validation F1, a tie going to linear (simpler, and its
    # weights map to named features).
    best = {kernel: _select_stable(sweep) for kernel, sweep in sweeps.items()}
    final = "rff" if best["rff"]["validation_score"] > best["linear"]["validation_score"] else "linear"

    # Save sweeps, figures and the chosen configs.
    start = time.perf_counter()
    for kernel, sweep in sweeps.items():
        save_sweep_results(sweep, results_dir / "svm_tuning" / kernel)
    base = (f"bina.az tier, train={len(y_tr):,}{' (quick subsample)' if quick else ''}, "
            f"validation={len(y_val):,}")
    rff_note = " | star = chosen RFF config" + ("" if final == "rff" else "; final model is linear")
    figure_paths = _plot_rff_with_kernel_names(sweeps["rff"], figures_dir, base + rff_note)
    figure_paths.update(plot_linear_sweep(sweeps["linear"], final, figures_dir))
    figure_paths.update(plot_final_convergence(best[final], final, figures_dir))

    configs = {
        "linear": {"kernel": "linear", "lambda_": best["linear"]["lambda_"],
                   # Linear Pegasos ignores gamma and n_components; they keep
                   # the config valid for every PegasosSVM argument check.
                   "gamma": "scale", "n_components": N_COMPONENTS, "n_epochs": N_EPOCHS},
        "rff": {"kernel": "rff", "lambda_": best["rff"]["lambda_"], "gamma": best["rff"]["gamma"],
                "n_components": N_COMPONENTS, "n_epochs": N_EPOCHS},
    }
    record = {
        "configs": configs,
        "final": final,
        "selection": {
            "metric": "validation F1 (premium = 1)",
            "rule": SELECTION_RULE,
            **{kernel: {"validation_f1": row["validation_score"], "validation_roc_auc": row["roc_auc"],
                        "gamma_x_scale": None if row["gamma"] is None else round(row["gamma"] / scale, 6),
                        "lambda_on_grid_edge": row["lambda_"] in (min(LAMBDAS), max(LAMBDAS)),
                        "max_f1_config": {k: sweeps[kernel]["best_f1_params"][k]
                                          for k in ("lambda_", "gamma") if k in sweeps[kernel]["best_f1_params"]},
                        "max_f1": sweeps[kernel]["best_f1_score"]}
               for kernel, row in best.items()},
        },
        "search_space": {"lambdas": list(LAMBDAS), "kernels": ["linear", "rff"],
                         "gamma_multipliers_of_scale": list(GAMMA_MULTIPLIERS),
                         "gamma_scale": scale, "n_components": N_COMPONENTS, "n_epochs": N_EPOCHS,
                         "f1_tolerance": F1_TOLERANCE},
        "data": {"n_train": int(len(y_tr)), "n_validation": int(len(y_val)),
                 "quick_subsample": quick, "tier_threshold_train_median": threshold,
                 "n_features": int(X_tr.shape[1]), "seed": SEED, "test_used": False},
        "source": "src/svm_tuning.py",
        "date": date.today().isoformat(),
    }
    config_path = results_dir / "svm_final_config.json"
    config_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    timings["save"] = time.perf_counter() - start

    table = _table(sweeps, scale)
    table.to_csv(results_dir / "svm_tuning" / "sweep_table.csv", index=False)
    figure_paths.update(plot_validation_f1(table, best, final, figures_dir))
    figure_paths.update(plot_validation_f1(table, best, final, figures_dir,
                                           "slide_svm_validation_f1", SLIDE_STYLE))

    start = time.perf_counter()
    n_max, n_folds = (QUICK_STABILITY_ROWS, QUICK_STABILITY_FOLDS) if quick else (STABILITY_ROWS, STABILITY_FOLDS)
    stability = stability_check(X_pool, price_pool, names, sweeps, scale, n_max, n_folds)
    stability.to_csv(results_dir / "svm_tuning" / "stability_check.csv", index=False)
    timings["stability check"] = time.perf_counter() - start
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"\nStability check ({n_folds}-fold CV on {n_max:,} train+val rows):\n"
              + stability.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    def rows_matching(kernel, params):
        hit = (table.kernel == kernel) & (table.lambda_ == params["lambda_"])
        if kernel == "rff":
            hit &= table.gamma == params["gamma"]
        return hit.to_numpy()

    marks = np.full(len(table), "", dtype=object)
    for kernel, sweep in sweeps.items():
        marks[rows_matching(kernel, sweep["best_f1_params"])] = "maxF1"
        marks[rows_matching(kernel, sweep["best_params"])] = "FINAL" if kernel == final else "chosen"
    table.insert(0, "pick", marks)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n" + table.drop(columns="gamma").to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nSelection rule: {SELECTION_RULE}")
    for kernel, row in best.items():
        sweep = sweeps[kernel]
        moved = ("" if sweep["best_f1_params"]["lambda_"] == row["lambda_"]
                 and sweep["best_f1_params"].get("gamma") == sweep["best_params"].get("gamma")
                 else f"  (max-F1 config lambda_={sweep['best_f1_params']['lambda_']:g}, "
                      f"F1 {sweep['best_f1_score']:.4f})")
        print(f"chosen {kernel:<6}: {json.dumps(configs[kernel])} | val F1 {row['validation_score']:.4f}, "
              f"ROC-AUC {row['roc_auc']:.4f}{moved}")
    print(f"final model (higher validation F1 of the chosen configs): {final}")
    print("Wrote: " + str(config_path.relative_to(ROOT)) + ", " + ", ".join(
        str(p["png"].relative_to(ROOT)) for p in figure_paths.values()))
    timings["total"] = time.perf_counter() - start_all
    print("Timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    return {"sweeps": sweeps, "config": record, "table": table, "timings": timings}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune PegasosSVM on train -> validation.")
    parser.add_argument("--quick", action="store_true",
                        help=f"train on a {QUICK_TRAIN_ROWS:,}-row stratified subsample")
    run_tuning(parser.parse_args().quick)
