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
~166x and ~1660x 'scale' for our 166 standardized features.

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
    results/svm_final_config.json                      (read by src/benchmark.py)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import data_prep
from . import evaluate as ev
from .data_prep import SEED
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
    "lambda (Pegasos needs T >> 1/lambda updates; lambda=1e-6 did not converge on 16k-row "
    "CV folds); final = chosen config with the higher validation F1, tie -> linear"
)

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
    O(1/(lambda * T)) after T updates, so it needs T >> 1/lambda. With
    lambda = 1e-6, 1/lambda = 1e6 updates, but a 16k-row CV fold gives only
    16,000 x 20 epochs = 320k: the RFF model picked by raw F1 (lambda = 1e-6)
    scored 0.908 F1 on validation yet 0.850 +/- 0.035 across CV folds. When F1
    is near-tied, the larger lambda converges in far fewer updates and is the
    more regularized model, so the choice is stable across training sizes.

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


def _figure(height=2.6):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(_WIDTH, height), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.grid(True, which="major", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=_SMALL)
    return fig, ax


def _finish(fig, ax, stem, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=_INK, fontsize=_FONT)
    ax.set_ylabel(ylabel, color=_INK, fontsize=_FONT)
    paths = {}
    for ext in ("png", "pdf"):
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


def plot_validation_f1(table, chosen, final_kernel, out_dir):
    """
    Validation F1 vs lambda for every swept configuration: the linear kernel
    and RFF at each gamma, from the sweep table. Stars mark the configuration
    chosen per kernel (solid star: the final model).
    """
    fig, ax = _figure(2.8)
    groups = [("Linear", table[table.kernel == "linear"])]
    groups += [(f"RFF, γ = {m:g}·scale", table[(table.kernel == "rff") & (table.gamma_x_scale == m)])
               for m in GAMMA_MULTIPLIERS]
    for (label, rows), color, marker in zip(groups, _SLOTS, "os^D"):
        rows = rows.sort_values("lambda_")
        ax.plot(rows["lambda_"], rows["val_F1"], color=color, linewidth=1.5, marker=marker, markersize=4,
                linestyle="-" if label == "Linear" else "--", label=label)
    for kernel, row in chosen.items():
        final = kernel == final_kernel
        ax.scatter([row["lambda_"]], [row["validation_score"]], marker="*", s=130, zorder=5,
                   facecolors=_INK if final else "none", edgecolors=_INK, linewidths=1,
                   label="Final model" if final else f"Chosen {KERNEL_LABEL[kernel]}")
    _lambda_axis(ax)
    _legend(ax, "lower left")
    return {"svm_validation_f1": _finish(fig, ax, out_dir / "svm_validation_f1",
                                         "Regularization λ", "Validation F1")}


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
    X_tr, price_tr, X_val, price_val, _, _ = data_prep.train_val_test_split(X, price)
    with contextlib.redirect_stdout(io.StringIO()):
        y_tr, threshold = data_prep.make_tier_label(price_tr)
        y_val, _ = data_prep.make_tier_label(price_val, threshold)
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
