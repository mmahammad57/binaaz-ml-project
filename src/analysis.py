"""
analysis.py — Problem 6: main results table, figures and error analysis.

Run from the repo root, after a full `python -m src.benchmark`:
    python -m src.analysis

Inputs: the benchmark outputs in results/ (test_predictions.csv,
tree_importances.csv, tree_comparison.csv, the CV and test tables,
benchmark_meta.json), results/svm_final_config.json, and
data_prep.clean(load_raw()) for listing attributes.

No main model is refit. The one exception is the final linear Pegasos SVM,
refit with its saved config only to read its weights (the benchmark does not
store them); the refit is deterministic and checked against the benchmark's
saved test scores.

Outputs
    report/tables/results_main.{csv,tex}         all models x both tasks, test + CV
    report/figures/*.{png,pdf}                   see the plot_* functions
    results/error_analysis.md                    tables + findings for the report
    results/error_analysis/*.csv                 every error table
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_prep
from . import evaluate as ev
from .data_prep import SEED
from .svm import PegasosSVM

ROOT = Path(__file__).resolve().parents[1]
# Output folders; run_analysis(quick=True) redirects them all under results/quick.
PATHS = {}


def _set_paths(quick):
    results = ROOT / ("results/quick" if quick else "results")
    PATHS.update(results=results, errors=results / "error_analysis",
                 figures=results if quick else ROOT / "report/figures",
                 tables=results if quick else ROOT / "report/tables")

# Same style as benchmark_scaling: dataviz reference palette, categorical slots
# in fixed order, ink-coloured text, recessive grid, white background.
INK, MUTED, GRID = "#1f1f1f", "#5f5f5f", "#e3e3e3"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
BLUE_RAMP = ("#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
DIVERGING = {"positive": "#2a78d6", "negative": "#e34948"}     # blue <-> red pair

CATEGORY_EN = {"Yeni tikili": "New build", "Köhnə tikili": "Old build",
               "Həyət evi/Bağ evi": "House / villa", "Ofis": "Office"}
DISTRICT_BINS = [(0, 0, "0 (unseen in train)"), (1, 49, "1-49"), (50, 199, "50-199"),
                 (200, 999, "200-999"), (1000, np.inf, ">= 1000")]
DISTANCE_BINS = [0, 5, 10, 20, 40, np.inf]      # |price / threshold - 1|, in %


# --- Loading ----------------------------------------------------------------
def _slug(name):
    """Same rule as benchmark._slug, which named the prediction columns."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _parse_cell(text):
    """'0.91 ± 0.01' -> (0.91, 0.01); '0.91' -> 0.91 (save_table's format)."""
    text = str(text)
    if "±" in text:
        mean, std = text.split("±")
        return float(mean), float(std)
    return float(text)


def _read_table(path):
    frame = pd.read_csv(path, index_col="model", dtype=str)
    return {model: {metric: _parse_cell(v) for metric, v in row.items()}
            for model, row in frame.iterrows()}


def load_inputs(quick):
    results = PATHS["results"]
    meta_path = results / "benchmark_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path.relative_to(ROOT)} not found; run python -m src.benchmark"
                                f"{' --quick' if quick else ''} first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["quick"] != quick:
        raise RuntimeError(f"{meta_path.relative_to(ROOT)} is from a {'QUICK' if meta['quick'] else 'FULL'} "
                           f"benchmark run; re-run the benchmark{' --quick' if quick else ''}")
    tables = {(kind, task): _read_table(results / f"benchmark_{kind}_task_{task.lower()}.csv")
              for kind in ("cv", "test") for task in ("A", "B")}
    preds = pd.read_csv(results / "test_predictions.csv")
    importances = pd.read_csv(results / "tree_importances.csv")
    tree_cmp = pd.read_csv(results / "tree_comparison.csv", index_col=0)
    # The SVM configs the benchmark actually used (not whatever the config file says now).
    svm_cfg = meta["configs"]["svm"]
    return meta, tables, preds, importances, tree_cmp, svm_cfg


def rebuild_split(df, X, price):
    """
    Recreate the benchmark's split (same call, same row-id trick) to get the
    training rows (for district sizes, decile edges and the SVM refit).
    """
    row_id = np.arange(len(price), dtype=float)
    with contextlib.redirect_stdout(io.StringIO()):
        X_tr, _, _, _, X_te, _ = data_prep.train_val_test_split(np.column_stack([X, row_id]), price)
    return X_tr[:, -1].astype(int), X_te[:, -1].astype(int)


# --- Main results table -----------------------------------------------------
# (benchmark metric, column in the main table, scale factor)
A_METRICS = [("RMSE log", "A RMSE log", 1), ("MAE log", "A MAE log", 1), ("R2 log", "A R2 log", 1),
             ("RMSE AZN", "A RMSE kAZN", 1e-3), ("MAE AZN", "A MAE kAZN", 1e-3), ("R2 AZN", "A R2 AZN", 1)]
B_METRICS = [("Precision", "B Precision", 1), ("Recall", "B Recall", 1),
             ("F1", "B F1", 1), ("ROC-AUC", "B ROC-AUC", 1)]


def _scaled(value, factor):
    return tuple(v * factor for v in value) if isinstance(value, tuple) else value * factor


def main_results_table(tables):
    """
    One table, all models x both tasks: a test row and a CV row (mean ± std)
    per model. A model only has the metrics of its own task; the others print
    as n/a. AZN errors are in thousands of AZN.
    """
    rows = {}
    for task, specs in (("A", A_METRICS), ("B", B_METRICS)):
        for model in tables[("test", task)]:
            for split, kind in (("test", "test"), ("CV", "cv")):
                values = tables[(kind, task)][model]
                rows[f"{model}, {split}"] = {col: _scaled(values[m], k) for m, col, k in specs}
    ev.save_table(rows, PATHS["tables"] / "results_main.csv", PATHS["tables"] / "results_main.tex", decimals=3)
    return rows


# --- Figure helpers -----------------------------------------------------------
def _figure(ncols=1, size=(7.0, 4.4), **kwargs):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=size)
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(1, ncols, **kwargs))
    for ax in axes:
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
    return fig, axes


def _label(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK)


def _save(fig, stem, title, note=None):
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, color=INK)
    if note:
        fig.text(0.01, 0.005, note, fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 0.97))
    paths = []
    for ext in ("png", "pdf"):
        path = PATHS["figures"] / f"{stem}.{ext}"
        fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
        paths.append(path)
    return paths


def _blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)


# --- Task B figures -------------------------------------------------------------
def plot_confusion(y, labels_by_model):
    fig, axes = _figure(len(labels_by_model), size=(4.2 * len(labels_by_model), 4.0))
    names = ["Standard", "Premium"]
    for ax, (model, labels) in zip(axes, labels_by_model.items()):
        cm = ev.confusion_matrix(y, labels)
        rates = cm / cm.sum(axis=1, keepdims=True)          # share of each TRUE class
        ax.grid(False)
        ax.imshow(rates, cmap=_blue_cmap(), vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{rates[i, j]:.1%}", ha="center", va="center",
                        fontsize=10, color="white" if rates[i, j] > 0.5 else INK)
        ax.set_xticks([0, 1], names)
        ax.set_yticks([0, 1], names)
        accuracy = np.trace(cm) / cm.sum()
        _label(ax, f"{model}  (accuracy {accuracy:.1%})", "Predicted tier", "True tier")
    return _save(fig, "analysis_confusion_matrices", "Task B confusion matrices on the test set",
                 "Cells: listing count and share of the true tier (rows sum to 100%).")


def _task_b_groups(models):
    return [("Trees", [m for m in models if m.startswith("Tree")]),
            ("Linear SVMs", [m for m in models if "linear" in m]),
            ("RFF / RBF SVMs", [m for m in models if "RFF" in m or "RBF" in m])]


def plot_roc_pr(y, preds, models):
    """One ROC figure and one PR figure; panels by model family (<= 4 series each)."""
    paths = []
    for kind in ("roc", "pr"):
        fig, axes = _figure(3, size=(13.5, 4.6), sharey=True)
        for ax, (family, members) in zip(axes, _task_b_groups(models)):
            for i, model in enumerate(members):
                scores = preds[f"b_{_slug(model)}_score"].to_numpy()
                labels = preds[f"b_{_slug(model)}_label"].to_numpy()
                style = dict(color=SLOTS[i], linewidth=2, linestyle="-" if "(ours)" in model else "--")
                if kind == "roc":
                    fpr, tpr, _ = ev.roc_curve(y, scores)
                    ax.plot(fpr, tpr, label=f"{model}  AUC {ev.roc_auc(y, scores):.3f}", **style)
                else:
                    precision, recall, _ = ev.precision_recall_curve(y, scores)
                    p, r, _ = ev.precision_recall_f1(y, labels)
                    ax.plot(recall, precision, label=model, **style)
                    # The model's own decision threshold (label predictions).
                    ax.scatter([r], [p], s=55, color=SLOTS[i], edgecolors="white", linewidths=1.5, zorder=5)
            if kind == "roc":
                ax.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1)
                _label(ax, family, "False positive rate", "True positive rate" if ax is axes[0] else None)
                ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
            else:
                ax.axhline(y.mean(), color=MUTED, linestyle=":", linewidth=1)
                ax.set_ylim(0.45, 1.01)
                _label(ax, family, "Recall (premium)", "Precision (premium)" if ax is axes[0] else None)
                ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
        if kind == "roc":
            paths += _save(fig, "analysis_roc_curves", "Task B ROC curves on the test set",
                           "Solid = ours, dashed = scikit-learn. Dotted diagonal = random ranking.")
        else:
            paths += _save(fig, "analysis_pr_curves", "Task B precision-recall curves on the test set",
                           f"Dots = each model's own decision threshold. Dotted line = premium share "
                           f"({y.mean():.1%}), the precision of random ranking.")
    return paths


# --- Task A figures -------------------------------------------------------------
def plot_task_a(df):
    paths = []
    fig, (ax,) = _figure(size=(6.4, 5.6))
    hb = ax.hexbin(df["price"], df["a_pred_price"], xscale="log", yscale="log", gridsize=55,
                   bins="log", mincnt=1, cmap=_blue_cmap(), linewidths=0)
    lo, hi = df["price"].min(), df["price"].max()
    ax.plot([lo, hi], [lo, hi], color=INK, linestyle="--", linewidth=1.2, label="Perfect prediction")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.colorbar(hb, ax=ax, label="Listings per cell (log scale)")
    _label(ax, None, "True price, AZN (log scale)", "Predicted price, AZN (log scale)")
    paths += _save(fig, "analysis_task_a_pred_vs_actual",
                   "Task A, our tree: predicted vs true price (test set)",
                   "Predictions are exp(predicted log price).")

    fig, (ax,) = _figure(size=(7.5, 4.6))
    ax.hexbin(df["price"], df["a_res_log"], xscale="log", gridsize=60, bins="log", mincnt=1,
              cmap=_blue_cmap(), linewidths=0)
    ax.axhline(0, color=INK, linewidth=1)
    deciles = df.groupby("decile", observed=True).agg(price=("price", "median"), res=("a_res_log", "median"))
    ax.plot(deciles["price"], deciles["res"], color=SLOTS[1], linewidth=2, marker="o", markersize=7,
            label="Median residual per true-price decile")
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    _label(ax, None, "True price, AZN (log scale)", "log(predicted) - log(true)\n(+ = over-prediction)")
    paths += _save(fig, "analysis_task_a_residuals", "Task A, our tree: residuals vs true price (test set)",
                   "A residual of 0.1 is about 10.5% over-prediction; -0.1 is about 9.5% under.")
    return paths


# --- Feature importance and SVM weights -------------------------------------------
def plot_importances(importances, top=15):
    fig, axes = _figure(2, size=(13.0, 6.2))
    for ax, (task, ours, theirs) in zip(axes, (
            ("Task A (regression)", "task_a_tree_ours", "task_a_tree_sklearn"),
            ("Task B (gini)", "task_b_gini_ours", "task_b_gini_sklearn"))):
        rows = importances.nlargest(top, ours).iloc[::-1]
        y = np.arange(len(rows))
        ax.barh(y + 0.2, rows[ours], height=0.38, color=SLOTS[0], label="Our tree")
        ax.barh(y - 0.2, rows[theirs], height=0.38, color=SLOTS[1], label="scikit-learn tree")
        ax.set_yticks(y, rows["feature"])
        ax.grid(axis="y", visible=False)
        corr = np.corrcoef(importances[ours], importances[theirs])[0, 1]
        _label(ax, f"{task}: top {top} by our importance (all-feature corr {corr:.3f})",
               "Mean decrease in impurity (normalised)")
        ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
    return _save(fig, "analysis_feature_importance", "Feature importance: our trees vs scikit-learn")


def plot_svm_weights(weights, names, stats, top=15):
    order = np.argsort(weights)
    pick = np.r_[order[:top], order[-top:]]              # 15 most negative, 15 most positive
    fig, (ax,) = _figure(size=(7.8, 8.2))
    y = np.arange(len(pick))
    colors = [DIVERGING["positive"] if weights[j] > 0 else DIVERGING["negative"] for j in pick]
    ax.barh(y, weights[pick], color=colors, height=0.7)
    ax.set_yticks(y, [names[j] for j in pick])
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    _label(ax, "Blue pushes towards premium, red towards standard",
           "Weight per +1 standard deviation of the feature")
    return _save(fig, "analysis_svm_weights",
                 f"Final linear Pegasos SVM: top ±{top} weights",
                 f"||w|| = {stats['weight_norm']:.3f}, margin 2/||w|| = {stats['margin_width']:.3f} "
                 f"(standardized units), {stats['support_count']:,} training points on or inside the margin.")


# --- Error analysis ---------------------------------------------------------------
def analysis_frame(df_clean, preds, train_ids, names_tree, name_svm, threshold):
    """One row per test listing: attributes, both tasks' errors, and the groupings."""
    attrs = df_clean.loc[preds["clean_index"]].reset_index(drop=True)
    train = df_clean.iloc[train_ids]
    out = pd.DataFrame({
        "clean_index": preds["clean_index"], "price": preds["price"], "tier": preds["tier"],
        "category": attrs["Kateqoriya"].map(CATEGORY_EN), "city": attrs["city"].str.title(),
        "district": attrs["location"], "area_m2": attrs["Sahə"], "rooms": attrs["Otaq sayı"],
        "floor": attrs["floor"], "total_floors": attrs["total_floors"],
        "land_sot": attrs["land_area_sot"], "repair": attrs["Təmir"], "title_deed": attrs["Çıxarış"],
    })
    # Task A (our tree): residual on the log scale and absolute % error in AZN.
    log_pred = preds[f"a_{_slug(names_tree['A'])}_log_pred"]
    out["a_pred_price"] = np.exp(log_pred)
    out["a_res_log"] = log_pred - preds["log_price"]
    out["a_abs_log"] = out["a_res_log"].abs()
    out["a_ape"] = (out["a_pred_price"] - out["price"]).abs() / out["price"]
    # Task B: misclassification indicators.
    out["b_err_tree"] = preds[f"b_{_slug(names_tree['B'])}_label"] != preds["tier"]
    out["b_err_svm"] = preds[f"b_{_slug(name_svm)}_label"] != preds["tier"]

    # District size = TRAINING rows of that district.
    counts = train["location"].value_counts()
    n_train = out["district"].map(counts).fillna(0)
    out["district_train_rows"] = n_train.astype(int)
    out["district_bin"] = pd.Categorical(
        [next(label for lo, hi, label in DISTRICT_BINS if lo <= n <= hi) for n in n_train],
        categories=[label for *_, label in DISTRICT_BINS], ordered=True)
    # Price deciles with edges from TRAINING prices.
    edges = np.quantile(train["price"], np.linspace(0.1, 0.9, 9))
    bounds = np.r_[0, edges, np.inf]
    labels = [f"D{i + 1} ({_k(bounds[i])}-{_k(bounds[i + 1])})" for i in range(10)]
    out["decile"] = pd.cut(out["price"], bounds, labels=labels, right=False)
    # Distance from the tier threshold, in % of the threshold.
    distance = (out["price"] / threshold - 1).abs() * 100
    dist_labels = [f"{a}-{b}%" if np.isfinite(b) else f">= {a}%" for a, b in zip(DISTANCE_BINS, DISTANCE_BINS[1:])]
    out["threshold_distance"] = pd.cut(distance, DISTANCE_BINS, labels=dist_labels, right=False)
    out["top_1pct"] = np.where(out["price"] >= out["price"].quantile(0.99), "Top 1% by price", "Other 99%")
    return out


def _k(value):
    if not np.isfinite(value):
        return "max"
    return "0" if value == 0 else f"{value / 1000:,.0f}k"


def errors_by(frame, key):
    g = frame.groupby(key, observed=True, sort=True)
    table = pd.DataFrame({
        "n test": g.size(),
        "A MdAPE %": g["a_ape"].median() * 100,
        "A MAE log": g["a_abs_log"].mean(),
        "A median bias %": (np.exp(g["a_res_log"].median()) - 1) * 100,
        "B premium %": g["tier"].mean() * 100,
        "B error % tree": g["b_err_tree"].mean() * 100,
        "B error % SVM": g["b_err_svm"].mean() * 100,
    })
    table.index.name = key
    return table


def plot_error_breakdowns(frame):
    paths = []
    dec = errors_by(frame, "decile")
    fig, axes = _figure(2, size=(13.0, 4.6))
    x = np.arange(len(dec))
    axes[0].bar(x, dec["A MdAPE %"], color=SLOTS[0], width=0.7)
    _label(axes[0], "Task A, our tree: median absolute % error", None, "Median |error| / price, %")
    for ax in axes:
        ax.set_xticks(x, [label.split(" ")[0] for label in dec.index])
        ax.set_xlabel("True-price decile (edges from training prices)", color=INK)
    for i, (col, label) in enumerate((("B error % tree", "Our tree (gini)"),
                                      ("B error % SVM", "Our linear Pegasos SVM"))):
        axes[1].plot(x, dec[col], color=SLOTS[i], linewidth=2, marker="os"[i], markersize=7,
                     linestyle="-", label=label)
    _label(axes[1], "Task B: misclassification rate", None, "Error rate, %")
    axes[1].legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK)
    paths += _save(fig, "analysis_error_by_price_decile", "Errors by true-price decile (test set)",
                   "Deciles: " + "; ".join(dec.index))

    dist = errors_by(frame, "threshold_distance")
    fig, (ax,) = _figure(size=(7.0, 4.4))
    x = np.arange(len(dist))
    for i, (col, label) in enumerate((("B error % tree", "Our tree (gini)"),
                                      ("B error % SVM", "Our linear Pegasos SVM"))):
        ax.plot(x, dist[col], color=SLOTS[i], linewidth=2, marker="os"[i], markersize=7, label=label)
    ax.set_xticks(x, [f"{d}\n(n={n:,})" for d, n in zip(dist.index, dist["n test"])])
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK)
    _label(ax, None, "|price / threshold - 1|", "Error rate, %")
    paths += _save(fig, "analysis_error_vs_threshold_distance",
                   "Task B: error rate vs distance from the tier threshold (test set)")
    return paths


def worst_listings(frame, n=10):
    cols = ["clean_index", "category", "city", "district", "area_m2", "rooms", "floor", "total_floors",
            "land_sot", "repair", "title_deed", "price", "a_pred_price"]
    worst = frame.nlargest(n, "a_abs_log")[cols].copy()
    worst["pred / true"] = worst["a_pred_price"] / worst["price"]
    return worst.rename(columns={"a_pred_price": "predicted"})


# --- Markdown ---------------------------------------------------------------------
def _md(frame, index=True, digits=2):
    frame = frame.reset_index() if index else frame
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{digits}f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "|" + "|".join("---" for _ in frame.columns) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


GLOSSARY = {"Sahə": "floor area", "Otaq sayı": "rooms", "has_floor": "being an apartment",
            "floor": "floor", "total_floors": "building height", "land_area_sot": "land area",
            "lat": "latitude", "lng": "longitude", "views": "page views"}
# Suffixes of bina.az location names: q. = qəsəbə (settlement), m. = metro
# station area, r. = rayon (district).
LOCATION_SUFFIX = {" q.": " settlement", " m.": " metro area", " r.": " district"}


def _gloss(feature):
    """Readable English name for a feature in the findings text."""
    if feature in GLOSSARY:
        return f"{GLOSSARY[feature]} ({feature})"
    if feature.startswith("location_"):
        place = feature[len("location_"):]
        for suffix, word in LOCATION_SUFFIX.items():
            if place.endswith(suffix):
                place = place[: -len(suffix)] + word
        return f"being in {place}"
    return feature


def matched_baselines(model, models):
    """The scikit-learn rows benchmarked against one of our models."""
    if model.startswith("Tree"):
        return [model.replace("(ours)", "(sklearn)")]
    family = ("linear",) if "linear" in model else ("RFF", "RBF")
    # "sklearn" without parentheses: the SVC rows read "(sklearn, max 10k rows)".
    return [m for m in models if "sklearn" in m and not m.startswith("Tree")
            and any(f in m for f in family)]


def findings(frame, tables, tree_cmp, svm_stats, top_weights, names_tree, name_svm):
    """Bullet findings, every number computed from this run."""
    test_a, test_b = tables[("test", "A")], tables[("test", "B")]
    tree_a, tree_b = test_a[names_tree["A"]], test_b[names_tree["B"]]
    svm_b = test_b[name_svm]
    best_b = max(test_b, key=lambda m: test_b[m]["F1"])
    out = []

    # Each of our Task B models vs the best of the scikit-learn baselines matched to it.
    gaps = {m: test_b[m]["F1"] - max(test_b[b]["F1"] for b in matched_baselines(m, test_b))
            for m in test_b if "(ours)" in m}
    worst_gap = min(gaps, key=gaps.get)
    out.append(
        f"Overall: our Task A tree reaches R² {tree_a['R2 log']:.3f} on log price "
        f"(R² {tree_a['R2 AZN']:.3f} in AZN, median absolute error {frame['a_ape'].median():.1%}). "
        f"On Task B our gini tree gets F1 {tree_b['F1']:.3f} / ROC-AUC {tree_b['ROC-AUC']:.3f} and the "
        f"final linear Pegasos SVM F1 {svm_b['F1']:.3f} / ROC-AUC {svm_b['ROC-AUC']:.3f}; the highest test "
        f"F1 is {test_b[best_b]['F1']:.3f} ({best_b}). Each of our models is within "
        f"{abs(min(gaps.values())):.3f} F1 of its best matched scikit-learn baseline "
        f"(largest gap: {worst_gap}, {gaps[worst_gap]:+.4f}); trees beat the SVMs by about "
        f"{tree_b['F1'] - svm_b['F1']:.2f} F1.")

    d = errors_by(frame, "district_bin")
    sparse = d.loc[[b for b in ("0 (unseen in train)", "1-49") if b in d.index]]
    sparse_n = int(sparse["n test"].sum())
    sparse_ape = frame.loc[frame["district_bin"].isin(sparse.index), "a_ape"].median()
    sparse_err = frame.loc[frame["district_bin"].isin(sparse.index), "b_err_tree"].mean()
    dense = d.loc[">= 1000"]
    unseen = int(d["n test"].get("0 (unseen in train)", 0))
    out.append(
        f"Sparse districts do worse: listings in districts with fewer than 50 training rows "
        f"(n = {sparse_n} test listings, {unseen} of them from districts never seen in training) "
        f"have a median absolute error of {sparse_ape:.1%} vs "
        f"{dense['A MdAPE %']:.1f}% in districts with 1,000+ training rows, and a Task B tree error "
        f"rate of {sparse_err:.1%} vs {dense['B error % tree']:.1f}%.")

    dec = errors_by(frame, "decile")
    lo, hi = dec.iloc[0], dec.iloc[-1]
    mid = frame.loc[frame["decile"].isin(dec.index[3:7]), "a_ape"].median() * 100
    out.append(
        f"Errors grow at both price extremes and are biased towards the middle: the cheapest decile has "
        f"median absolute error {lo['A MdAPE %']:.1f}% (median bias {lo['A median bias %']:+.1f}%), the most "
        f"expensive {hi['A MdAPE %']:.1f}% (bias {hi['A median bias %']:+.1f}%), vs {mid:.1f}% for deciles 4-7. "
        "A depth-limited tree predicts leaf means, so extreme prices are pulled towards their leaf's average.")

    top = errors_by(frame, "top_1pct")
    t = top.loc["Top 1% by price"]
    out.append(
        f"The top 1% most expensive listings (n = {int(t['n test'])}, price >= "
        f"{frame['price'].quantile(0.99):,.0f} AZN) have median absolute error {t['A MdAPE %']:.1f}% and "
        f"median bias {t['A median bias %']:+.1f}%; they are rare in training, so the tree has few leaves for them. "
        f"For Task B they are easy (tree error {t['B error % tree']:.1f}%), being far above the threshold.")

    dist = errors_by(frame, "threshold_distance")
    near, far = dist.iloc[0], dist.iloc[-1]
    out.append(
        f"Task B errors concentrate at the tier boundary: within 5% of the {frame.attrs['threshold']:,.0f} AZN "
        f"threshold the tree misclassifies {near['B error % tree']:.1f}% and the SVM {near['B error % SVM']:.1f}% "
        f"of listings, vs {far['B error % tree']:.1f}% / {far['B error % SVM']:.1f}% when the price is 40%+ away. "
        f"{(frame['b_err_tree'] & (frame['threshold_distance'] == dist.index[0])).sum() / frame['b_err_tree'].sum():.0%} "
        "of all tree errors are listings within 5% of the threshold.")

    cat = errors_by(frame, "category")
    large = cat[cat["n test"] >= 100]           # don't rank on a handful of listings
    worst_cat = large["A MdAPE %"].idxmax()
    small = cat[cat["n test"] < 100]
    out.append(
        "By category, median absolute error is "
        + ", ".join(f"{c} {r['A MdAPE %']:.1f}% (n = {int(r['n test'])})" for c, r in cat.iterrows())
        + f". Among categories with 100+ test listings, {worst_cat} is hardest to price"
        + ("" if small.empty else
           "; " + ", ".join(f"{c} ({int(r['n test'])} listings)" for c, r in small.iterrows())
           + " is too small a group to rank reliably")
        + ".")

    pos = ", ".join(_gloss(n) for n, _ in top_weights["positive"][:3])
    neg = ", ".join(_gloss(n) for n, _ in top_weights["negative"][:3])
    out.append(
        f"The final linear SVM (||w|| = {svm_stats['weight_norm']:.2f}, margin 2/||w|| = "
        f"{svm_stats['margin_width']:.3f} in standardized units, {svm_stats['support_count']:,} training points "
        f"on or inside the margin) pushes towards premium mainly through {pos}, and towards standard through {neg}.")

    a = tree_cmp.loc["Task A tree"]
    out.append(
        f"Our trees and scikit-learn's are practically the same model: feature-importance correlation "
        f"at least {np.floor(tree_cmp['importance corr'].min() * 1e4) / 1e4:.4f}; Task A predictions correlate "
        f"{a['pred corr']:.4f}, differ by "
        f"{a['mean |diff| log']:.4f} log units on average, and {a['within 0.01 log %']:.1f}% are within 0.01 "
        f"(about 1%); Task B labels agree on {tree_cmp['test agreement %'].dropna().min():.1f}% or more of test listings.")
    return out


def write_markdown(path, frame, tables, svm_stats, worst, bullets, threshold):
    sections = [
        "# Error analysis (test set)",
        f"Generated by `python -m src.analysis` from the benchmark outputs. Test listings: {len(frame):,}. "
        f"Tier threshold (training median): {threshold:,.0f} AZN. Task A errors are for our tree "
        "(MdAPE = median |predicted - true| / true; bias = median of exp(log residual) - 1). Task B error "
        "rates are for our gini tree and the final linear Pegasos SVM.",
        "## Findings", "\n".join(f"- {b}" for b in bullets),
        "## By category", _md(errors_by(frame, "category")),
        "## By city", _md(errors_by(frame, "city")),
        "## By district size (training rows per district)", _md(errors_by(frame, "district_bin")),
        "## By true-price decile (edges from training prices)", _md(errors_by(frame, "decile")),
        "## Top 1% most expensive listings", _md(errors_by(frame, "top_1pct")),
        "## Task B: distance from the tier threshold", _md(errors_by(frame, "threshold_distance")),
        "## Final linear SVM",
        f"||w|| = {svm_stats['weight_norm']:.4f}; margin width 2/||w|| = {svm_stats['margin_width']:.4f} "
        f"(standardized feature units); support vectors (training points with margin <= 1): "
        f"{svm_stats['support_count']:,} of {svm_stats['n_train']:,}. Refit check: max |score difference| vs "
        f"the benchmark's saved test scores = {svm_stats['score_check']:.2e}.",
        "## 10 worst Task A listings (largest |log error|)",
        "Key listing attributes only; no names, contact details or coordinates.",
        _md(worst, index=False),
    ]
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


# --- Entry point ------------------------------------------------------------------
def run_analysis(quick: bool = False) -> dict:
    start = time.perf_counter()
    _set_paths(quick)
    for key in ("figures", "tables", "errors"):
        PATHS[key].mkdir(parents=True, exist_ok=True)
    meta, tables, preds, importances, tree_cmp, svm_cfg = load_inputs(quick)
    threshold = meta["test"]["tier_threshold_train_median"]
    final_kernel = svm_cfg["final"]
    name_svm = f"Pegasos {'linear' if final_kernel == 'linear' else 'RFF'} (ours)"
    names_tree = {"A": "Tree (ours)", "B": "Tree gini (ours)"}

    # Listing attributes; check the rebuilt split matches the benchmark's.
    with contextlib.redirect_stdout(io.StringIO()):
        df = data_prep.clean(data_prep.load_raw(data_prep.DATA_PATH))
        X, price, names = data_prep.make_features(df)
    train_ids, test_ids = rebuild_split(df, X, price)
    if not np.array_equal(df.index[test_ids], preds["clean_index"]):
        raise RuntimeError("rebuilt split does not match test_predictions.csv; re-run the benchmark")
    print(f"Loaded benchmark outputs: {len(preds):,} test listings, threshold {threshold:,.0f} AZN, "
          f"final SVM kernel = {final_kernel}")

    # 1. Main results table.
    main_results_table(tables)
    print(f"Wrote {PATHS['tables'].relative_to(ROOT)}/results_main.{{csv,tex}}")

    # 2. SVM weights: refit the final LINEAR config (benchmark does not save w).
    cfg = dict(svm_cfg["linear"])
    if cfg["gamma"] == "scale":
        cfg["gamma"] = 1.0 / X.shape[1]
    X_tr, X_te = X[train_ids], X[test_ids]
    if not (np.isfinite(X_tr).all() and np.isfinite(X_te).all()):
        raise ValueError("NaN/inf in features before standardize")
    X_tr_s, X_te_s = data_prep.standardize(X_tr, X_te)
    with contextlib.redirect_stdout(io.StringIO()):
        y_tr, _ = data_prep.make_tier_label(price[train_ids], threshold)
    svm = PegasosSVM(random_state=SEED, history_every=len(X_tr), **cfg).fit(X_tr_s, y_tr)
    saved = preds["b_pegasos_linear_ours_score"].to_numpy()
    svm_stats = {"weight_norm": svm.weight_norm_, "margin_width": svm.margin_width_,
                 "support_count": len(svm.support_), "n_train": len(X_tr),
                 "score_check": float(np.max(np.abs(svm.decision_function(X_te_s) - saved)))}
    order = np.argsort(svm.w)
    top_weights = {"positive": [(names[j], float(svm.w[j])) for j in order[::-1][:15]],
                   "negative": [(names[j], float(svm.w[j])) for j in order[:15]]}
    pd.DataFrame({"feature": names, "weight": svm.w}).sort_values("weight").to_csv(
        PATHS["errors"] / "svm_linear_weights.csv", index=False)
    # The quick benchmark trains on a 5k subsample, so its scores differ from this full-split refit.
    check_note = " (quick run: benchmark SVM was fit on a 5k subsample, so no match expected)" if quick else ""
    print(f"Linear SVM refit: ||w|| {svm_stats['weight_norm']:.4f}, margin {svm_stats['margin_width']:.4f}, "
          f"{svm_stats['support_count']:,} support vectors; max |score - saved| = "
          f"{svm_stats['score_check']:.1e}{check_note}")

    # 3. Figures.
    frame = analysis_frame(df, preds, train_ids, names_tree, name_svm, threshold)
    frame.attrs["threshold"] = threshold
    y = preds["tier"].to_numpy()
    figures = []
    figures += plot_confusion(y, {names_tree["B"]: preds[f"b_{_slug(names_tree['B'])}_label"].to_numpy(),
                                  name_svm: preds[f"b_{_slug(name_svm)}_label"].to_numpy()})
    figures += plot_roc_pr(y, preds, list(tables[("test", "B")]))
    figures += plot_task_a(frame)
    figures += plot_importances(importances)
    figures += plot_svm_weights(svm.w, names, svm_stats)
    figures += plot_error_breakdowns(frame)

    # 4. Error tables, worst listings, findings.
    for key in ("category", "city", "district_bin", "decile", "top_1pct", "threshold_distance"):
        errors_by(frame, key).to_csv(PATHS["errors"] / f"errors_by_{key}.csv")
    worst = worst_listings(frame)
    worst.to_csv(PATHS["errors"] / "worst_task_a_listings.csv", index=False)
    bullets = findings(frame, tables, tree_cmp, svm_stats, top_weights, names_tree, name_svm)
    markdown = PATHS["results"] / "error_analysis.md"
    write_markdown(markdown, frame, tables, svm_stats, worst, bullets, threshold)

    print("Wrote " + ", ".join(sorted({p.stem for p in figures}))
          + f" (.png/.pdf) in {PATHS['figures'].relative_to(ROOT)}/")
    print(f"Wrote {markdown.relative_to(ROOT)} and {PATHS['errors'].relative_to(ROOT)}/*.csv")
    print("\nFindings:\n" + "\n".join(f"- {b}" for b in bullets))
    print(f"\nDone in {time.perf_counter() - start:.1f}s")
    return {"findings": bullets, "figures": figures, "markdown": markdown}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Results table, figures and error analysis.")
    parser.add_argument("--quick", action="store_true",
                        help="analyse the quick benchmark outputs in results/quick")
    run_analysis(parser.parse_args().quick)
