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
    report/tables/results_headline.{csv,tex}     headline metrics, all models x both tasks
    report/tables/results_task_{a,b}.{csv,tex}   full results per task, test + CV (appendix)
    report/figures/*.{png,pdf}                   see the plot_* functions
    results/error_analysis.md                    tables + findings for the report
    results/error_analysis/*.csv                 every error table
    results/report_facts.csv                     setup constants and data facts quoted in the report
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
from .splits import get_splits
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
# Figures are drawn at the size they are printed in the two-column report
# (one column 3.5 in, full width 7 in), so the font sizes below are the
# printed sizes; LaTeX scales 3.5 in to the 3.33 in column (x0.95).
SINGLE, DOUBLE = 3.5, 7.0
FONT, SMALL = 9, 8.5

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


# --- Results tables -----------------------------------------------------------
# Full results, one table per task (report appendix): (benchmark metric,
# column header, scale factor, decimals). AZN errors are in thousands of AZN.
A_METRICS = [("RMSE log", "RMSE (log)", 1, 3), ("MAE log", "MAE (log)", 1, 3), ("R2 log", "$R^2$ (log)", 1, 3),
             ("RMSE AZN", "RMSE (kAZN)", 1e-3, 1), ("MAE AZN", "MAE (kAZN)", 1e-3, 1),
             ("R2 AZN", "$R^2$ (AZN)", 1, 3)]
B_METRICS = [("Precision", "Precision", 1, 3), ("Recall", "Recall", 1, 3),
             ("F1", "F1", 1, 3), ("ROC-AUC", "ROC-AUC", 1, 3)]
# Headline table (report body): (task, "test" | "cv", benchmark metric, header).
HEADLINE = [("A", "test", "RMSE log", "RMSE (log)"), ("A", "test", "R2 log", "$R^2$ (log)"),
            ("A", "cv", "R2 log", "$R^2$ (log), CV"), ("A", "test", "R2 AZN", "$R^2$ (AZN)"),
            ("B", "test", "F1", "F1"), ("B", "cv", "F1", "F1, CV"), ("B", "test", "ROC-AUC", "ROC-AUC")]
SVC_CAP_NOTE = ", max 10k rows"          # in the SVC model names; a footnote mark in the tables


def _cell(value, decimals, factor=1, pm=r" $\pm$ "):
    """A table cell: 'n/a', a number, or a (mean, std) pair as 'mean ± std'."""
    if value is None:
        return "n/a"
    if isinstance(value, tuple):
        return f"{value[0] * factor:.{decimals}f}{pm}{value[1] * factor:.{decimals}f}"
    return f"{value * factor:.{decimals}f}"


def _tex_model(model):
    """Model name for LaTeX: the SVC row-cap note becomes a footnote mark."""
    return ev._latex_escape(model.replace(SVC_CAP_NOTE, "")) + (r"$^{*}$" if SVC_CAP_NOTE in model else "")


def _write_tables(stem, colspec, header_tex, rows_tex, header_csv, rows_csv):
    """Write one table as booktabs LaTeX (a bare tabular) and as CSV."""
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *header_tex, r"\midrule",
             *(" & ".join(row) + r" \\" for row in rows_tex), r"\bottomrule", r"\end{tabular}"]
    (PATHS["tables"] / f"{stem}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(rows_csv, columns=header_csv).to_csv(PATHS["tables"] / f"{stem}.csv", index=False)


def headline_table(tables):
    """
    The report's main table: every model on both tasks, test values plus the
    CV mean ± std of R^2 (log) and F1. A model only has its own task's
    metrics; the other task's cells are n/a.
    """
    rows_tex, rows_csv = [], []
    for task in ("A", "B"):
        for model in tables[("test", task)]:
            values = [tables[(kind, t)][model][metric] if t == task else None
                      for t, kind, metric, _ in HEADLINE]
            rows_tex.append([_tex_model(model), *(_cell(v, 3) for v in values)])
            rows_csv.append([model, *(_cell(v, 3, pm=" ± ") for v in values)])
    n_a = sum(t == "A" for t, *_ in HEADLINE)
    header_tex = [
        rf" & \multicolumn{{{n_a}}}{{c}}{{Task A: log price}}"
        rf" & \multicolumn{{{len(HEADLINE) - n_a}}}{{c}}{{Task B: premium tier}} \\",
        rf"\cmidrule(lr){{2-{n_a + 1}}}\cmidrule(lr){{{n_a + 2}-{len(HEADLINE) + 1}}}",
        "Model & " + " & ".join(h for *_, h in HEADLINE) + r" \\",
    ]
    header_csv = ["model", *(f"{t} {h.replace('$R^2$', 'R2')}" for t, _, _, h in HEADLINE)]
    _write_tables("results_headline", "l" + "r" * len(HEADLINE), header_tex, rows_tex, header_csv, rows_csv)


def full_results_tables(tables):
    """Appendix tables, one per task: a test row and a CV row (mean ± std) per model."""
    for task, specs in (("A", A_METRICS), ("B", B_METRICS)):
        rows_tex, rows_csv = [], []
        for model in tables[("test", task)]:
            for kind, label in (("test", "test"), ("cv", "CV")):
                values = tables[(kind, task)][model]
                cells = [_cell(values[m], d, k) for m, _, k, d in specs]
                rows_tex.append([_tex_model(model) if kind == "test" else "", label, *cells])
                rows_csv.append([model, label, *(_cell(values[m], d, k, pm=" ± ") for m, _, k, d in specs)])
        header_tex = ["Model & Eval. & " + " & ".join(h for _, h, _, _ in specs) + r" \\"]
        header_csv = ["model", "eval", *(h.replace("$R^2$", "R2") for _, h, _, _ in specs)]
        _write_tables(f"results_task_{task.lower()}", "ll" + "r" * len(specs),
                      header_tex, rows_tex, header_csv, rows_csv)


# --- Figure helpers -----------------------------------------------------------
def _figure(ncols=1, size=(SINGLE, 2.6), **kwargs):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # Constrained layout keeps every label inside the figure, so the saved
    # file is exactly `size` and prints at that size without rescaling fonts.
    fig = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(1, ncols, **kwargs))
    for ax in axes:
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=SMALL)
    return fig, axes


def _label(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=FONT, color=INK, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=FONT)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=FONT)


def _legend(ax, loc, **kwargs):
    ax.legend(loc=loc, frameon=False, fontsize=SMALL, labelcolor=INK, **kwargs)


def _save(fig, stem):
    """Save at the drawn size (no tight bbox); titles and notes live in the report captions."""
    paths = []
    for ext in ("png", "pdf"):
        path = PATHS["figures"] / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
        paths.append(path)
    return paths


# Short English labels for feature names on figure axes.
FEATURE_PREFIX = {"location_": "", "city_": "city: ", "Kateqoriya_": "category: ",
                  "Binanın növü_": "building type: ", "Təmir_": "renovated: ", "Çıxarış_": "title deed: ",
                  "İpoteka_": "mortgage: ", "products_label_": "label: "}


VALUE_EN = {**CATEGORY_EN, "var": "yes", "yoxdur": "no"}


def _feature_label(name):
    if name in GLOSSARY:
        return GLOSSARY[name]
    for prefix, english in FEATURE_PREFIX.items():
        if name.startswith(prefix):
            value = name[len(prefix):]
            return english + VALUE_EN.get(value, value)
    return name


def _blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)


# --- Task B figures -------------------------------------------------------------
def plot_confusion(y, labels_by_model):
    fig, axes = _figure(len(labels_by_model), size=(SINGLE, 2.0), sharey=True)
    names = ["Standard", "Premium"]
    for ax, (model, labels) in zip(axes, labels_by_model.items()):
        cm = ev.confusion_matrix(y, labels)
        rates = cm / cm.sum(axis=1, keepdims=True)          # share of each TRUE class
        ax.grid(False)
        ax.imshow(rates, cmap=_blue_cmap(), vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{rates[i, j]:.1%}", ha="center", va="center",
                        fontsize=SMALL, color="white" if rates[i, j] > 0.5 else INK)
        ax.set_xticks([0, 1], names)
        ax.set_yticks([0, 1], names)
        accuracy = np.trace(cm) / cm.sum()
        ax.set_title(f"{model}\naccuracy {accuracy:.1%}", fontsize=SMALL, color=INK)
        _label(ax, None, "Predicted tier", "True tier" if ax is axes[0] else None)
    return _save(fig, "analysis_confusion_matrices")


def _task_b_groups(models):
    return [("Trees", [m for m in models if m.startswith("Tree")]),
            ("Linear SVMs", [m for m in models if "linear" in m]),
            ("RFF / RBF SVMs", [m for m in models if "RFF" in m or "RBF" in m])]


def _short(model):
    """Legend name inside a family panel: the family word is in the panel title."""
    for word in ("Tree ", " linear", " RFF", " RBF"):
        model = model.replace(word, "")
    model = model.replace(SVC_CAP_NOTE, "").replace("SGD hinge", "SGD")
    return model[0].upper() + model[1:]


def plot_roc_pr(y, preds, models):
    """One ROC figure and one PR figure; panels by model family (<= 4 series each)."""
    paths = []
    for kind in ("roc", "pr"):
        fig, axes = _figure(3, size=(DOUBLE, 2.6), sharey=True)
        for ax, (family, members) in zip(axes, _task_b_groups(models)):
            for i, model in enumerate(members):
                scores = preds[f"b_{_slug(model)}_score"].to_numpy()
                labels = preds[f"b_{_slug(model)}_label"].to_numpy()
                style = dict(color=SLOTS[i], linewidth=1.5, linestyle="-" if "(ours)" in model else "--")
                if kind == "roc":
                    fpr, tpr, _ = ev.roc_curve(y, scores)
                    ax.plot(fpr, tpr, label=f"{_short(model)}  {ev.roc_auc(y, scores):.3f}", **style)
                else:
                    precision, recall, _ = ev.precision_recall_curve(y, scores)
                    p, r, _ = ev.precision_recall_f1(y, labels)
                    ax.plot(recall, precision, label=_short(model), **style)
                    # The model's own decision threshold (label predictions).
                    ax.scatter([r], [p], s=30, color=SLOTS[i], edgecolors="white", linewidths=1, zorder=5)
            if kind == "roc":
                # Zoomed on the top-left corner, where the curves differ; the
                # empty lower right then holds the legend (label = model, AUC).
                ax.set(xlim=(-0.005, 0.3), ylim=(0.3, 1.005))
                _label(ax, family, "False positive rate", "True positive rate" if ax is axes[0] else None)
                _legend(ax, "lower right", handlelength=1.5)
            else:
                ax.axhline(y.mean(), color=MUTED, linestyle=":", linewidth=1)
                ax.set_ylim(0.45, 1.01)
                _label(ax, family, "Recall (premium)", "Precision (premium)" if ax is axes[0] else None)
                _legend(ax, "lower left", handlelength=1.5)
        paths += _save(fig, "analysis_roc_curves" if kind == "roc" else "analysis_pr_curves")
    return paths


# --- Task A figures -------------------------------------------------------------
def plot_task_a(df):
    paths = []
    fig, (ax,) = _figure(size=(SINGLE, 3.0))
    hb = ax.hexbin(df["price"], df["a_pred_price"], xscale="log", yscale="log", gridsize=45,
                   bins="log", mincnt=1, cmap=_blue_cmap(), linewidths=0)
    lo, hi = df["price"].min(), df["price"].max()
    ax.plot([lo, hi], [lo, hi], color=INK, linestyle="--", linewidth=1, label="Perfect prediction")
    _legend(ax, "upper left")
    bar = fig.colorbar(hb, ax=ax)
    bar.set_label("Listings per cell (log)", fontsize=SMALL, color=INK)
    bar.ax.tick_params(labelsize=SMALL, colors=MUTED)
    _label(ax, None, "True price, AZN", "Predicted price, AZN")
    paths += _save(fig, "analysis_task_a_pred_vs_actual")

    fig, (ax,) = _figure(size=(SINGLE, 2.6))
    ax.hexbin(df["price"], df["a_res_log"], xscale="log", gridsize=45, bins="log", mincnt=1,
              cmap=_blue_cmap(), linewidths=0)
    ax.axhline(0, color=INK, linewidth=1)
    deciles = df.groupby("decile", observed=True).agg(price=("price", "median"), res=("a_res_log", "median"))
    ax.plot(deciles["price"], deciles["res"], color=SLOTS[1], linewidth=1.5, marker="o", markersize=4,
            label="Median per true-price decile")
    _legend(ax, "lower left")
    _label(ax, None, "True price, AZN", "log(predicted / true)")
    paths += _save(fig, "analysis_task_a_residuals")
    return paths


# --- Feature importance and SVM weights -------------------------------------------
def plot_importances(importances, top=15):
    fig, axes = _figure(2, size=(DOUBLE, 3.6))
    for ax, (task, ours, theirs) in zip(axes, (
            ("Task A (MSE)", "task_a_tree_ours", "task_a_tree_sklearn"),
            ("Task B (Gini)", "task_b_gini_ours", "task_b_gini_sklearn"))):
        rows = importances.nlargest(top, ours).iloc[::-1]
        y = np.arange(len(rows))
        ax.barh(y + 0.2, rows[ours], height=0.38, color=SLOTS[0], label="Our tree")
        ax.barh(y - 0.2, rows[theirs], height=0.38, color=SLOTS[1], label="scikit-learn tree")
        ax.set_yticks(y, [_feature_label(f) for f in rows["feature"]])
        ax.grid(axis="y", visible=False)
        _label(ax, task, "Mean decrease in impurity")
        _legend(ax, "lower right")
    return _save(fig, "analysis_feature_importance")


def plot_svm_weights(weights, names, top=15):
    order = np.argsort(weights)
    pick = np.r_[order[:top], order[-top:]]              # 15 most negative, 15 most positive
    fig, (ax,) = _figure(size=(SINGLE, 4.8))
    y = np.arange(len(pick))
    colors = [DIVERGING["positive"] if weights[j] > 0 else DIVERGING["negative"] for j in pick]
    ax.barh(y, weights[pick], color=colors, height=0.7)
    ax.set_yticks(y, [_feature_label(names[j]) for j in pick])
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    _label(ax, None, "Weight per +1 std of the feature\n(blue: premium, red: standard)")
    return _save(fig, "analysis_svm_weights")


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
    fig, axes = _figure(2, size=(DOUBLE, 2.4))
    x = np.arange(len(dec))
    axes[0].bar(x, dec["A MdAPE %"], color=SLOTS[0], width=0.7)
    _label(axes[0], "Task A, our tree", None, "Median |error| / price, %")
    for ax in axes:
        ax.set_xticks(x, [label.split(" ")[0] for label in dec.index])
        ax.set_xlabel("True-price decile (edges from training prices)", color=INK, fontsize=FONT)
    for i, (col, label) in enumerate((("B error % tree", "Our tree (Gini)"),
                                      ("B error % SVM", "Our linear Pegasos SVM"))):
        axes[1].plot(x, dec[col], color=SLOTS[i], linewidth=1.5, marker="os"[i], markersize=4,
                     linestyle="-", label=label)
    _label(axes[1], "Task B: misclassification rate", None, "Error rate, %")
    # Headroom above the peak holds the legend, so it never covers a line.
    axes[1].set_ylim(0, 1.4 * max(dec["B error % tree"].max(), dec["B error % SVM"].max()))
    _legend(axes[1], "upper right")
    paths += _save(fig, "analysis_error_by_price_decile")

    dist = errors_by(frame, "threshold_distance")
    fig, (ax,) = _figure(size=(SINGLE, 2.5))
    x = np.arange(len(dist))
    for i, (col, label) in enumerate((("B error % tree", "Our tree (Gini)"),
                                      ("B error % SVM", "Our linear Pegasos SVM"))):
        ax.plot(x, dist[col], color=SLOTS[i], linewidth=1.5, marker="os"[i], markersize=4, label=label)
    ax.set_xticks(x, [f"{d}\n(n={n:,})" for d, n in zip(dist.index, dist["n test"])])
    _legend(ax, "upper right")
    _label(ax, None, "|price / threshold - 1|", "Error rate, %")
    paths += _save(fig, "analysis_error_vs_threshold_distance")
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
            "lat": "latitude", "lng": "longitude", "views": "page views",
            "repair": "renovated (flag)", "bill_of_sale": "title deed (flag)",
            "mortgage": "mortgage (flag)", "vip": "VIP listing (flag)", "featured": "featured listing (flag)"}
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
        f"Sparse districts are harder to price: listings in districts with fewer than 50 training rows "
        f"(n = {sparse_n} test listings, {unseen} of them from districts never seen in training) "
        f"have a median absolute error of {sparse_ape:.1%} vs "
        f"{dense['A MdAPE %']:.1f}% in districts with 1,000+ training rows; their Task B tree error "
        f"rate is {sparse_err:.1%} vs {dense['B error % tree']:.1f}%.")

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


# --- Report facts -----------------------------------------------------------------
def _parse_number(series):
    """The string-to-number rule of data_prep.clean, step 4 (same three replacements)."""
    s = series.astype(str).str.replace("m²", "", regex=False)
    return pd.to_numeric(s.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")


def cleaning_counts(raw):
    """
    Rows removed by each row filter of data_prep.clean, in its order:
    (1) missing price, area or rooms, (2) exact duplicates, (3) price or
    area outside the allowed bounds. (1) is counted directly. For (2) and
    (3), clean() itself is run a second time on a copy whose price and area
    are mapped one-to-one (a linear rescale) into the allowed ranges: a
    one-to-one map keeps missing values missing and duplicate rows
    duplicate, so on that copy clean() removes exactly (1) and (2).
    """
    price, area, rooms = (_parse_number(raw[c]) for c in ("price", "Sahə", "Otaq sayı"))
    missing = int((price.isna() | area.isna() | rooms.isna()).sum())

    def into(values, lo, hi):
        return lo + (values - values.min()) / (values.max() - values.min()) * (hi - lo)

    inside = raw.copy()
    inside["price"] = into(price, 10_000, 5_000_000)        # the bounds in data_prep.clean, step 6
    inside["Sahə"] = into(area, 10, 2_000)
    with contextlib.redirect_stdout(io.StringIO()):
        after_duplicates = len(data_prep.clean(inside))
        final = len(data_prep.clean(raw))
    duplicates = len(raw) - missing - after_duplicates
    if duplicates < 0 or final > after_duplicates:
        raise RuntimeError("cleaning_counts no longer matches data_prep.clean")
    return {"missing": missing, "duplicates": duplicates, "outliers": after_duplicates - final,
            "rooms_missing_raw": int(rooms.isna().sum()), "area_unparsed_raw": int(area.isna().sum())}


def report_facts(meta, raw, df, X, names, price, split_ids, threshold):
    """
    Setup constants and data checks quoted in the report, as one key/value
    CSV, so every number in the text traces back to a regenerated file.
    """
    facts = []

    def add(key, value, description):
        facts.append({"key": key, "value": value, "description": description})

    sizes = meta["split_sizes"]
    train_ids = split_ids["train"]
    # Raw data and cleaning (data_prep.load_raw / clean / make_features).
    add("raw_rows", raw.shape[0], "rows of the raw CSV")
    add("raw_columns", raw.shape[1], "columns of the raw CSV")
    currency = [c for c in ("currency_x", "currency_y") if c in raw.columns]
    add("raw_rows_currency_not_azn", int((raw[currency] != "AZN").any(axis=1).sum()),
        "raw rows whose currency columns are not AZN")
    dropped = [c for c in data_prep.LEAKAGE_COLUMNS if c in raw.columns]
    add("dropped_columns", len(dropped), "raw columns removed by data_prep.LEAKAGE_COLUMNS")
    add("raw_columns_kept", raw.shape[1] - len(dropped), "raw columns kept (before parsing)")
    counts = cleaning_counts(raw)
    add("rows_removed_missing", counts["missing"], "rows removed: price, area or rooms missing after parsing")
    add("rows_removed_duplicates", counts["duplicates"], "rows removed: exact duplicates")
    add("rows_removed_outliers", counts["outliers"], "rows removed: price or area outside the bounds")
    add("raw_rows_rooms_missing", counts["rooms_missing_raw"], "raw rows without a room count")
    add("raw_rows_area_unparsed", counts["area_unparsed_raw"], "raw rows whose area does not parse")
    add("rows_clean", len(df), "listings after data_prep.clean")
    add("n_features", X.shape[1], "columns of the design matrix (after one-hot encoding)")
    # A feature named like a cleaned column is either numeric as cleaned, or a
    # text column that make_features turned into a 0/1 presence flag.
    same_name = [n for n in names if n in df.columns]
    numeric = [n for n in same_name if pd.api.types.is_numeric_dtype(df[n])]
    flags = [n for n in same_name if n not in numeric]
    add("n_numeric_features", len(numeric), "numeric features used as they are: " + ", ".join(numeric))
    add("n_flag_features", len(flags), "presence flags (1 = value present, 0 = missing): " + ", ".join(flags))
    for flag in flags:
        add(f"flag_share_{flag}", float(X[:, names.index(flag)].mean()), f"share of 1s in {flag}, all cleaned rows")
    # Three flags restate a "yes" of another column; count the rows where they disagree.
    for flag, column in (("repair", "Təmir"), ("bill_of_sale", "Çıxarış"), ("mortgage", "İpoteka")):
        if flag in flags and column in df.columns:
            add(f"flag_mismatch_{flag}", int((df[flag].notna() != (df[column] == "var")).sum()),
                f"rows where {flag} present != ({column} == 'var')")
    text_cols = [c for c in df.columns if c != "price" and c not in flags
                 and not pd.api.types.is_numeric_dtype(df[c])]
    by_source = {c: sum(n.startswith(c + "_") for n in names) for c in text_cols}
    add("n_onehot_features", sum(by_source.values()), "one-hot columns: "
        + ", ".join(f"{c} {k} (of {df[c].nunique()} values)" for c, k in by_source.items()))
    for split, ids in split_ids.items():
        add(f"premium_share_{split}", float(np.mean(price[ids] >= threshold)),
            f"share of premium listings in the {split} split (training-median threshold)")
    for split in ("train", "val", "test"):
        add(f"n_{split}", sizes[split], f"{split} split size")
    add("seed", meta["seed"], "data_prep.SEED, used everywhere")
    add("tier_threshold_train_median", threshold, "tier threshold: median training price (AZN)")
    full_median = float(np.median(price))
    add("full_data_median", full_median, "median price over ALL rows (not used for labels)")
    add("labels_differ_full_vs_train_median",
        int(((price >= full_median) != (price >= threshold)).sum()),
        "rows whose tier would change if the full-data median were used")
    add("cv_rows", meta["cv"]["n_rows"], "stratified CV sample from train+val")
    add("cv_pool_rows", sizes["train"] + sizes["val"], "train+val rows the CV sample is drawn from")
    add("cv_folds", meta["cv"]["n_folds"], "StratifiedKFold splits")
    add("cv_fit_rows_per_fold", meta["cv"]["n_rows"] * (meta["cv"]["n_folds"] - 1) // meta["cv"]["n_folds"],
        "training rows per CV fold")
    add("svc_max_rows", max(meta["cv"]["svc_rows"]), "rows exact-kernel SVC is fit on (cap)")
    add("max_tree_depth_cap", meta["max_tree_depth"], "hard cap on tree max_depth")
    for task in ("tree_task_a", "tree_task_b"):
        for key, value in meta["configs"][task].items():
            add(f"{task}_{key}", value, f"{task} setting")
        add(f"{task}_source", meta["config_sources"][task], f"where the {task} config came from")
    for kernel in ("linear", "rff"):
        for key in ("lambda_", "n_components", "n_epochs"):
            add(f"svm_{kernel}_{key}", meta["configs"]["svm"][kernel][key], f"{kernel} Pegasos {key}")
        add(f"svm_{kernel}_gamma", meta["svm_gamma_resolved"][kernel], f"{kernel} Pegasos gamma (resolved)")
    add("svm_final_kernel", meta["configs"]["svm"]["final"], "final SVM kernel (higher validation F1)")
    n_components = meta["configs"]["svm"]["rff"]["n_components"]
    add("kernel_matrix_gb", sizes["train"] ** 2 * 8 / 1e9,
        "memory of an exact float64 kernel matrix on the training split, GB")
    add("rff_matrix_mb", sizes["train"] * n_components * 8 / 1e6,
        "memory of the float64 RFF training matrix, MB")
    tuning_path = PATHS["results"] / "svm_final_config.json"
    if tuning_path.exists():
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        space = tuning["search_space"]
        add("svm_grid_lambda_min", min(space["lambdas"]), "smallest lambda in the SVM grid")
        add("svm_grid_lambda_max", max(space["lambdas"]), "largest lambda in the SVM grid")
        add("svm_grid_n_lambdas", len(space["lambdas"]), "lambda values per kernel")
        add("svm_f1_tolerance", space.get("f1_tolerance"),
            "selection: largest lambda within this validation F1 of the best")
        add("svm_grid_gamma_multipliers", " ".join(f"{m:g}" for m in space["gamma_multipliers_of_scale"]),
            "RFF gamma values, as multiples of 1/n_features")
        for kernel in ("linear", "rff"):
            sel = tuning["selection"][kernel]
            add(f"svm_{kernel}_val_f1", sel["validation_f1"], f"validation F1 of the chosen {kernel} config")
            add(f"svm_{kernel}_val_auc", sel["validation_roc_auc"], f"validation ROC-AUC, chosen {kernel} config")
            add(f"svm_{kernel}_max_f1", sel["max_f1"], f"highest validation F1 in the {kernel} sweep")
            add(f"svm_{kernel}_max_f1_lambda", sel["max_f1_config"]["lambda_"],
                f"lambda of the highest-F1 {kernel} config")
    # Data checks for the Limitations section.
    zero_cols = [names[j] for j in range(X.shape[1])
                 if set(np.unique(X[:, j])) <= {0.0, 1.0} and not X[train_ids, j].any()]
    add("onehot_all_zero_in_train", len(zero_cols), "one-hot columns with no training rows: " + ", ".join(zero_cols))
    add("rows_no_floor_filled_0", int((df["has_floor"] == 0).sum()),
        "rows with no floor in the raw data (floor, total_floors set to 0, has_floor = 0)")
    for raw, label in (("Həyət evi/Bağ evi", "house"), ("Ofis", "office")):
        add(f"rows_no_floor_{label}", int(((df["has_floor"] == 0) & (df["Kateqoriya"] == raw)).sum()),
            f"of which {label} listings")
    add("rows_land_area_filled_0", int((df["land_area_sot"] == 0).sum()),
        "rows with no land area in the raw data (land_area_sot set to 0)")
    frame = pd.DataFrame(facts)
    frame.to_csv(PATHS["results"] / "report_facts.csv", index=False)
    return frame


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
        raw = data_prep.load_raw(data_prep.DATA_PATH)
        df = data_prep.clean(raw)
        X, price, names = data_prep.make_features(df)
    split = get_splits(X, price)                  # the benchmark's split (src/splits.py)
    train_ids, val_ids, test_ids = split.train, split.val, split.test
    if not np.array_equal(df.index[test_ids], preds["clean_index"]):
        raise RuntimeError("rebuilt split does not match test_predictions.csv; re-run the benchmark")
    print(f"Loaded benchmark outputs: {len(preds):,} test listings, threshold {threshold:,.0f} AZN, "
          f"final SVM kernel = {final_kernel}")

    # 1. Results tables, and the setup facts quoted in the report.
    headline_table(tables)
    full_results_tables(tables)
    report_facts(meta, raw, df, X, names, price, {"train": train_ids, "val": val_ids, "test": test_ids},
                 threshold)
    print(f"Wrote {PATHS['tables'].relative_to(ROOT)}/results_{{headline,task_a,task_b}}.{{csv,tex}} "
          f"and {PATHS['results'].relative_to(ROOT)}/report_facts.csv")

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
    figures += plot_svm_weights(svm.w, names)
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
