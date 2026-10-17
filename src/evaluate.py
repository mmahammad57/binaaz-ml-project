"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH where reasonable).

Implement the metrics yourself in NumPy. scikit-learn may be used ONLY to
sanity-check your numbers or for the baseline models you compare against.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _paired_1d(y_true, y_other):
    """Return both inputs as equal-length, nonempty 1-D float arrays."""
    y_true = np.asarray(y_true, dtype=float)
    y_other = np.asarray(y_other, dtype=float)
    if y_true.ndim != 1 or y_other.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    if len(y_true) == 0 or len(y_true) != len(y_other):
        raise ValueError("inputs must be nonempty and of equal length")
    return y_true, y_other


def _binary_labels(y_true, y_pred):
    """Return both label arrays as booleans (True = premium = 1)."""
    y_true, y_pred = _paired_1d(y_true, y_pred)
    if not (np.isin(y_true, (0, 1)).all() and np.isin(y_pred, (0, 1)).all()):
        raise ValueError("labels must be 0 (standard) or 1 (premium)")
    return y_true == 1, y_pred == 1


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _paired_1d(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _paired_1d(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = _paired_1d(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    # Constant y_true makes R^2 undefined; follow sklearn: 1.0 if the
    # predictions are perfect, otherwise 0.0.
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]]."""
    t, p = _binary_labels(y_true, y_pred)
    return np.array([
        [np.sum(~t & ~p), np.sum(~t & p)],
        [np.sum(t & ~p), np.sum(t & p)],
    ])


def precision_recall_f1(y_true, y_pred):
    """
    Return (precision, recall, f1) for the positive (premium) class.
    Any ratio whose denominator is zero is reported as 0.0.
    """
    (_, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def roc_auc(y_true, scores) -> float:
    """
    AUC from decision scores (e.g. SVM margins or tree probabilities).
    Rank-based (Mann-Whitney) computation is fine.
    """
    y_true, scores = _paired_1d(y_true, scores)
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("labels must be 0 (standard) or 1 (premium)")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        # Same behaviour as sklearn: warn and return NaN, so a one-class
        # segment in error analysis does not abort the whole loop.
        warnings.warn("Only one class is present in y_true; ROC AUC is undefined.",
                      RuntimeWarning, stacklevel=2)
        return float("nan")

    # 1-based ranks of the scores, with tied scores sharing the average of
    # the ranks they occupy. A tie group with c members ending at sorted
    # position e covers ranks e-c+1 .. e, whose average is e - (c-1)/2.
    _, group, counts = np.unique(scores, return_inverse=True, return_counts=True)
    group_end = np.cumsum(counts)
    ranks = (group_end - (counts - 1) / 2.0)[group]

    # Mann-Whitney U: how many (positive, negative) pairs the positive wins,
    # counting ties as half. AUC is U divided by the number of pairs.
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _ranked_counts(y_true, scores):
    """
    Cumulative true/false positives when the decision threshold is lowered
    through each distinct score, highest first. Returns (tps, fps, thresholds).
    """
    y_true, scores = _paired_1d(y_true, scores)
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("labels must be 0 (standard) or 1 (premium)")
    order = np.argsort(-scores, kind="mergesort")
    scores, y_true = scores[order], y_true[order]
    # Last position of each run of tied scores: a threshold can't split ties.
    last = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tps = np.cumsum(y_true)[last]
    fps = (last + 1) - tps
    return tps, fps, scores[last]


def roc_curve(y_true, scores):
    """(fpr, tpr, thresholds), starting at (0, 0); thresholds descend."""
    tps, fps, thresholds = _ranked_counts(y_true, scores)
    if tps[-1] == 0 or fps[-1] == 0:
        raise ValueError("ROC curve needs both classes in y_true")
    return np.r_[0.0, fps / fps[-1]], np.r_[0.0, tps / tps[-1]], np.r_[np.inf, thresholds]


def precision_recall_curve(y_true, scores):
    """(precision, recall, thresholds) for the premium class; thresholds descend."""
    tps, fps, thresholds = _ranked_counts(y_true, scores)
    if tps[-1] == 0:
        raise ValueError("precision-recall curve needs at least one positive")
    return tps / (tps + fps), tps / tps[-1], thresholds


# --- Tree diagnostics -------------------------------------------------------
def tree_feature_importances(tree, n_features: int) -> np.ndarray:
    """
    Feature importances of a fitted DecisionTree, as an array of length
    n_features that sums to 1 (all zeros if the tree is a single leaf).
    """
    root = tree.root
    if root is None:
        raise ValueError("fit the tree before computing feature importances")
    if getattr(root, "n_samples", None) is None or getattr(root, "impurity", None) is None:
        raise AttributeError(
            "tree nodes must record n_samples and impurity "
            "(the DecisionTree update on the decision-tree branch adds them)"
        )

    # Mean decrease in impurity (MDI), the definition sklearn uses for
    # feature_importances_. Each split removes
    #     n_node * impurity_node - (n_left * impurity_left + n_right * impurity_right)
    # of sample-weighted impurity; that amount is credited to the feature the
    # node splits on. sklearn also divides by the root's sample count, but that
    # cancels out in the final normalisation, so it is skipped here.
    importances = np.zeros(n_features)
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf():
            continue
        left, right = node.left, node.right
        importances[node.feature] += (
            node.n_samples * node.impurity
            - (left.n_samples * left.impurity + right.n_samples * right.impurity)
        )
        stack.extend((left, right))

    total = importances.sum()
    return importances / total if total > 0 else importances


# --- Comparison helper ------------------------------------------------------
def _format_value(value, decimals, pm=" ± "):
    """Format a float, or a (mean, std) pair as 'mean ± std'."""
    if value is None:
        return "n/a"
    if isinstance(value, (tuple, list)):
        mean, std = value
        return f"{mean:.{decimals}f}{pm}{std:.{decimals}f}"
    return f"{value:.{decimals}f}"


def _metric_names(name_to_metrics):
    """Union of metric names across models, in first-seen order."""
    names = {}
    for metrics in name_to_metrics.values():
        names.update(dict.fromkeys(metrics))
    return list(names)


def results_table(name_to_metrics: dict, decimals: int = 4) -> pd.DataFrame:
    """
    Return the comparison table as a DataFrame of formatted strings:
    one row per model (index "model"), one column per metric.
    A metric that a model does not report is shown as "n/a".
    """
    columns = _metric_names(name_to_metrics)
    rows = {
        model: [_format_value(metrics.get(col), decimals) for col in columns]
        for model, metrics in name_to_metrics.items()
    }
    table = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    table.index.name = "model"
    return table


def compare(name_to_metrics: dict, decimals: int = 4) -> None:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    Keep the SAME preprocessing/split for a fair comparison.
    """
    table = results_table(name_to_metrics, decimals)
    header = [table.index.name, *table.columns]
    body = [[model, *row] for model, row in zip(table.index, table.values.tolist())]
    widths = [max(len(str(r[i])) for r in [header, *body]) for i in range(len(header))]

    def line(cells):
        # Model names left-aligned, metric values right-aligned.
        return "  ".join(
            str(c).ljust(w) if i == 0 else str(c).rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths))
        )

    print(line(header))
    print("-" * len(line(header)))
    for row in body:
        print(line(row))


def _latex_escape(text) -> str:
    """Escape LaTeX special characters in a model or metric name."""
    special = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(special.get(ch, ch) for ch in str(text))


def save_table(name_to_metrics: dict, csv_path, tex_path, decimals: int = 4) -> None:
    """
    Write the comparison table to CSV (UTF-8) and to a booktabs LaTeX
    tabular (needs \\usepackage{booktabs} in the report preamble).
    """
    csv_path, tex_path = Path(csv_path), Path(tex_path)
    for path in (csv_path, tex_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    results_table(name_to_metrics, decimals).to_csv(csv_path, encoding="utf-8")

    # Built by hand rather than with DataFrame.to_latex, which needs jinja2
    # (not in requirements.txt). "±" becomes $\pm$ so pdflatex can typeset it.
    columns = _metric_names(name_to_metrics)
    lines = [
        r"\begin{tabular}{l" + "r" * len(columns) + "}",
        r"\toprule",
        " & ".join(["Model", *map(_latex_escape, columns)]) + r" \\",
        r"\midrule",
    ]
    for model, metrics in name_to_metrics.items():
        cells = [_format_value(metrics.get(col), decimals, pm=r" $\pm$ ") for col in columns]
        lines.append(" & ".join([_latex_escape(model), *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
