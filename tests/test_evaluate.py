"""
test_evaluate.py — check the from-scratch metrics against sklearn.metrics.

Run from the repo root:
    python -m pytest tests/test_evaluate.py -v
"""

import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.exceptions import UndefinedMetricWarning

from src import evaluate as ev
from src.data_prep import SEED


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def labels(rng):
    """Random 0/1 truth and predictions that agree about 70% of the time."""
    y_true = rng.integers(0, 2, 1000)
    flip = rng.random(1000) < 0.3
    return y_true, np.where(flip, 1 - y_true, y_true)


# --- Regression -------------------------------------------------------------
def test_regression_metrics_match_sklearn(rng):
    y_true = rng.normal(200_000, 50_000, 500)
    y_pred = y_true + rng.normal(0, 20_000, 500)

    assert ev.rmse(y_true, y_pred) == pytest.approx(skm.root_mean_squared_error(y_true, y_pred))
    assert ev.mae(y_true, y_pred) == pytest.approx(skm.mean_absolute_error(y_true, y_pred))
    assert ev.r2(y_true, y_pred) == pytest.approx(skm.r2_score(y_true, y_pred))


def test_r2_negative_for_worse_than_mean(rng):
    y_true = rng.normal(size=200)
    y_pred = rng.normal(size=200) * 5

    assert ev.r2(y_true, y_pred) < 0
    assert ev.r2(y_true, y_pred) == pytest.approx(skm.r2_score(y_true, y_pred))


@pytest.mark.parametrize("y_pred", [[3.0, 3.0, 3.0], [3.0, 2.0, 4.0]])
def test_r2_constant_target_matches_sklearn(y_pred):
    y_true = [3.0, 3.0, 3.0]
    assert ev.r2(y_true, y_pred) == skm.r2_score(y_true, y_pred)


# --- Classification ---------------------------------------------------------
def test_confusion_matrix_matches_sklearn(labels):
    y_true, y_pred = labels
    np.testing.assert_array_equal(
        ev.confusion_matrix(y_true, y_pred),
        skm.confusion_matrix(y_true, y_pred, labels=[0, 1]),
    )


def test_precision_recall_f1_match_sklearn(labels):
    y_true, y_pred = labels
    expected = (
        skm.precision_score(y_true, y_pred, zero_division=0),
        skm.recall_score(y_true, y_pred, zero_division=0),
        skm.f1_score(y_true, y_pred, zero_division=0),
    )
    assert ev.precision_recall_f1(y_true, y_pred) == pytest.approx(expected)


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        ([1, 1, 1, 1], [1, 0, 1, 1]),  # all-premium truth
        ([0, 0, 0, 0], [0, 1, 0, 0]),  # all-standard truth: recall is 0/0
        ([0, 1, 0, 1], [0, 0, 0, 0]),  # no premium predicted: precision is 0/0
        ([0, 0, 0, 0], [0, 0, 0, 0]),  # everything is 0/0
    ],
)
def test_single_class_and_zero_division_match_sklearn(y_true, y_pred):
    np.testing.assert_array_equal(
        ev.confusion_matrix(y_true, y_pred),
        skm.confusion_matrix(y_true, y_pred, labels=[0, 1]),
    )
    expected = (
        skm.precision_score(y_true, y_pred, zero_division=0),
        skm.recall_score(y_true, y_pred, zero_division=0),
        skm.f1_score(y_true, y_pred, zero_division=0),
    )
    assert ev.precision_recall_f1(y_true, y_pred) == pytest.approx(expected)


def test_classification_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        ev.confusion_matrix([0, 1, 2], [0, 1, 1])


# --- ROC AUC ----------------------------------------------------------------
def test_roc_auc_matches_sklearn_continuous_scores(rng):
    y_true = rng.integers(0, 2, 1000)
    scores = y_true + rng.normal(0, 1.0, 1000)
    assert ev.roc_auc(y_true, scores) == pytest.approx(skm.roc_auc_score(y_true, scores))


def test_roc_auc_matches_sklearn_with_tied_scores(rng):
    # Only 5 distinct score values over 1000 rows, so nearly every score is tied.
    y_true = rng.integers(0, 2, 1000)
    scores = (y_true + rng.integers(0, 4, 1000)).astype(float)
    assert len(np.unique(scores)) == 5
    assert ev.roc_auc(y_true, scores) == pytest.approx(skm.roc_auc_score(y_true, scores))


def test_roc_auc_known_values():
    y_true = [0, 0, 1, 1]
    assert ev.roc_auc(y_true, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert ev.roc_auc(y_true, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert ev.roc_auc(y_true, [0.5, 0.5, 0.5, 0.5]) == 0.5


@pytest.mark.parametrize("y_true", [[1, 1, 1, 1], [0, 0, 0, 0]])
def test_roc_auc_single_class_returns_nan_like_sklearn(y_true):
    # sklearn >= 1.6 warns and returns NaN instead of raising.
    scores = [0.1, 0.4, 0.35, 0.8]
    with pytest.warns(RuntimeWarning, match="one class"):
        assert np.isnan(ev.roc_auc(y_true, scores))
    with pytest.warns(UndefinedMetricWarning):
        assert np.isnan(skm.roc_auc_score(y_true, scores))


# --- Comparison table -------------------------------------------------------
RESULTS = {
    "Our tree": {"F1": 0.81234, "AUC": (0.9, 0.01234)},
    "sklearn_tree": {"F1": 0.8, "AUC": (0.91, 0.02)},
    "Our SVM": {"F1": 0.75},
}


def test_results_table_formats_values():
    table = ev.results_table(RESULTS, decimals=3)

    assert list(table.columns) == ["F1", "AUC"]
    assert list(table.index) == ["Our tree", "sklearn_tree", "Our SVM"]
    assert table.loc["Our tree", "F1"] == "0.812"
    assert table.loc["Our tree", "AUC"] == "0.900 ± 0.012"
    assert table.loc["Our SVM", "AUC"] == "n/a"


def test_compare_prints_aligned_table(capsys):
    ev.compare(RESULTS)
    lines = capsys.readouterr().out.splitlines()

    assert len(lines) == 2 + len(RESULTS)
    assert len({len(line) for line in lines}) == 1
    assert "0.9000 ± 0.0123" in lines[2]


def test_save_table_writes_csv_and_booktabs(tmp_path):
    csv_path, tex_path = tmp_path / "out" / "results.csv", tmp_path / "out" / "results.tex"
    ev.save_table(RESULTS, csv_path, tex_path)

    csv_text = csv_path.read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == "model,F1,AUC"
    assert "0.9000 ± 0.0123" in csv_text

    tex = tex_path.read_text(encoding="utf-8")
    assert tex.startswith(r"\begin{tabular}{lrr}")
    for rule in (r"\toprule", r"\midrule", r"\bottomrule"):
        assert rule in tex
    assert r"0.9000 $\pm$ 0.0123" in tex
    assert r"sklearn\_tree" in tex
    assert "±" not in tex
