"""
test_evaluate.py — check the from-scratch metrics against sklearn.metrics.

Run from the repo root:
    python -m pytest tests/test_evaluate.py -v
"""

import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src import evaluate as ev
from src.data_prep import SEED
from src.decision_tree import DecisionTree


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


@pytest.mark.parametrize("tied", [False, True])
def test_roc_curve_matches_sklearn(rng, tied):
    y = rng.integers(0, 2, 500)
    scores = y + rng.normal(0, 1.0, 500)
    if tied:
        scores = np.round(scores)
    fpr, tpr, thr = ev.roc_curve(y, scores)
    sk_fpr, sk_tpr, sk_thr = skm.roc_curve(y, scores, drop_intermediate=False)
    np.testing.assert_allclose(fpr, sk_fpr)
    np.testing.assert_allclose(tpr, sk_tpr)
    np.testing.assert_array_equal(thr[1:], sk_thr[1:])


@pytest.mark.parametrize("tied", [False, True])
def test_precision_recall_curve_matches_sklearn(rng, tied):
    y = rng.integers(0, 2, 500)
    scores = y + rng.normal(0, 1.0, 500)
    if tied:
        scores = np.round(scores)
    precision, recall, thr = ev.precision_recall_curve(y, scores)
    sk_p, sk_r, sk_thr = skm.precision_recall_curve(y, scores)
    # sklearn lists thresholds ascending, stops once recall reaches 1 and
    # appends a (precision=1, recall=0) end point; compare on its thresholds.
    ours = dict(zip(thr.tolist(), zip(precision, recall)))
    for t, p, r in zip(sk_thr, sk_p[:-1], sk_r[:-1]):
        assert ours[t] == pytest.approx((p, r))


# --- Tree feature importances -----------------------------------------------
# Small hand-fixed datasets on which our tree and sklearn's grow identical
# trees, with every split winning by a clear margin (no ties for either
# implementation to break differently). Importance lands on two features,
# and one feature gets exactly zero.
X_CLF = np.array([[3, 9, 6], [1, 4, 4], [10, 6, 2], [11, 12, 12], [12, 5, 3], [8, 8, 7],
                  [6, 7, 10], [7, 2, 8], [4, 10, 5], [5, 3, 11], [9, 11, 1], [2, 1, 9]], float)
Y_CLF = np.array([1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0])
X_REG = np.array([[3, 4, 2], [1, 10, 5], [8, 5, 4], [7, 6, 3], [10, 7, 8],
                  [6, 9, 7], [4, 8, 1], [5, 2, 6], [9, 3, 10], [2, 1, 9]], float)
Y_REG = np.array([3.6, 8.1, 14.2, 15.0, 15.1, 17.6, 7.4, 0.8, 11.5, 0.0])

MATCHED_TREES = [
    pytest.param(DecisionTree("classification", "gini", max_depth=3),
                 DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=SEED),
                 X_CLF, Y_CLF, id="gini"),
    pytest.param(DecisionTree("classification", "entropy", max_depth=3),
                 DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=SEED),
                 X_CLF, Y_CLF, id="entropy"),
    pytest.param(DecisionTree("regression", "mse", max_depth=2),
                 DecisionTreeRegressor(criterion="squared_error", max_depth=2, random_state=SEED),
                 X_REG, Y_REG, id="mse"),
]


def _preorder(node):
    """(split feature, n_samples) per node, depth-first, left before right; -2 marks a leaf."""
    if node.is_leaf():
        return [(-2, node.n_samples)]
    return [(node.feature, node.n_samples)] + _preorder(node.left) + _preorder(node.right)


@pytest.mark.parametrize("ours, theirs, X, y", MATCHED_TREES)
def test_tree_feature_importances_match_sklearn(ours, theirs, X, y):
    ours.fit(X, y)
    theirs.fit(X, y)

    # Precondition: identical splits. sklearn numbers its nodes in the same
    # depth-first order, with -2 as the leaf feature, so the lists line up.
    sk = theirs.tree_
    assert _preorder(ours.root) == list(zip(sk.feature.tolist(), sk.n_node_samples.tolist()))

    importances = ev.tree_feature_importances(ours, X.shape[1])
    np.testing.assert_allclose(importances, theirs.feature_importances_, rtol=0, atol=1e-9)
    assert (importances > 0).sum() == 2


@pytest.mark.parametrize("task, criterion", [("classification", "gini"),
                                             ("classification", "entropy"),
                                             ("regression", "mse")])
def test_tree_feature_importances_are_a_distribution(rng, task, criterion):
    X = rng.normal(size=(300, 6))
    signal = X[:, 0] + 0.5 * X[:, 1] * X[:, 2]
    y = (signal > 0).astype(int) if task == "classification" else signal
    tree = DecisionTree(task, criterion, max_depth=5).fit(X, y)

    importances = ev.tree_feature_importances(tree, X.shape[1])
    assert importances.shape == (6,)
    assert np.all(importances >= 0)
    assert importances.sum() == pytest.approx(1.0)


def test_tree_feature_importances_single_leaf_is_all_zero():
    tree = DecisionTree("classification", "gini").fit(X_CLF, np.ones(len(X_CLF), int))
    assert tree.root.is_leaf()
    np.testing.assert_array_equal(ev.tree_feature_importances(tree, 3), np.zeros(3))


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
