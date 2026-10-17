"""
Every stage (eda, svm_tuning, benchmark, analysis) must use the same
train / validation / test split: the stratified one from src/splits.py.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

from src import analysis, benchmark, data_prep, eda, splits, svm_tuning

STAGES = (eda, svm_tuning, benchmark, analysis)


def _data(n=2_000, seed=0):
    rng = np.random.default_rng(seed)
    price = np.round(np.exp(rng.normal(12.3, 0.6, n)), -3)      # many ties, like listing prices
    X = rng.normal(size=(n, 4))
    return X, price


@pytest.mark.parametrize("stage", STAGES, ids=lambda m: m.__name__)
def test_every_stage_splits_with_get_splits(stage):
    # Same function object, and no stage calls the raw splitter itself, so
    # all four get identical splits from identical (X, price).
    assert stage.get_splits is splits.get_splits
    assert "train_val_test_split(" not in inspect.getsource(stage)


def test_identical_splits_on_repeated_calls():
    X, price = _data()
    a, b = splits.get_splits(X, price), splits.get_splits(X.copy(), price.copy())
    for field in ("train", "val", "test", "y_train", "y_val", "y_test"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert a.threshold == b.threshold


def test_split_partitions_rows_70_15_15():
    X, price = _data()
    s = splits.get_splits(X, price)
    ids = np.concatenate([s.train, s.val, s.test])
    assert len(ids) == len(price) and len(np.unique(ids)) == len(price)
    assert len(s.test) == pytest.approx(0.15 * len(price), abs=1)
    assert len(s.val) == pytest.approx(0.15 * len(price), abs=2)


def test_stratified_on_full_data_tier_labels_from_training_median():
    X, price = _data()
    s = splits.get_splits(X, price)
    # Stratification label: the full-data tier; its share is the same in every split.
    strata, _ = data_prep.make_tier_label(price)
    shares = [strata[ids].mean() for ids in (s.train, s.val, s.test)]
    assert max(shares) - min(shares) < 0.01
    # The labels the models use come from the training median only.
    assert s.threshold == np.median(price[s.train])
    np.testing.assert_array_equal(s.y_train, (price[s.train] >= s.threshold).astype(int))
    np.testing.assert_array_equal(s.y_val, (price[s.val] >= s.threshold).astype(int))
    np.testing.assert_array_equal(s.y_test, (price[s.test] >= s.threshold).astype(int))


@pytest.mark.skipif(not Path(data_prep.DATA_PATH).exists(), reason="raw CSV not present")
def test_real_data_split_sizes():
    df = data_prep.clean(data_prep.load_raw(data_prep.DATA_PATH))
    X, price, _ = data_prep.make_features(df)
    s = splits.get_splits(X, price)
    assert len(s.train) + len(s.val) + len(s.test) == len(price)
    assert s.threshold == np.median(price[s.train])
