"""
splits.py — the one train / validation / test split that every stage uses
(eda, svm_tuning, benchmark, analysis).

get_splits(X, price) calls data_prep.train_val_test_split WITH a
stratification label: the tier label computed from the median of ALL prices
(data_prep.make_tier_label on the full price vector). That label is used only
to keep the premium share the same in the three splits; no model ever sees
it. The tier labels the models use are then recomputed with
data_prep.make_tier_label from the TRAINING prices only, and that training
median labels validation and test.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np

from . import data_prep


@dataclass(frozen=True)
class Splits:
    train: np.ndarray       # row positions in X / price
    val: np.ndarray
    test: np.ndarray
    threshold: float        # tier threshold = median TRAINING price
    y_train: np.ndarray     # tier labels (1 = premium) from that threshold
    y_val: np.ndarray
    y_test: np.ndarray


def get_splits(X, price) -> Splits:
    """
    Seeded, stratified 70/15/15 split of the rows of X / price.

    data_prep.train_val_test_split returns arrays, not row positions, so it
    is given a one-column matrix of row ids in place of X: the split depends
    only on the number of rows, the seed and the stratification label, so the
    ids come back in exactly the positions the rows of X would.
    """
    price = np.asarray(price, dtype=float)
    if len(X) != len(price):
        raise ValueError("X and price must have the same number of rows")
    row_id = np.arange(len(price), dtype=float)[:, None]
    with contextlib.redirect_stdout(io.StringIO()):
        strata, _ = data_prep.make_tier_label(price)            # full data: stratification only
        id_tr, p_tr, id_val, p_val, id_te, p_te, *_ = data_prep.train_val_test_split(
            row_id, price, y_tier=strata)
        train, val, test = (ids[:, 0].astype(int) for ids in (id_tr, id_val, id_te))
        if not (np.array_equal(price[train], p_tr) and np.array_equal(price[val], p_val)
                and np.array_equal(price[test], p_te)):
            raise RuntimeError("train_val_test_split returned rows out of step with the row ids")
        # The labels the models use: threshold from the training rows only.
        y_train, threshold = data_prep.make_tier_label(price[train])
        y_val, _ = data_prep.make_tier_label(price[val], threshold)
        y_test, _ = data_prep.make_tier_label(price[test], threshold)
    return Splits(train, val, test, float(threshold), y_train, y_val, y_test)
