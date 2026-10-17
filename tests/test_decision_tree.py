"""
test_decision_tree.py - Sanity check to compare custom DecisionTree with scikit-learn.
"""

import sys
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Add the root directory to the path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.decision_tree import DecisionTree


def test_classification_match():
    """Sanity check for the Classification task."""
    # A very small, fully-determined toy dataset
    X = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [8.0, 8.0],
        [1.0, 0.6],
        [9.0, 11.0]
    ])
    y = np.array([0, 0, 1, 1, 0, 1])

    # Our custom decision tree
    custom_tree = DecisionTree(task="classification", criterion="gini", max_depth=2, random_state=42)
    custom_tree.fit(X, y)
    custom_preds = custom_tree.predict(X)

    # Scikit-learn decision tree
    sklearn_tree = DecisionTreeClassifier(criterion="gini", max_depth=2, random_state=42)
    sklearn_tree.fit(X, y)
    sklearn_preds = sklearn_tree.predict(X)

    # Check if the predictions match exactly
    np.testing.assert_array_equal(custom_preds, sklearn_preds)
    print("✅ Classification test passed: Custom decision tree matches scikit-learn!")


def test_regression_match():
    """Sanity check for the Regression task."""
    # Simple toy dataset for regression
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([1.2, 1.9, 3.1, 3.9, 5.2, 6.1])

    # Our custom regression tree
    custom_tree = DecisionTree(task="regression", criterion="mse", max_depth=2, random_state=42)
    custom_tree.fit(X, y)
    custom_preds = custom_tree.predict(X)

    # Scikit-learn regression tree (MSE is called "squared_error" in sklearn)
    sklearn_tree = DecisionTreeRegressor(criterion="squared_error", max_depth=2, random_state=42)
    sklearn_tree.fit(X, y)
    sklearn_preds = sklearn_tree.predict(X)

    # Check if the predictions match (using almost_equal for floats)
    np.testing.assert_almost_equal(custom_preds, sklearn_preds, decimal=5)
    print("✅ Regression test passed: Custom decision tree matches scikit-learn!")


if __name__ == "__main__":
    print("Running tests...\n" + "-"*40)
    test_classification_match()
    test_regression_match()
    print("-"*40 + "\n🎉 All sanity tests passed successfully!")