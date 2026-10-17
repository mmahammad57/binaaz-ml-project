"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree. Stores split information or leaf values."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node: left child
        self.right = None        # Node: right child
        self.value = None        # leaf prediction (class proba array or mean float)
        
        # Added for P6 feature importance requirements
        self.n_samples = None    # int: number of samples reaching this node
        self.impurity = None     # float: impurity of this node

    def is_leaf(self) -> bool:
        """Returns True if the node is a leaf (has no children)."""
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None

        self.classes_ = None

    # --- impurity -----------------------------------------------------------
    def _impurity(self, y) -> float:
        """Gini / entropy (classification) or MSE/variance (regression)."""
        if len(y) == 0:
            return 0.0

        if self.task == "regression":
            # Variance for MSE criterion
            return np.var(y)
        else:
            # Calculate class probabilities
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            
            if self.criterion == "gini":
                # Gini impurity: 1 - sum(p_i^2)
                return 1.0 - np.sum(probs ** 2)
            elif self.criterion == "entropy":
                # Entropy: -sum(p_i * log2(p_i))
                return -np.sum(probs * np.log2(probs))
            else:
                raise ValueError(f"Invalid criterion for classification: {self.criterion}")

    def _best_split(self, X, y):
        """
        Search features & candidate thresholds; return the split that
        maximizes impurity decrease (feature, threshold, decrease), or
        None if no valid split improves things.
        """
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)
        
        best_feature = None
        best_threshold = None
        max_decrease = -1.0

        # Iterate over all features
        for feature_idx in range(n_features):
            column_values = X[:, feature_idx]
            # Test all unique values in the feature as potential thresholds
            thresholds = np.unique(column_values)

            for threshold in thresholds:
                # Split the data
                left_mask = column_values <= threshold
                right_mask = ~left_mask
                
                y_left = y[left_mask]
                y_right = y[right_mask]

                # Check min_samples_leaf constraint
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                # Calculate impurity of children
                n_left = len(y_left)
                n_right = len(y_right)
                
                left_impurity = self._impurity(y_left)
                right_impurity = self._impurity(y_right)
                
                # Weighted average of child impurities
                child_impurity = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity
                
                # Information Gain (or Variance Reduction)
                impurity_decrease = parent_impurity - child_impurity

                if impurity_decrease > max_decrease:
                    max_decrease = impurity_decrease
                    best_feature = feature_idx
                    best_threshold = threshold

        # Only return the split if it meets the minimum impurity decrease threshold
        if max_decrease >= self.min_impurity_decrease and best_feature is not None:
            return best_feature, best_threshold, max_decrease
            
        return None

    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        node = Node()
        n_samples = len(y)
        
        # Store node statistics for P6 feature importance
        node.n_samples = n_samples
        node.impurity = self._impurity(y)
        
        # Determine if node is pure (all samples have same target value)
        is_pure = len(np.unique(y)) == 1
        
        # Check stopping criteria
        if (is_pure or 
            n_samples < self.min_samples_split or 
            (self.max_depth is not None and depth >= self.max_depth)):
            node.value = self._calculate_leaf_value(y)
            return node

        # Find the best split
        split = self._best_split(X, y)
        
        # If no valid split is found (e.g., due to min_samples_leaf), make it a leaf
        if split is None:
            node.value = self._calculate_leaf_value(y)
            return node

        best_feature, best_threshold, _ = split

        # Perform the actual split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        node.feature = best_feature
        node.threshold = best_threshold
        
        # Recursively grow left and right branches
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[right_mask], y[right_mask], depth + 1)

        return node
    
    def _calculate_leaf_value(self, y):
        """Helper to compute the prediction value for a leaf node."""
        if self.task == "regression":
            # Leaf value is the mean of the target values
            return np.mean(y)
        else:
            # Leaf value is an array of class probabilities
            counts = np.zeros(len(self.classes_))
            for i, c in enumerate(self.classes_):
                counts[i] = np.sum(y == c)
            return counts / len(y)

    def fit(self, X, y):
        # Fix random seed for reproducibility
        np.random.seed(self.random_state)
        
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if self.task == "classification":
            # Store the unique classes found in the training data
            self.classes_ = np.unique(y)

        # Start recursive tree growth from the root
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        """Traverse the tree to find the leaf prediction for a single sample."""
        if node.is_leaf():
            return node.value

        # Traverse left if feature value <= threshold, otherwise right
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
        
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = [self._predict_one(x, self.root) for x in X]

        if self.task == "regression":
            return np.array(predictions)
        else:
            # For classification, return the class with the highest probability
            probabilities = np.array(predictions)
            best_class_indices = np.argmax(probabilities, axis=1)
            return self.classes_[best_class_indices]

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")
            
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root) for x in X])