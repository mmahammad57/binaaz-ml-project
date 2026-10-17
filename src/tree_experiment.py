"""
tree_experiment.py — P4a: Hyperparameter sweeps and validation curves for Decision Tree.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.decision_tree import DecisionTree

# Directory to save generated figures
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_dummy_data():
    """Generate synthetic classification data for testing until real data is integrated."""
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)

def plot_validation_curve(X_train, y_train, X_val, y_val):
    """Plot validation curve for max_depth (Overfitting / Bias-Variance analysis)."""
    depths = range(1, 16)
    train_scores = []
    val_scores = []

    print("Sweeping max_depth...")
    for depth in depths:
        tree = DecisionTree(task="classification", criterion="gini", max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        
        train_preds = tree.predict(X_train)
        val_preds = tree.predict(X_val)
        
        train_scores.append(accuracy_score(y_train, train_preds))
        val_scores.append(accuracy_score(y_val, val_preds))
        print(f"Depth: {depth:2d} | Train Acc: {train_scores[-1]:.3f} | Val Acc: {val_scores[-1]:.3f}")

    # Plotting the curve
    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_scores, label="Train Accuracy", marker='o')
    plt.plot(depths, val_scores, label="Validation Accuracy", marker='s')
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Validation Curve (Overfitting Analysis)")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(FIGURES_DIR, "tree_validation_curve.png")
    plt.savefig(plot_path)
    print(f"✅ Figure saved to: {plot_path}")
    plt.close()