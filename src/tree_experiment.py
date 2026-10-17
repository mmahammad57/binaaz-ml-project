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