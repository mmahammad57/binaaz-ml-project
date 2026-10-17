"""
tree_experiment.py - P4a: Hyperparameter sweeps and validation curves for Decision Tree.
Replaces synthetic skeleton with real bina.az data logic.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.decision_tree import DecisionTree
from src.splits import get_splits
from src.evaluate import rmse, precision_recall_f1
import src.data_prep as data_prep

def run(quick: bool = False):
    """
    Run hyperparameter sweeps for Task A and Task B on a stratified subsample.
    Mirrors svm_tuning.py structure.
    """
    print(f"--- Running Decision Tree Hyperparameter Sweep (quick={quick}) ---")
    
    # Setup output directories based on mode
    results_dir = os.path.join("results", "quick" if quick else "")
    figures_dir = os.path.join("report", "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Load data
    splits = get_splits()
    X_train_full = splits["X_train"]
    y_train_full = splits["y_train"] # price
    X_val = splits["X_val"]
    y_val_price = splits["y_val"]
    
    # 2. Make Task B labels (tier) using ONLY training median
    train_median = np.median(y_train_full)
    y_train_tier_full = (y_train_full > train_median).astype(int)
    y_val_tier = (y_val_price > train_median).astype(int)
    
    # Task A targets (log price)
    y_train_log_full = np.log(y_train_full)
    y_val_log = np.log(y_val_price)
    
    # 3. Subsample training data (stratified by tier)
    n_samples = 5000 if quick else 16000
    np.random.seed(data_prep.SEED)
    
    idx_0 = np.where(y_train_tier_full == 0)[0]
    idx_1 = np.where(y_train_tier_full == 1)[0]
    
    prop_1 = np.mean(y_train_tier_full)
    n_1 = int(n_samples * prop_1)
    n_0 = n_samples - n_1
    
    n_0 = min(n_0, len(idx_0))
    n_1 = min(n_1, len(idx_1))
    
    chosen_idx_0 = np.random.choice(idx_0, size=n_0, replace=False)
    chosen_idx_1 = np.random.choice(idx_1, size=n_1, replace=False)
    chosen_idx = np.concatenate([chosen_idx_0, chosen_idx_1])
    np.random.shuffle(chosen_idx)
    
    X_train = X_train_full.iloc[chosen_idx].values if hasattr(X_train_full, 'iloc') else X_train_full[chosen_idx]
    y_train_tier = y_train_tier_full[chosen_idx]
    y_train_log = y_train_log_full.iloc[chosen_idx].values if hasattr(y_train_log_full, 'iloc') else y_train_log_full[chosen_idx]
    
    X_val = X_val.values if hasattr(X_val, 'values') else X_val
    y_val_tier = y_val_tier.values if hasattr(y_val_tier, 'values') else y_val_tier
    y_val_log = y_val_log.values if hasattr(y_val_log, 'values') else y_val_log

if __name__ == "__main__":
    import sys
    quick_mode = "--quick" in sys.argv
    run(quick=quick_mode)