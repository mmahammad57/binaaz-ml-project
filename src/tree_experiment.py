"""
tree_experiment.py - P4a: Hyperparameter sweeps and validation curves for Decision Tree.
Replaces synthetic skeleton with real bina.az data logic.
"""

import os
import time
import json
import numpy as np
import pandas as pd

from src.decision_tree import DecisionTree
from src.splits import get_splits
from src.evaluate import rmse, precision_recall_f1
import src.data_prep as data_prep

# Figure conventions shared with svm_tuning.py and analysis.py: palette slots
# in a fixed order, recessive grid, text in ink colours, drawn at the printed
# size with explicit 8.5-9 pt fonts (no global matplotlib settings).
INK, MUTED, GRID = "#1f1f1f", "#5f5f5f", "#e3e3e3"
SLOTS = ("#2a78d6", "#eb6834")
FONT, SMALL = 9, 8.5


def _figure(size, nrows=1, ncols=1):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(nrows, ncols)).ravel()
    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=SMALL)
    return fig, axes


def _label(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=FONT, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=FONT, color=INK)
    ax.set_ylabel(ylabel, fontsize=FONT, color=INK)
    ax.legend(frameon=False, fontsize=SMALL, labelcolor=INK)


def _save(fig, figures_dir, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figures_dir, f"{stem}.{ext}"), dpi=300, facecolor="white")


def plot_depth_curves(depths, task_a_curves, task_b_curves, figures_dir):
    """Train (dashed) and validation (solid) score against max_depth, min_samples_leaf = 1."""
    fig, (ax1, ax2) = _figure((7.0, 2.6), ncols=2)
    ax1.plot(depths, task_a_curves["train"], color=SLOTS[0], linewidth=1.5, linestyle="--", label="Train")
    ax1.plot(depths, task_a_curves["val"], color=SLOTS[0], linewidth=1.5, marker="o", markersize=4,
             label="Validation")
    _label(ax1, "Task A: regression tree", "max_depth", "RMSE (log price)")
    for crit, color, marker in (("gini", SLOTS[0], "o"), ("entropy", SLOTS[1], "s")):
        ax2.plot(depths, task_b_curves[crit]["train"], color=color, linewidth=1.5, linestyle="--",
                 label=f"Train, {crit}")
        ax2.plot(depths, task_b_curves[crit]["val"], color=color, linewidth=1.5, marker=marker, markersize=4,
                 label=f"Validation, {crit}")
    _label(ax2, "Task B: classification tree", "max_depth", "F1 (premium)")
    for ax in (ax1, ax2):
        ax.set_xticks(depths)
    _save(fig, figures_dir, "tree_depth_curves")


def plot_learning_curve(curve, configs, figures_dir):
    """
    Train (dashed) and validation (solid) score of the chosen configs against
    the number of training rows; one panel per task, one report column wide.
    """
    fig, axes = _figure((3.5, 4.4), nrows=2)
    panels = (("A_Regression", "Task A", "RMSE (log price)", configs["A"]),
              ("B_Classification", "Task B", "F1 (premium)", configs["B"]))
    for ax, (task, name, ylabel, cfg) in zip(axes, panels):
        rows = curve[curve["Task"] == task]
        ax.plot(rows["Train_Size"], rows["Train_Metric"], color=SLOTS[0], linewidth=1.5, linestyle="--",
                label="Train")
        ax.plot(rows["Train_Size"], rows["Val_Metric"], color=SLOTS[0], linewidth=1.5, marker="o",
                markersize=4, label="Validation")
        ax.set_xscale("log", base=2)
        ax.set_xticks(rows["Train_Size"], [f"{k / 1000:g}k" for k in rows["Train_Size"]])
        ax.minorticks_off()
        _label(ax, f"{name}: {cfg}", "Training rows", ylabel)
    _save(fig, figures_dir, "tree_learning_curve")


TOLERANCE = 0.002   # validation-score tolerance of the selection rule


def _select(rows, higher_is_better, simplicity):
    """
    Selection rule for one sweep step: find the best validation score among
    `rows`, keep every row within TOLERANCE of it, and return the simplest of
    those, i.e. the one with the smallest simplicity(row) (depth step: the
    max_depth; leaf step: minus min_samples_leaf). Rows equally simple are
    ordered by validation score.
    """
    scores = [r["Val_Metric"] for r in rows]
    best = max(scores) if higher_is_better else min(scores)
    close = [r for r in rows if abs(r["Val_Metric"] - best) <= TOLERANCE + 1e-12]
    sign = -1 if higher_is_better else 1
    return min(close, key=lambda r: (simplicity(r), sign * r["Val_Metric"]))


def run(quick: bool = False):
    """
    Run hyperparameter sweeps for Task A and Task B on a stratified subsample.
    Mirrors svm_tuning.py structure.
    """
    print(f"--- Running Decision Tree Hyperparameter Sweep (quick={quick}) ---")
    
    # Output folders as in svm_tuning.py: the config JSONs go to results/
    # (results/quick/ in quick mode), the sweep table to <results>/tree_experiment/,
    # and the figures to report/figures/ in full mode only (quick mode keeps them
    # next to the sweep table, so it never overwrites a report figure).
    results_dir = os.path.join("results", "quick") if quick else "results"
    table_dir = os.path.join(results_dir, "tree_experiment")
    figures_dir = table_dir if quick else os.path.join("report", "figures")
    for folder in (results_dir, table_dir, figures_dir):
        os.makedirs(folder, exist_ok=True)
    
    # 1. Load data
    df = data_prep.clean(data_prep.load_raw(data_prep.DATA_PATH))
    X, price, names = data_prep.make_features(df)
    splits = get_splits(X, price)
    X_train_full = X[splits.train]
    y_train_full = price[splits.train] # price
    X_val = X[splits.val]
    y_val_price = price[splits.val]
    
    # 2. Make Task B labels (tier) using ONLY training median
    train_median = splits.threshold
    y_train_tier_full = splits.y_train
    y_val_tier = splits.y_val
    
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

    results_table = []
    
    # --- TASK B: Classification (Price Tier) ---
    print("\nStarting Task B (Classification) sweep...")
    task_b_curves = {"gini": {"train": [], "val": []}, "entropy": {"train": [], "val": []}}
    depths = [2, 4, 6, 8, 10, 12]
    
    for crit in ["gini", "entropy"]:
        for d in depths:
            start_t = time.time()
            tree = DecisionTree(task="classification", criterion=crit, max_depth=d, min_samples_split=10, min_samples_leaf=1)
            tree.fit(X_train, y_train_tier)
            fit_time = time.time() - start_t
            
            tr_preds = tree.predict(X_train)
            val_preds = tree.predict(X_val)
            
            _, _, tr_f1 = precision_recall_f1(y_train_tier, tr_preds)
            _, _, val_f1 = precision_recall_f1(y_val_tier, val_preds)
            
            task_b_curves[crit]["train"].append(tr_f1)
            task_b_curves[crit]["val"].append(val_f1)
            
            results_table.append({
                "Task": "B_Classification", "Step": "depth", "Criterion": crit, "Max_Depth": d,
                "Min_Samples_Leaf": 1, "Train_Metric": tr_f1, "Val_Metric": val_f1, "Fit_Time_s": fit_time
            })

    depth_rows = [r for r in results_table if r["Task"] == "B_Classification"]
    chosen = _select(depth_rows, higher_is_better=True, simplicity=lambda r: r["Max_Depth"])
    best_b_config = {"criterion": chosen["Criterion"], "max_depth": chosen["Max_Depth"], "min_samples_leaf": 1}
    leaf_rows = [chosen]

    print(f"Best Task B base config: {best_b_config}. Sweeping min_samples_leaf...")
    for msl in [5, 20, 50]:
        start_t = time.time()
        tree = DecisionTree(task="classification", criterion=best_b_config["criterion"], 
                            max_depth=best_b_config["max_depth"], min_samples_split=10, min_samples_leaf=msl)
        tree.fit(X_train, y_train_tier)
        fit_time = time.time() - start_t
        
        tr_preds = tree.predict(X_train)
        val_preds = tree.predict(X_val)
        _, _, tr_f1 = precision_recall_f1(y_train_tier, tr_preds)
        _, _, val_f1 = precision_recall_f1(y_val_tier, val_preds)
        
        results_table.append({
            "Task": "B_Classification", "Step": "leaf", "Criterion": best_b_config["criterion"],
            "Max_Depth": best_b_config["max_depth"], "Min_Samples_Leaf": msl,
            "Train_Metric": tr_f1, "Val_Metric": val_f1, "Fit_Time_s": fit_time
        })
        leaf_rows.append(results_table[-1])

    chosen = _select(leaf_rows, higher_is_better=True, simplicity=lambda r: -r["Min_Samples_Leaf"])
    best_b_config["min_samples_leaf"] = chosen["Min_Samples_Leaf"]

    # --- TASK A: Regression (Log Price) ---
    print("\nStarting Task A (Regression) sweep...")
    task_a_curves = {"train": [], "val": []}
    
    for d in depths:
        start_t = time.time()
        tree = DecisionTree(task="regression", criterion="mse", max_depth=d, min_samples_split=10, min_samples_leaf=1)
        tree.fit(X_train, y_train_log)
        fit_time = time.time() - start_t
        
        tr_preds = tree.predict(X_train)
        val_preds = tree.predict(X_val)
        
        tr_rmse = rmse(y_train_log, tr_preds)
        val_rmse = rmse(y_val_log, val_preds)
        
        task_a_curves["train"].append(tr_rmse)
        task_a_curves["val"].append(val_rmse)
        
        results_table.append({
            "Task": "A_Regression", "Step": "depth", "Criterion": "mse", "Max_Depth": d,
            "Min_Samples_Leaf": 1, "Train_Metric": tr_rmse, "Val_Metric": val_rmse, "Fit_Time_s": fit_time
        })

    depth_rows = [r for r in results_table if r["Task"] == "A_Regression"]
    chosen = _select(depth_rows, higher_is_better=False, simplicity=lambda r: r["Max_Depth"])
    best_a_config = {"max_depth": chosen["Max_Depth"], "min_samples_leaf": 1}
    leaf_rows = [chosen]

    print(f"Best Task A base config: {best_a_config}. Sweeping min_samples_leaf...")
    for msl in [5, 20, 50]:
        start_t = time.time()
        tree = DecisionTree(task="regression", criterion="mse", 
                            max_depth=best_a_config["max_depth"], min_samples_split=10, min_samples_leaf=msl)
        tree.fit(X_train, y_train_log)
        fit_time = time.time() - start_t
        
        tr_preds = tree.predict(X_train)
        val_preds = tree.predict(X_val)
        tr_rmse = rmse(y_train_log, tr_preds)
        val_rmse = rmse(y_val_log, val_preds)
        
        results_table.append({
            "Task": "A_Regression", "Step": "leaf", "Criterion": "mse",
            "Max_Depth": best_a_config["max_depth"], "Min_Samples_Leaf": msl,
            "Train_Metric": tr_rmse, "Val_Metric": val_rmse, "Fit_Time_s": fit_time
        })
        leaf_rows.append(results_table[-1])

    chosen = _select(leaf_rows, higher_is_better=False, simplicity=lambda r: -r["Min_Samples_Leaf"])
    best_a_config["min_samples_leaf"] = chosen["Min_Samples_Leaf"]

    # --- LEARNING CURVE: the chosen configs on 1/8, 1/4, 1/2 and all of the
    # subsample (2k, 4k, 8k, 16k rows in full mode). The subsample is in random
    # order, so each prefix is a random subset; scored on validation only.
    sizes = [len(X_train) // 8, len(X_train) // 4, len(X_train) // 2, len(X_train)]
    curve_rows = []
    for k in sizes:
        tree = DecisionTree(task="regression", criterion="mse", max_depth=best_a_config["max_depth"],
                            min_samples_split=10, min_samples_leaf=best_a_config["min_samples_leaf"])
        tree.fit(X_train[:k], y_train_log[:k])
        curve_rows.append({"Task": "A_Regression", "Train_Size": k,
                           "Train_Metric": rmse(y_train_log[:k], tree.predict(X_train[:k])),
                           "Val_Metric": rmse(y_val_log, tree.predict(X_val))})
        tree = DecisionTree(task="classification", criterion=best_b_config["criterion"],
                            max_depth=best_b_config["max_depth"], min_samples_split=10,
                            min_samples_leaf=best_b_config["min_samples_leaf"])
        tree.fit(X_train[:k], y_train_tier[:k])
        curve_rows.append({"Task": "B_Classification", "Train_Size": k,
                           "Train_Metric": precision_recall_f1(y_train_tier[:k], tree.predict(X_train[:k]))[2],
                           "Val_Metric": precision_recall_f1(y_val_tier, tree.predict(X_val))[2]})
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(os.path.join(table_dir, "tree_learning_curve.csv"), index=False)

    # --- OUTPUTS (JSON/CSV) ---
    print("\n--- Final Selected Configurations ---")
    selection_rule = (f"In each sweep step, the best validation score (Task A: RMSE of log price, lower is better; "
                      f"Task B: F1, higher is better); among all configs within {TOLERANCE} of it, the smallest "
                      "max_depth (depth step) or the largest min_samples_leaf (leaf step).")
    print(selection_rule)
    
    df_results = pd.DataFrame(results_table)
    df_results.to_csv(os.path.join(table_dir, "tree_sweep_table.csv"), index=False)
    
    final_config = {
        "config": {
            "tree_task_a": {
                "max_depth": best_a_config["max_depth"],
                "min_samples_split": 10,
                "min_samples_leaf": best_a_config["min_samples_leaf"]
            },
            "tree_task_b": {
                "max_depth": best_b_config["max_depth"],
                "min_samples_split": 10,
                "min_samples_leaf": best_b_config["min_samples_leaf"]
            }
        }
    }
    with open(os.path.join(results_dir, "tree_final_config.json"), 'w') as f:
        json.dump(final_config, f, indent=4)
        
    selection_info = {
        "task_b_criterion": best_b_config["criterion"],
        "selection_rule": selection_rule
    }
    with open(os.path.join(results_dir, "tree_selection.json"), 'w') as f:
        json.dump(selection_info, f, indent=4)
        
    # --- FIGURES (Part 3) ---
    plot_depth_curves(depths, task_a_curves, task_b_curves, figures_dir)
    configs = {"A": f"depth {best_a_config['max_depth']}, leaf {best_a_config['min_samples_leaf']}",
               "B": f"{best_b_config['criterion']}, depth {best_b_config['max_depth']}, "
                    f"leaf {best_b_config['min_samples_leaf']}"}
    plot_learning_curve(curve, configs, figures_dir)
    print(f"Figures saved to {figures_dir}")

if __name__ == "__main__":
    import sys
    quick_mode = "--quick" in sys.argv
    run(quick=quick_mode)