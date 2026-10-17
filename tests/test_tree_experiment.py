"""
tree_experiment.run end to end on a tiny training subsample. Every output goes
to pytest's tmp_path; skipped when the raw CSV is absent (as in test_splits.py).
"""

import json
from pathlib import Path

import pytest

from src import data_prep
from src.tree_experiment import run


@pytest.mark.skipif(not Path(data_prep.DATA_PATH).exists(), reason="raw CSV not present")
def test_tree_experiment_tiny_run(tmp_path):
    run(quick=True, n_samples=400, out_dir=tmp_path)

    data = json.loads((tmp_path / "tree_final_config.json").read_text())
    assert set(data) == {"config"}, "JSON must contain only the 'config' key"
    config = data["config"]
    assert set(config) == {"tree_task_a", "tree_task_b"}, "Config must only contain tree_task_a and tree_task_b"
    for task_key in ("tree_task_a", "tree_task_b"):
        task_config = config[task_key]
        assert set(task_config) == {"max_depth", "min_samples_split", "min_samples_leaf"}, f"Invalid keys in {task_key}"
        assert task_config["max_depth"] <= 12, f"max_depth for {task_key} exceeds 12!"
        assert task_config["min_samples_split"] == 10, f"min_samples_split for {task_key} must be 10!"

    selection = json.loads((tmp_path / "tree_selection.json").read_text())
    assert selection["task_b_criterion"] in ("gini", "entropy")
    for name in ("tree_sweep_table.csv", "tree_learning_curve.csv",
                 "tree_depth_curves.png", "tree_depth_curves.pdf",
                 "tree_learning_curve.png", "tree_learning_curve.pdf"):
        assert (tmp_path / name).exists(), f"{name} is missing"
