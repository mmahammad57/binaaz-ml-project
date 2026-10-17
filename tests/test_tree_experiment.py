import os
import json
from src.tree_experiment import run

def test_tree_experiment_quick():
    run(quick=True)
    
    results_dir = os.path.join("results", "quick")
    config_path = os.path.join(results_dir, "tree_final_config.json")
    selection_path = os.path.join(results_dir, "tree_selection.json")
    
    assert os.path.exists(config_path), "tree_final_config.json is missing!"
    assert os.path.exists(selection_path), "tree_selection.json is missing!"
    
    with open(config_path, "r") as f:
        data = json.load(f)
        
    assert "config" in data, "JSON missing 'config' key"
    config = data["config"]
    
    assert set(config.keys()) == {"tree_task_a", "tree_task_b"}, "Config must only contain tree_task_a and tree_task_b"
    
    for task_key in ["tree_task_a", "tree_task_b"]:
        task_config = config[task_key]
        assert set(task_config.keys()) == {"max_depth", "min_samples_split", "min_samples_leaf"}, f"Invalid keys in {task_key}"
        assert task_config["max_depth"] <= 12, f"max_depth for {task_key} exceeds 12!"
        assert task_config["min_samples_split"] == 10, f"min_samples_split for {task_key} must be 10!"