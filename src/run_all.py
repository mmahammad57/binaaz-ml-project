"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all            # full run (8.8 minutes measured on an Apple M4)
    python -m src.run_all --quick    # smoke test on subsamples (about 70 seconds)

Stages, in order (each receives --quick):
  1. EDA             src/eda.py           figures and summary of the training split
  2. SVM tuning      src/svm_tuning.py    train -> validation sweeps, writes the SVM configs
  3. tree tuning     src/tree_experiment.py run(quick), if it exists; otherwise the
                     benchmark uses its fallback tree config (a warning is printed)
  4. benchmark       src/benchmark.py     CV + held-out test vs scikit-learn
  5. analysis        src/analysis.py      results tables, figures, error analysis

All randomness is seeded with data_prep.SEED inside each stage. A full run
writes to results/, report/figures/ and report/tables/; --quick writes under
results/quick/ only, so it never overwrites the report's numbers.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import time
from pathlib import Path

from . import analysis, benchmark, data_prep, eda, svm_tuning
from . import evaluate as ev
from .data_prep import SEED

ROOT = Path(__file__).resolve().parents[1]


def _check_data():
    path = Path(data_prep.DATA_PATH)
    if not path.exists():
        raise SystemExit(
            f"Data file not found: {path} (looked in {Path.cwd()}).\n"
            "Download the bina.az sale dataset from Kaggle and save the CSV as data/bina_az_sale.csv;\n"
            "see the 'Data' section of README.md. Run this command from the repository root."
        )


def _tree_tuning(quick):
    """Call src/tree_experiment.py's run(quick) when it exists; warn otherwise."""
    fallback = benchmark.FALLBACK_CONFIGS

    def warn(why):
        print(f"WARNING: {why}. The benchmark will use the FALLBACK tree config "
              f"(Task A {fallback['tree_task_a']}, Task B {fallback['tree_task_b']}) from "
              "src/benchmark.py. Tuned values belong in results/tree_final_config.json.")

    if importlib.util.find_spec(f"{__package__}.tree_experiment") is None:
        warn("src/tree_experiment.py not found")
        return
    module = importlib.import_module(f"{__package__}.tree_experiment")
    run = getattr(module, "run", None)
    if not callable(run):
        warn("src/tree_experiment.py has no run(quick) function")
        return
    run(quick=quick)


def _headline(bench):
    """Test-set metrics of the final models: our Task A tree, our Task B tree and final SVM."""
    test = bench["test"]
    final_kernel = bench["meta"]["configs"]["svm"]["final"]
    svm_name = f"Pegasos {'linear' if final_kernel == 'linear' else 'RFF'} (ours)"
    print("\nTask A (log price), test set:")
    ev.compare({"Tree (ours)": {k: test["A"]["Tree (ours)"][k]
                                for k in ("RMSE log", "R2 log", "RMSE AZN", "MAE AZN", "R2 AZN")}})
    print("\nTask B (price tier), test set:")
    ev.compare({name: {k: test["B"][name][k] for k in ("Precision", "Recall", "F1", "ROC-AUC")}
                for name in ("Tree gini (ours)", svm_name)})


def _written_since(start_time, folders):
    """Files under `folders` modified during this run, grouped by folder."""
    found = {}
    for folder in folders:
        if folder.exists():
            files = sorted(p for p in folder.rglob("*") if p.is_file() and p.stat().st_mtime >= start_time)
            if files:
                found[folder] = files
    return found


def main(quick: bool = False) -> None:
    _check_data()
    wall_start = time.time()
    print(f"run_all ({'QUICK' if quick else 'FULL'}), SEED={SEED}, data={data_prep.DATA_PATH}")

    stages = [
        ("1. EDA", lambda: eda.run_eda(quick)),
        ("2. SVM tuning", lambda: svm_tuning.run_tuning(quick)),
        ("3. tree tuning", lambda: _tree_tuning(quick)),
        ("4. benchmark", lambda: benchmark.run_benchmark(quick)),
        ("5. analysis", lambda: analysis.run_analysis(quick)),
    ]
    outputs, seconds = {}, {}
    for name, stage in stages:
        print(f"\n{'#' * 78}\n# {name}\n{'#' * 78}")
        start = time.perf_counter()
        outputs[name] = stage()
        seconds[name] = time.perf_counter() - start

    bench = outputs["4. benchmark"]
    print(f"\n{'=' * 78}\nSUMMARY ({'QUICK' if quick else 'FULL'} run)\n{'=' * 78}")
    _headline(bench)

    print("\nConfig sources:")
    for key, source in bench["meta"]["config_sources"].items():
        print(f"  {key:<12} {source}")

    print("\nFiles written:")
    results = ROOT / ("results/quick" if quick else "results")
    folders = [results] if quick else [results, ROOT / "report/figures", ROOT / "report/tables"]
    for folder, files in _written_since(wall_start, folders).items():
        print(f"  {folder.relative_to(ROOT)}/  ({len(files)} files)")
        for path in files:
            print(f"    {path.relative_to(folder)}")

    print("\nTime per stage:")
    for name, s in seconds.items():
        print(f"  {name:<16} {s:7.1f}s")
    print(f"  {'total':<16} {sum(seconds.values()):7.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce every number, table and figure.")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test on subsamples; writes under results/quick/ only")
    main(parser.parse_args().quick)
