# Contribution Report

This report records each member's responsibilities and completed work. Every
claim is checked against the Git history of the `Khizri` branch, which
contains all merged branches, and against the merged pull requests of
`mmahammad57/binaaz-ml-project`.

## Team

| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Mammadli, Mahammad | mmahammad57 | Data pipeline: loading, cleaning, parsing, leakage removal, encoding, stratified split, training-only tier threshold (P1) |
| Aghazade, Ibrahim | iaghazada15 | From-scratch decision tree: Gini/entropy/MSE, split search, stopping parameters, node statistics, scikit-learn match test (P2); tree hyperparameter study (P4a) |
| Aliyev, Sahib | SahibAliyev5 | From-scratch Pegasos SVM: margin, support vectors, random Fourier features; sweep and plotting tools with tests (P3, P4b tooling) |
| Sakhrayev, Khizri | khizrisakhrayev | Metrics, benchmark vs scikit-learn, SVM tuning on real data, error analysis, EDA, one-command pipeline, README, report and slides (P5–P7) |

## Per-member summary

**Mammadli, Mahammad —** Designed and implemented the data pipeline in
`src/data_prep.py`: CSV loading, the `LEAKAGE_COLUMNS` list, `clean()` (type
conversion, removal of rows missing price, area or rooms, duplicates and
outliers), one-hot encoding in `make_features`, the training-median tier label
`make_tier_label`, the seeded split `train_val_test_split` with stratification
support (`58ac42b`), and `standardize` with training-only statistics. He
debugged parsing that produced missing values for floor, storeys and land area
(`0d899ca`), set floor and storeys to 0 with a `has_floor` flag for houses and
offices (`b79bf9a`), and moved the imputation statistics to the training split
only (`16eb4f2`). His work was merged through PRs #2, #4 and #5.

**Aghazade, Ibrahim —** Implemented the from-scratch decision tree in
`src/decision_tree.py` with Gini, entropy and variance (MSE) impurities, the
exhaustive split search over observed thresholds, the four stopping parameters
(`max_depth`, `min_samples_split`, `min_samples_leaf`,
`min_impurity_decrease`), leaf values with `predict` and `predict_proba`, node
statistics and the scikit-learn match test, and the tree hyperparameter study
(`tree_experiment.py`). The tree and `tests/test_decision_tree.py` came in
`4f550d3` (merged in PR #3) and the per-node sample counts and impurities used
for feature importances in `5ae6e15` (merged through PR #6). For the study he
wrote the data loading and stratified training subsample (`777bcfc`), the
Task A and Task B depth and leaf sweeps with the selection rule and the
JSON/CSV outputs (`eb61bb0`), the depth-curve figure (`3ec561b`) and the
output tests (`422a32e`); `b05f08b` updated the documentation and comments of
`src/decision_tree.py`. He merged PRs #8 and #9 into `main`.

**Aliyev, Sahib —** Implemented a NumPy soft-margin SVM with Codex assistance,
including Pegasos optimization, label encoding, configurable epochs and
learning-rate schedule, objective tracking, margin diagnostics, and
identification of training points on or inside the margin in `src/svm.py`.
Implemented seeded random Fourier features to approximate the RBF kernel
without constructing a full pairwise kernel matrix, and added cached
lambda/gamma sweeps in `src/svm_experiment.py`, PNG/PDF figures in
`src/svm_plots.py`, and usage/method/handoff documentation in `docs/SVM.md`.
Added and verified 33 tests across `tests/test_svm.py`,
`tests/test_svm_experiment.py`, and `tests/test_svm_plots.py`, including exact
updates, nonlinear learning, reproducibility, feature reuse, and plot values.
Evidence includes commits `7dce5df` (RFF transition and sweeps), `f0d173e`
(plots), and `20de4e6` (documentation), merged in PR #1; the real-data sweeps,
final configuration selection and final housing-data figures were then run
with these tools in `src/svm_tuning.py`.

**Sakhrayev, Khizri —** Implemented the NumPy metrics and tree feature
importances in `src/evaluate.py` with scikit-learn cross-check tests, the
matched scikit-learn benchmark with cross-validation, timing and scaling in
`src/benchmark.py` (`a10e5b3`), and the SVM tuning on the real data in
`src/svm_tuning.py`, including the stability-based selection rule and the
cross-validation stability check that exposed the RFF convergence problem
(`155129f`). He wrote the error analysis (`src/analysis.py`), the EDA figures
(`src/eda.py`), the shared stratified split (`src/splits.py`), the
one-command pipeline (`src/run_all.py`), the README, the report
(`report/report.tex`) and the slides (`presentation/slides.tex`). In the data
pipeline he fixed the encoding of the five presence flags (`repair`,
`mortgage`, `bill_of_sale`, `vip`, `featured`), which one-hot encoding with
`drop_first=True` had dropped (`b154faa`). He merged PRs #3, #4 and #5 into
`main`, and his own work was merged through PRs #6 and #8.

## Rough effort split

| Member | Share |
|--------|-------|
| Mammadli, Mahammad | 25% |
| Aghazade, Ibrahim | 25% |
| Aliyev, Sahib | 25% |
| Sakhrayev, Khizri | 25% |

The team agreed on an equal split. Commit and line counts in the Git history
differ widely between members: the integration work (benchmark, tuning,
analysis, pipeline, report) produced the most code and text, and the
decision tree's core algorithm is compact.

| Member | Commits | Merge commits | Lines added | Lines deleted |
|--------|--------:|--------------:|------------:|--------------:|
| Mammadli, Mahammad | 16 | 2 | 315 | 95 |
| Aghazade, Ibrahim | 7 | 2 | 536 | 36 |
| Aliyev, Sahib | 16 | 1 | 1,575 | 305 |
| Sakhrayev, Khizri | 21 | 7 | 5,777 | 1,185 |

Counted with `git log --no-merges --numstat` (and `git log --merges` for the
merge commits) on the `Khizri` branch at commit `931009b`, which contains
every merged branch. Each member's Git identities are combined (Aghazade,
Ibrahim: `ibrahim15` and `Ibrahim Aghazada`; Sakhrayev, Khizri:
`Khizri Sakhrayev` and `khizrisakhrayev`).
The starter-pack commit `97ba518` (687 lines, committed by mmahammad57) is
excluded, and so are binary files. Khizri Sakhrayev's added lines include 360
lines of generated report tables.
