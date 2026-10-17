# Contribution Report

This report records each member's responsibilities and completed work. Every
claim is checked against the Git history of the `Khizri` branch, which
contains all merged branches, and against the merged pull requests of
`mmahammad57/binaaz-ml-project`.

## Team

| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Mammadli, Mahammad | mmahammad57 | Data pipeline: loading, cleaning, parsing, leakage removal, encoding, stratified split, training-only tier threshold (P1) |
| Aghazade, Ibrahim | iaghazada15 | From-scratch decision tree: Gini/entropy/MSE, split search, stopping parameters, node statistics, scikit-learn match test (P2) |
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
`src/decision_tree.py`: Gini, entropy and variance (MSE) impurities, the
exhaustive split search over observed thresholds, the four stopping parameters
(`max_depth`, `min_samples_split`, `min_samples_leaf`,
`min_impurity_decrease`), leaf values with `predict` and `predict_proba`, and
`tests/test_decision_tree.py`, which compares the tree's predictions with
scikit-learn's on toy data (`4f550d3`, merged in PR #3). A follow-up commit
stores each node's sample count and impurity so that feature importances can be
computed (`5ae6e15`, merged through PR #6). The tree hyperparameter sweep was
not completed, so the final tree uses a fixed configuration (maximum depth 10,
at least 10 rows to split, at least 5 rows per leaf).

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
analysis, pipeline, report) produced the most code and text, the decision
tree's core algorithm is compact, and the tree hyperparameter sweep was not
completed.

| Member | Commits | Merge commits | Lines added | Lines deleted |
|--------|--------:|--------------:|------------:|--------------:|
| Mammadli, Mahammad | 16 | 1 | 315 | 95 |
| Aghazade, Ibrahim | 2 | 0 | 238 | 17 |
| Aliyev, Sahib | 16 | 1 | 1,575 | 305 |
| Sakhrayev, Khizri | 12 | 6 | 5,166 | 1,013 |

Counted with `git log --no-merges --numstat` on the `Khizri` branch at commit
`155129f`, which contains every merged branch. Each member's Git identities
are combined (Sakhrayev, Khizri: `Khizri Sakhrayev` and `khizrisakhrayev`).
The starter-pack commit `97ba518` (687 lines, committed by mmahammad57) is
excluded, and so are binary files. Khizri Sakhrayev's added lines include 360
lines of generated report tables.
