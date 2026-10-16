# Contribution Report

This report records assigned responsibilities and completed work. Teammates'
names, GitHub handles, and individual summaries still need to be supplied.

## Team

| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Name pending | To confirm | Data preprocessing, EDA, leakage removal, encoding, stratified splits, training-only tier threshold, shared metrics, results table and error analysis. |
| Name pending | To confirm | Decision tree from scratch, split criteria, stopping rules, probabilities and feature importances. |
| Aliyev, Sahib | SahibAliyev5 | NumPy Pegasos SVM, random Fourier features, regularization and bandwidth sweeps, convergence and margin/validation plots, tests and implementation documentation. |
| Name pending | To confirm | Tree tuning, sklearn benchmarks, cross-validation, timing, reproducibility, pipeline integration, and report and slides assembly. |

## Per-member summary

**Name pending — Data preparation and evaluation:** Completed-work summary to
be supplied by the member.

**Name pending — Decision tree:** Completed-work summary to be supplied by the
member.

**Aliyev, Sahib —** Implemented a NumPy soft-margin SVM with Codex assistance,
including Pegasos optimization, label encoding, configurable
epochs and learning-rate schedule, objective tracking, margin diagnostics, and
identification of training points on or inside the margin in `src/svm.py`.
Implemented seeded random Fourier features to approximate the RBF kernel
without constructing a full pairwise kernel matrix, and added cached
lambda/gamma sweeps in `src/svm_experiment.py`, PNG/PDF figures in
`src/svm_plots.py`, and usage/method/handoff documentation in `docs/SVM.md`.
Added and verified 33 tests across `tests/test_svm.py`,
`tests/test_svm_experiment.py`, and `tests/test_svm_plots.py`, including exact
updates, nonlinear learning, reproducibility, feature reuse, and plot values.
Evidence includes commits `7dce5df` (RFF transition and sweeps), `f0d173e`
(plots), and `20de4e6` (documentation); real-data sweeps, final configuration
selection, and final housing-data figures remain pending the shared
preprocessing and metric implementations.

**Name pending — Benchmarking and integration:** Completed-work summary to be
supplied by the member.

## Rough effort split

The team planned an approximately equal workload across four members, with
Sahib responsible for the SVM implementation and its tuning experiments.
The provisional shares below sum to 100%. The team should confirm or revise
them after integration to reflect actual effort and document any imbalance.

| Member | Planned share (provisional) |
|--------|----------------------------|
| Name pending — data preparation and evaluation | 25% |
| Name pending — decision tree | 25% |
| Aliyev, Sahib | 25% |
| Name pending — benchmarking and integration | 25% |
