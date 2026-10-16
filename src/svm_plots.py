from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_sweep_results(sweep, output_dir, *, data_label, score_as_percent=False):
    if not isinstance(data_label, str) or not data_label.strip():
        raise ValueError("data_label must identify the plotted data")
    rows = sweep["results"]
    if not rows:
        raise ValueError("The sweep must contain results")
    pairs = [(row["lambda_"], row["gamma"]) for row in rows]
    if len(set(pairs)) != len(rows):
        raise ValueError("Each lambda/gamma combination must be unique")
    values = np.array([[row["lambda_"], row["gamma"], row["validation_score"]] for row in rows])
    if not np.isfinite(values).all() or np.any(values[:, :2] <= 0):
        raise ValueError("Parameters must be positive and scores must be finite")
    if len({row["n_components"] for row in rows}) != 1:
        raise ValueError("Plot a single Fourier feature dimension at a time")
    best = [row for row in rows if row["lambda_"] == sweep["best_params"]["lambda_"]
            and row["gamma"] == sweep["best_params"]["gamma"]]
    if len(best) != 1:
        raise ValueError("The selected configuration must appear in the sweep")
    best = best[0]
    history = best["history"]
    if not history:
        raise ValueError("The selected configuration needs objective history")
    checkpoints = np.array([[row["step"], row["hinge_loss"], row["objective"]] for row in history])
    if (not np.isfinite(checkpoints).all() or np.any(checkpoints < 0)
            or np.any(np.diff(checkpoints[:, 0]) <= 0)):
        raise ValueError("History must contain finite nonnegative values and increasing steps")
    widths = np.array([row["margin_width"] for row in rows])
    if np.isnan(widths).any() or np.any(widths <= 0):
        raise ValueError("Margin widths must be positive or infinite for zero weights")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = ("#2463A6", "#B86A0B", "#697C35", "#A84275", "#5B4D99")
    markers = ("o", "s", "^", "D", "v", "P", "X")
    gammas = sorted({row["gamma"] for row in rows})
    lambdas = sorted({row["lambda_"] for row in rows})
    gamma_colors = {value: palette[i % len(palette)] for i, value in enumerate(gammas)}
    gamma_markers = {value: markers[i % len(markers)] for i, value in enumerate(gammas)}
    lambda_colors = {value: palette[i % len(palette)] for i, value in enumerate(lambdas)}
    lambda_markers = {value: markers[i % len(markers)] for i, value in enumerate(lambdas)}
    direction = "higher is better" if sweep["greater_is_better"] else "lower is better"
    metric_label = sweep["metric_name"].replace("_", " ")
    score_label = f"Validation {metric_label}\n({direction})"
    context = f"{data_label} | RFF + Pegasos | D={rows[0]['n_components']}"
    paths = {}

    def style(ax, validation=False):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D9DDE2", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)
        if validation:
            ax.set_ylabel(score_label)
            if score_as_percent:
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    def save(fig, stem, note):
        FigureCanvasAgg(fig)
        fig.text(0.5, 0.02, note, ha="center", fontsize=9, color="#50565E")
        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
        paths[stem] = {}
        for extension in ("png", "pdf"):
            path = output_dir / f"{stem}.{extension}"
            fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
            paths[stem][extension] = path
        fig.clear()

    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    style(ax)
    ax.plot(checkpoints[:, 0], checkpoints[:, 2], color=palette[0], linewidth=1.7,
            marker="o", markersize=3, label="Regularized objective")
    ax.plot(checkpoints[:, 0], checkpoints[:, 1], color=palette[1], linewidth=1.7,
            linestyle="--", label="Mean hinge loss")
    ax.set(xlabel="Training update", ylabel="Loss", ylim=(0, None))
    ax.set_title(f"Selected model: λ={best['lambda_']:g}, γ={best['gamma']:g}", fontsize=11)
    ax.legend(frameon=False)
    fig.suptitle(f"Training convergence\n{context}", fontsize=12)
    save(fig, "svm_convergence", "Selected using validation only. Stochastic losses may fluctuate between checkpoints.")

    fig = Figure(figsize=(12, 5))
    axes = fig.subplots(1, 2, sharey=True)
    for ax in axes:
        style(ax, validation=True)
    for gamma in gammas:
        group = sorted((row for row in rows if row["gamma"] == gamma), key=lambda row: row["lambda_"])
        axes[0].plot([row["lambda_"] for row in group], [row["validation_score"] for row in group],
                     color=gamma_colors[gamma], marker=gamma_markers[gamma], label=f"γ={gamma:g}")
    for lambda_ in lambdas:
        group = sorted((row for row in rows if row["lambda_"] == lambda_), key=lambda row: row["gamma"])
        axes[1].plot([row["gamma"] for row in group], [row["validation_score"] for row in group],
                     color=lambda_colors[lambda_], marker=lambda_markers[lambda_], label=f"λ={lambda_:g}")
    axes[0].set(xscale="log", xlabel="Regularization λ", title="Regularization sweep")
    axes[1].set(xscale="log", xlabel="RBF bandwidth parameter γ", title="Bandwidth sweep")
    for ax, x in zip(axes, (best["lambda_"], best["gamma"])):
        ax.scatter([x], [best["validation_score"]], marker="*", s=180, facecolors="none",
                   edgecolors="#20252B", linewidths=1.3, zorder=5, label="Selected")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Validation parameter sweeps\n{context}", fontsize=12)
    save(fig, "svm_validation_sweeps", "Each point is a measured configuration; connecting lines are guides. No test-set scores.")

    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    style(ax, validation=True)
    for row in rows:
        if np.isfinite(row["margin_width"]):
            ax.scatter(row["margin_width"], row["validation_score"], s=65,
                       color=gamma_colors[row["gamma"]], marker=lambda_markers[row["lambda_"]],
                       edgecolors="white", linewidths=0.5)
    if np.isfinite(best["margin_width"]):
        ax.scatter(best["margin_width"], best["validation_score"], marker="*", s=230,
                   facecolors="none", edgecolors="#20252B", linewidths=1.3, zorder=5)
    gamma_handles = [Line2D([], [], color=gamma_colors[value], marker="o", linestyle="",
                           label=f"γ={value:g}") for value in gammas]
    lambda_handles = [Line2D([], [], color="#50565E", marker=lambda_markers[value], linestyle="",
                            label=f"λ={value:g}") for value in lambdas]
    selected_handle = Line2D([], [], color="#20252B", marker="*", markerfacecolor="none",
                             linestyle="", markersize=12, label="Selected")
    ax.legend(handles=gamma_handles + lambda_handles + [selected_handle],
              loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
    ax.set(xlabel="Feature-space margin width 2 / ||w||", xlim=(0, None))
    fig.suptitle(f"Margin versus validation performance\n{context}", fontsize=12)
    omitted = int(np.sum(~np.isfinite(widths)))
    note = "RFF feature-space distances; widths across γ describe different feature maps."
    if omitted:
        note += f" Omitted {omitted} infinite width(s) from zero-weight models."
    save(fig, "svm_margin_validation", note)
    return paths
