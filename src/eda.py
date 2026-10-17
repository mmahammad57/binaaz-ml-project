"""
eda.py — exploratory data analysis of the TRAINING split only.

Run from the repo root:
    python -m src.eda            # figures in report/figures, summary in results/
    python -m src.eda --quick    # same analysis, everything under results/quick/eda/

Only training rows are looked at, so nothing seen here can inform a choice
through validation or test prices. The split is the benchmark's split
(data_prep.train_val_test_split, rebuilt with analysis.rebuild_split).

Outputs
    report/figures/eda_price.{png,pdf}      price and log-price histograms, tier balance
    report/figures/eda_features.{png,pdf}   area vs price, price by rooms, map coloured by log price
    results/eda_summary.csv                 every number quoted from these figures
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_prep
from .analysis import BLUE_RAMP, DOUBLE, FONT, GRID, INK, MUTED, SLOTS, SMALL, rebuild_split

ROOT = Path(__file__).resolve().parents[1]
MAX_ROOMS = 6                 # rooms >= 6 share one box
MAP_QUANTILES = (0.01, 0.99)  # map extent: central 98% of training latitudes / longitudes


def load_training_rows():
    """Cleaned listings of the training split, and the tier threshold (training median)."""
    with contextlib.redirect_stdout(io.StringIO()):
        df = data_prep.clean(data_prep.load_raw(data_prep.DATA_PATH))
        X, price, _ = data_prep.make_features(df)
        train_ids, _, _ = rebuild_split(df, X, price)
        _, threshold = data_prep.make_tier_label(price[train_ids])
    return df.iloc[train_ids].copy(), threshold


def _axes(ncols, height):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # Drawn at the printed width (7 in, full page) so fonts print at 8.5-9 pt.
    fig = Figure(figsize=(DOUBLE, height), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, ncols)
    for ax in axes:
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
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


def _save(fig, folder, stem):
    paths = []
    for ext in ("png", "pdf"):
        path = folder / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
        paths.append(path)
    return paths


def _plain_log_ticks(axis, price=False):
    """Tick labels 10, 100, 1000 or 10k, 100k, 1M: a mathtext 10^k exponent would print below 8 pt."""
    from matplotlib.ticker import FuncFormatter

    def label(v, _):
        if not price:
            return f"{v:g}"
        return f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"
    axis.set_major_formatter(FuncFormatter(label))


def _blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)


def plot_price(train, threshold, folder, summary):
    price = train["price"]
    p99 = price.quantile(0.99)
    fig, axes = _axes(3, 2.3)

    # Price in thousands of AZN; the top 1% is left out so the bulk is visible.
    axes[0].hist(price[price <= p99] / 1e3, bins=60, color=SLOTS[0])
    axes[0].axvline(threshold / 1e3, color=INK, linestyle="--", linewidth=1)
    _label(axes[0], "Price (top 1% not shown)", "Price, thousand AZN", "Listings")

    axes[1].hist(np.log(price), bins=60, color=SLOTS[0])
    axes[1].axvline(np.log(threshold), color=INK, linestyle="--", linewidth=1)
    _label(axes[1], "log(price), the Task A target", "log(price / AZN)", "Listings")

    premium = price >= threshold
    counts = [int((~premium).sum()), int(premium.sum())]
    bars = axes[2].bar(["Standard", "Premium"], counts, color=[SLOTS[0], SLOTS[1]], width=0.6)
    for bar, count in zip(bars, counts):
        axes[2].annotate(f"{count:,}\n({count / len(price):.1%})", (bar.get_x() + bar.get_width() / 2, count),
                         xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                         fontsize=SMALL, color="white")
    axes[2].grid(axis="x", visible=False)
    _label(axes[2], "Task B tiers", None, "Listings")

    summary.update(price_p99=p99, rows_above_p99=int((price > p99).sum()),
                   n_standard=counts[0], n_premium=counts[1])
    return _save(fig, folder, "eda_price")


def plot_features(train, folder, summary):
    fig, axes = _axes(3, 2.5)
    price, area = train["price"], train["Sahə"]

    axes[0].hexbin(area, price, xscale="log", yscale="log", gridsize=40, bins="log", mincnt=1,
                   cmap=_blue_cmap(), linewidths=0)
    _label(axes[0], "Floor area vs price", "Floor area, m² (log)", "Price, AZN (log)")
    _plain_log_ticks(axes[0].xaxis)
    _plain_log_ticks(axes[0].yaxis, price=True)

    rooms = train["Otaq sayı"].clip(upper=MAX_ROOMS).astype(int)
    keys = sorted(rooms.unique())
    axes[1].boxplot([price[rooms == k] for k in keys], showfliers=False, widths=0.6,
                    tick_labels=[f"{k}+" if k == MAX_ROOMS else str(k) for k in keys],
                    medianprops=dict(color=SLOTS[1], linewidth=1.5),
                    boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    axes[1].set_yscale("log")
    _plain_log_ticks(axes[1].yaxis, price=True)
    _label(axes[1], "Price by rooms (no outliers drawn)", "Rooms", "Price, AZN (log)")

    lat_lo, lat_hi = train["lat"].quantile(MAP_QUANTILES)
    lng_lo, lng_hi = train["lng"].quantile(MAP_QUANTILES)
    inside = train["lat"].between(lat_lo, lat_hi) & train["lng"].between(lng_lo, lng_hi)
    shown = train[inside].sort_values("price")      # expensive listings drawn last, on top
    points = axes[2].scatter(shown["lng"], shown["lat"], c=np.log(shown["price"]), s=0.6, linewidths=0,
                             cmap=_blue_cmap(), rasterized=True)
    axes[2].set_aspect(1 / np.cos(np.radians(shown["lat"].mean())))   # equal ground distances
    bar = fig.colorbar(points, ax=axes[2], shrink=0.9)
    bar.set_label("log(price)", fontsize=SMALL, color=INK)
    bar.ax.tick_params(labelsize=SMALL, colors=MUTED)
    _label(axes[2], "Location", "Longitude", "Latitude")

    summary.update(corr_log_area_log_price=float(np.corrcoef(np.log(area), np.log(price))[0, 1]),
                   map_rows_shown=int(inside.sum()), map_rows_outside=int((~inside).sum()))
    for k in keys:
        summary[f"median_price_rooms_{k}{'_plus' if k == MAX_ROOMS else ''}"] = float(price[rooms == k].median())
    return _save(fig, folder, "eda_features")


def run_eda(quick: bool = False) -> dict:
    results = ROOT / ("results/quick/eda" if quick else "results")
    figures = results if quick else ROOT / "report/figures"
    for folder in (results, figures):
        folder.mkdir(parents=True, exist_ok=True)

    train, threshold = load_training_rows()
    price = train["price"]
    summary = {
        "n_train": len(train), "tier_threshold": threshold,
        "price_min": price.min(), "price_median": price.median(), "price_mean": price.mean(),
        "price_max": price.max(), "price_skew": price.skew(), "log_price_skew": np.log(price).skew(),
        "area_median": train["Sahə"].median(),
    }
    paths = plot_price(train, threshold, figures, summary) + plot_features(train, figures, summary)
    out = results / "eda_summary.csv"
    pd.DataFrame({"key": list(summary), "value": list(summary.values())}).to_csv(out, index=False)
    print(f"EDA on the training split ({len(train):,} rows): wrote "
          + ", ".join(sorted({p.stem for p in paths})) + f" (.png/.pdf) and {out.relative_to(ROOT)}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA figures from the training split.")
    parser.add_argument("--quick", action="store_true", help="write under results/quick/eda/")
    run_eda(parser.parse_args().quick)
