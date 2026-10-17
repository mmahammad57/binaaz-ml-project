"""
eda.py — Exploratory Data Analysis and Figure Generation.
Generates PNG plots for the IEEE report and saves them to report/figures/.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_prep import clean, load_raw, make_tier_label

# Output directory for report figures
FIGURES_DIR = "report/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)


def plot_price_distribution(cleaned_df: pd.DataFrame):
    """Plot raw price and log-transformed price distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Raw price histogram
    sns.histplot(
        cleaned_df["price"] / 1e3, kde=True, ax=axes[0], color="skyblue"
    )
    axes[0].set_title("Price Distribution (in '000 AZN)")
    axes[0].set_xlabel("Price (k AZN)")
    axes[0].set_ylabel("Count")

    # Log-transformed price histogram
    sns.histplot(
        np.log1p(cleaned_df["price"]), kde=True, ax=axes[1], color="salmon"
    )
    axes[1].set_title("Log(Price) Distribution")
    axes[1].set_xlabel("log(1 + Price)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "price_distribution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved figure: {fig_path}")


def plot_area_vs_price(cleaned_df: pd.DataFrame):
    """Plot relationship between property area (m²) and sale price."""
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=cleaned_df,
        x="Sahə",
        y=cleaned_df["price"] / 1e3,
        alpha=0.5,
        color="teal",
    )
    plt.title("Property Area (m²) vs. Price (k AZN)")
    plt.xlabel("Area (m²)")
    plt.ylabel("Price (k AZN)")
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "area_vs_price.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved figure: {fig_path}")


def plot_class_balance(cleaned_df: pd.DataFrame):
    """Plot class balance for binary price tier target (Task B)."""
    y_tier, threshold = make_tier_label(cleaned_df["price"].values)
    plt.figure(figsize=(5, 4))
    counts = np.bincount(y_tier)

    plt.bar(["Standard (0)", "Premium (1)"], counts, color=["#4C72B0", "#C44E52"])
    plt.title(f"Task B Class Balance (Threshold: {threshold:,.0f} AZN)")
    plt.ylabel("Number of Listings")
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "class_balance.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved figure: {fig_path}")


def run_eda():
    print("[EDA] Loading data for EDA...")
    raw_df = load_raw()
    cleaned_df = clean(raw_df)

    plot_price_distribution(cleaned_df)
    plot_area_vs_price(cleaned_df)
    plot_class_balance(cleaned_df)
    print("[EDA] All EDA figures generated successfully!")


if __name__ == "__main__":
    run_eda()