"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

Fill in the TODOs. Keep ALL data wrangling here so the rest of the code just
calls these functions. Fix your random seed everywhere (see SEED).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_split

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/bina_az_sale.csv"   # change if your filename differs

# Columns that LEAK the target (derived from price). Drop these before training.
# TODO: list the actual leakage columns you find (e.g. a unit/per-m2 price).
LEAKAGE_COLUMNS: list[str] = [
    "unit_price",
    "total_price",
    "id_x",
    "id_y",
    "estate_id",
    "estate_details_id_x",
    "estate_details_id_y",
    "rel_url",
    "estate_rel_url_x",
    "estate_rel_url_y",
    "estate_rel_url",
    "img_url",
    "datetime_scrape_x",
    "datetime_scrape_y",
    "day_x",
    "hour_x",
    "day_y",
    "hour_y",
    "updated",
    "city_when",
    "owner_name",
    "owner_title",
    "shop_name",
    "shop_title",
    "address",
    "description",
    "attributes",
    "extra_info",
    "currency_x",
    "currency_y",
]


def load_raw(path: str = DATA_PATH):
    """Read the raw CSV into a DataFrame (keep UTF-8). Return it unmodified."""
    df = pd.read_csv(path, low_memory=False)
    print(f"[Data Prep] Loaded raw dataset from {path}. Shape: {df.shape}")
    return df


def clean(df):
    """
    Clean the raw frame:
      - fix dtypes (prices/areas to numeric; strip currency/text),
      - handle missing values,
      - drop duplicates,
      - handle outliers (document your rule),
      - drop LEAKAGE_COLUMNS.
    Return the cleaned DataFrame.
    """
    df = df.copy()

    # 1. Drop target leakage and unnecessary metadata columns
    existing_drop_cols = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    df.drop(columns=existing_drop_cols, inplace=True)

    # 2. Convert numeric columns arriving as formatted strings
    numeric_columns = ['price', 'Sahə', 'Otaq sayı', 'Mərtəbə', 'Torpaq sahəsi', 'lat', 'lng', 'views']
    for col in numeric_columns:
        if col in df.columns:
            # Force string cleanup (replace m², spaces, commas) then convert to float
            s = df[col].astype(str).str.replace('m²', '', regex=False)
            s = s.str.replace(' ', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(s, errors='coerce')

    # 3. Handle missing values & remove exact duplicates
    df.dropna(subset=['price', 'Sahə', 'Otaq sayı'], inplace=True)
    df.drop_duplicates(inplace=True)

    # Fill remaining categorical missing values
    categorical_cols = ['location', 'city', 'Binanın növü', 'Kateqoriya', 'Təmir', 'Çıxarış', 'İpoteka']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str).str.strip()

    # 4. Outlier handling based on realistic real-estate price and area ranges
    df = df[(df['price'] >= 10000) & (df['price'] <= 5000000)]
    df = df[(df['Sahə'] >= 10) & (df['Sahə'] <= 2000)]

    print(f"[Data Prep] Cleaning completed. Cleaned shape: {df.shape}")
    return df


def make_features(df):
    """
    Turn the cleaned frame into a numeric design matrix.
      - encode categoricals (one-hot / ordinal — your choice, justify it),
      - assemble X (features) and y (target `price`).
    Return (X, y, feature_names).
    """
    df = df.copy()

    # Separate target variable
    y = df['price'].values.astype(np.float64)
    df_features = df.drop(columns=['price'])

    # Categorical One-Hot Encoding (automatically finds and encodes all string/object columns)
    df_encoded = pd.get_dummies(df_features, drop_first=True)

    feature_names = df_encoded.columns.tolist()
    X = df_encoded.values.astype(np.float64)

    print(f"[Data Prep] Feature matrix constructed. X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_names


def make_tier_label(y_price, threshold=None):
    """
    Derived classification target: price TIER (premium vs standard).
    Use the MEDIAN of the TRAINING price as the split threshold.
      - premium = 1 (price >  threshold), standard = 0.
    Return (y_tier, threshold).  The threshold must come from TRAIN only
    (no peeking at val/test) to avoid leakage.
    """
    raise NotImplementedError("TODO: derive the price-tier label")


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic split into train / validation / test.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    raise NotImplementedError("TODO: split the data")


def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits.
    Return the scaled arrays in the same order as given.
    (Needed for the SVM; trees don't require it.)
    """
    raise NotImplementedError("TODO: standardize using train stats")


if __name__ == '__main__':
    raw_df = load_raw()
    cleaned_df = clean(raw_df)
    X, y, features = make_features(cleaned_df)