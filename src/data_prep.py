"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

Keep ALL data wrangling here so the rest of the code just
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


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw frame:
      - drop LEAKAGE_COLUMNS and uninformative metadata,
      - parse complex strings ('Mərtəbə' into floor, total_floors & has_floor flag),
      - parse 'Torpaq sahəsi' into land_area_sot,
      - fix dtypes (prices/areas to numeric; strip currency/text/commas),
      - handle missing values and duplicates,
      - handle outliers with reasonable domain boundaries.
    Note: NO global dataset statistics (e.g. overall medians) are computed here
    to strictly prevent data leakage before the train/val/test split.
    """
    df = df.copy()

    # 1. Drop target leakage and unnecessary metadata columns
    existing_drop_cols = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    df.drop(columns=existing_drop_cols, inplace=True)

    # 2. Parse 'Mərtəbə' (e.g. '3 / 9' -> floor=3, total_floors=9, has_floor=1)
    if 'Mərtəbə' in df.columns:
        has_floor_mask = df['Mərtəbə'].notna() & (
            df['Mərtəbə'].astype(str).str.strip() != ''
        )
        floor_split = df['Mərtəbə'].astype(str).str.split('/', expand=True)

        # Assign 0.0 for missing floor listings (houses, villas, offices)
        df['floor'] = (
            pd.to_numeric(floor_split[0].str.strip(), errors='coerce')
            .fillna(0.0)
            .astype(float)
        )

        if floor_split.shape[1] > 1:
            df['total_floors'] = (
                pd.to_numeric(floor_split[1].str.strip(), errors='coerce')
                .fillna(0.0)
                .astype(float)
            )
        else:
            df['total_floors'] = 0.0

        # Create binary indicator flag: 1 for apartments with floor info, 0 for houses/offices
        df['has_floor'] = has_floor_mask.astype(int)

        df.drop(columns=['Mərtəbə'], inplace=True)

    # 3. Parse 'Torpaq sahəsi' (e.g. '5 sot' -> land_area_sot=5.0)
    if 'Torpaq sahəsi' in df.columns:
        sot_str = (
            df['Torpaq sahəsi']
            .astype(str)
            .str.replace('sot', '', regex=False)
            .str.replace(' ', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        df['land_area_sot'] = pd.to_numeric(sot_str, errors='coerce').fillna(
            0.0
        )
        df.drop(columns=['Torpaq sahəsi'], inplace=True)

    # 4. Convert numeric columns arriving as formatted strings
    numeric_columns = ['price', 'Sahə', 'Otaq sayı', 'lat', 'lng', 'views']
    for col in numeric_columns:
        if col in df.columns:
            s = df[col].astype(str).str.replace('m²', '', regex=False)
            s = s.str.replace(' ', '', regex=False).str.replace(
                ',', '.', regex=False
            )
            df[col] = pd.to_numeric(s, errors='coerce')

    # 5. Handle missing values & remove exact duplicates
    df.dropna(subset=['price', 'Sahə', 'Otaq sayı'], inplace=True)
    df.drop_duplicates(inplace=True)

    # Fill remaining categorical missing values
    categorical_cols = [
        'location',
        'city',
        'Binanın növü',
        'Kateqoriya',
        'Təmir',
        'Çıxarış',
        'İpoteka',
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str).str.strip()

    # 6. Outlier handling based on realistic real-estate price and area ranges
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

    # Presence flags: each column holds a single value or NaN, so get_dummies
    # (drop_first=True) would drop it entirely. Encode 1 = present, 0 = NaN.
    for col in ['repair', 'mortgage', 'bill_of_sale', 'vip', 'featured']:
        if col in df_features.columns:
            df_features[col] = df_features[col].notna().astype(int)

    # Categorical One-Hot Encoding (automatically finds and encodes all string/object columns)
    df_encoded = pd.get_dummies(df_features, drop_first=True)

    feature_names = df_encoded.columns.tolist()
    X = df_encoded.values.astype(np.float64)

    print(f"[Data Prep] Feature matrix constructed. X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_names


def make_tier_label(y_price: np.ndarray, threshold: float | None = None):
    """
    Derived classification target: price TIER (premium vs standard).
    Use the MEDIAN of the TRAINING price as the split threshold.
      - premium = 1 (price >= threshold), standard = 0 (price < threshold).
    Return (y_tier, threshold). The threshold must come from TRAIN only
    (no peeking at val/test) to avoid leakage.
    """
    # Calculate median threshold strictly from the training set if not provided
    if threshold is None:
        threshold = float(np.median(y_price))

    # Binary label: 1 for price >= threshold (premium), 0 otherwise (standard)
    y_tier = (y_price >= threshold).astype(int)
    print(f"[Data Prep] Price-tier binary labels generated with threshold: {threshold:.2f} AZN")
    return y_tier, threshold


def train_val_test_split(X, y, y_tier=None, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic split into train / validation / test.
    Supports stratified splitting on y_tier for Task B when provided.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te, [y_tr_tier, y_val_tier, y_te_tier]).
    """
    stratify_target = y_tier if y_tier is not None else None

    # First split off the test set (stratified if y_tier is provided)
    X_tr_val, X_te, y_tr_val, y_te = sklearn_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify_target
    )

    if y_tier is not None:
        y_tier_tr_val, y_tier_te = sklearn_split(
            y_tier, test_size=test_size, random_state=seed, shuffle=True, stratify=y_tier
        )

    # Calculate relative validation size
    val_relative_size = val_size / (1.0 - test_size)
    stratify_val = y_tier_tr_val if y_tier is not None else None

    # Split remaining train+val set into final train and validation sets
    X_tr, X_val, y_tr, y_val = sklearn_split(
        X_tr_val, y_tr_val, test_size=val_relative_size, random_state=seed, shuffle=True, stratify=stratify_val
    )

    print(f"[Data Prep] Train split: {X_tr.shape[0]} samples")
    print(f"[Data Prep] Validation split: {X_val.shape[0]} samples")
    print(f"[Data Prep] Test split: {X_te.shape[0]} samples")

    if y_tier is not None:
        y_tier_tr, y_tier_val = sklearn_split(
            y_tier_tr_val, test_size=val_relative_size, random_state=seed, shuffle=True, stratify=stratify_val
        )
        return X_tr, y_tr, X_val, y_val, X_te, y_te, y_tier_tr, y_tier_val, y_tier_te

    return X_tr, y_tr, X_val, y_val, X_te, y_te


def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (median impute, mean/std from X_tr),
    then apply the exact same transform to the other splits to prevent leakage.
    Return the scaled arrays in the same order as given.
    """
    X_tr = X_tr.copy()

    # 1. Compute column medians strictly from TRAIN set for any remaining NaNs
    train_medians = np.nanmedian(X_tr, axis=0)
    train_medians = np.nan_to_num(train_medians, nan=0.0)

    # Impute missing values in Train
    inds_tr = np.where(np.isnan(X_tr))
    X_tr[inds_tr] = np.take(train_medians, inds_tr[1])

    # Impute missing values in Val / Test using TRAIN medians
    others_imputed = []
    for X in others:
        X_copy = X.copy()
        inds = np.where(np.isnan(X_copy))
        X_copy[inds] = np.take(train_medians, inds[1])
        others_imputed.append(X_copy)

    # 2. Standardize features using TRAIN mean and standard deviation
    mean = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0)

    # Prevent division by zero for constant features
    std[std == 0.0] = 1.0

    # Scale the training features
    X_tr_scaled = (X_tr - mean) / std
    others_scaled = [(X - mean) / std for X in others_imputed]

    if others_scaled:
        return (X_tr_scaled, *others_scaled)
    return X_tr_scaled


if __name__ == '__main__':
    # Sanity test execution
    raw_df = load_raw()
    cleaned_df = clean(raw_df)
    X, y, features = make_features(cleaned_df)

    # Generate full dataset price-tier for stratified split
    y_tier_full, _ = make_tier_label(y)

    # Stratified split execution
    X_tr, y_tr, X_val, y_val, X_te, y_te, y_tr_tier, y_val_tier, y_te_tier = train_val_test_split(
        X, y, y_tier=y_tier_full
    )

    # Standardize features
    X_tr_sc, X_val_sc, X_te_sc = standardize(X_tr, X_val, X_te)

    # Verify no NaN values exist in design matrices
    assert not np.isnan(X_tr_sc).any(), "NaN values found in X_tr_sc!"
    print("[Data Prep] Verification passed: Zero NaNs in final design matrices.")
    print("[Data Prep] Preprocessing pipeline self-test finished successfully.")