# src/utils/data_utils.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os


# --------------------------------------------
# 📦 1. Loaders & Savers
# --------------------------------------------
def load_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {path} → shape={df.shape}")
    return df


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Saved data → {path}")


# --------------------------------------------
# 🧹 2. Cleaning Utilities
# --------------------------------------------
def handle_missing_values(df: pd.DataFrame, method: str = "ffill"):
    if method == "drop":
        df = df.dropna()
    else:
        df = df.fillna(method=method)
    return df


def clip_outliers(df: pd.DataFrame, cols, z_thresh=3):
    """Clip values beyond z-threshold."""
    for col in cols:
        mean, std = df[col].mean(), df[col].std()
        df[col] = np.clip(df[col], mean - z_thresh * std, mean + z_thresh * std)
    return df


# --------------------------------------------
# ⚙️ 3. Scaling & Splitting
# --------------------------------------------
def scale_features(df: pd.DataFrame, feature_cols):
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler


def split_sequences(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# --------------------------------------------
# 🧩 4. Time-Series Utilities
# --------------------------------------------
def create_sequences(df: pd.DataFrame, target_col: str, seq_len: int = 30):
    """
    Converts a DataFrame into LSTM-ready sequences.
    """
    X, y = [], []
    data = df.values
    target_idx = df.columns.get_loc(target_col)
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, target_idx])
    return np.array(X), np.array(y)


# --------------------------------------------
# 🧠 5. Merge Utility (safe merge)
# --------------------------------------------
def safe_merge(df1, df2, key, how="left"):
    df = pd.merge(df1, df2, on=key, how=how)
    print(f"[INFO] Merged → new shape: {df.shape}")
    return df


# --------------------------------------------
# 📅 6. Date Handling
# --------------------------------------------
def ensure_datetime(df: pd.DataFrame, col="Date"):
    df[col] = pd.to_datetime(df[col])
    df = df.sort_values(col).reset_index(drop=True)
    return df
