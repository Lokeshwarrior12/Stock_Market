import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils.config import CONFIG


def add_target(df: pd.DataFrame, forecast_horizon: int, target_type: str = "close_price"):
    df = df.copy()

    if target_type.lower() == "close_price":
        df["Target"] = df["Close"].shift(-forecast_horizon)

    elif target_type.lower() == "pct_change":
        df["Target"] = (df["Close"].shift(-forecast_horizon) - df["Close"]) / df["Close"]

    elif target_type.lower() == "direction":
        # Numeric 1 or 0 (not text)
        df["Target"] = np.where(df["Close"].shift(-forecast_horizon) > df["Close"], 1.0, 0.0)

    else:
        raise ValueError("Invalid target_type in config.")

    df.dropna(inplace=True)
    return df



def scale_features(df: pd.DataFrame, method: str = "standard"):
    """
    Scales numeric columns except Date and Target.
    """
    feature_cols = [c for c in df.columns if c not in ["Date", "Target"]]
    if method.lower() == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    scaled = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=df.index)
    scaled_df["Target"] = df["Target"]
    scaled_df["Date"] = df["Date"]
    return scaled_df, scaler


def create_sequences(df: pd.DataFrame, lookback: int, forecast_horizon: int):
    """
    Converts tabular data into supervised learning sequences for DL models.
    """
    feature_cols = [c for c in df.columns if c not in ["Date", "Target"]]
    X, y = [], []

    for i in range(len(df) - lookback - forecast_horizon):
        seq_x = df[feature_cols].iloc[i:(i + lookback)].values
        seq_y = df["Target"].iloc[i + lookback]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def save_datasets(X_train, y_train, X_test, y_test, base_path="data/processed/"):
    os.makedirs(base_path, exist_ok=True)
    np.save(os.path.join(base_path, "X_train.npy"), X_train)
    np.save(os.path.join(base_path, "y_train.npy"), y_train)
    np.save(os.path.join(base_path, "X_test.npy"), X_test)
    np.save(os.path.join(base_path, "y_test.npy"), y_test)
    print(f"[INFO] Saved datasets to {base_path}")


if __name__ == "__main__":
    cfg = CONFIG["dataset"]
    scaling_cfg = CONFIG["scaling"]

    file_path = f"{CONFIG['save_paths']['raw_data']}{CONFIG['stock_symbol']}_technical_data.csv"
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(inplace=True)

    print("[INFO] Preparing target column...")
    df = add_target(df, cfg["forecast_horizon"], cfg["target_type"])

    print("[INFO] Scaling features...")
    df_scaled, scaler = scale_features(df, scaling_cfg["method"])

    print("[INFO] Creating sequences...")
    X, y = create_sequences(df_scaled, cfg["lookback_window"], cfg["forecast_horizon"])

    # Split
    split_idx = int(len(X) * cfg["train_split"])
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"[SUCCESS] Shapes — X_train: {X_train.shape}, X_test: {X_test.shape}")

    save_datasets(X_train, y_train, X_test, y_test, CONFIG["save_paths"]["processed_data"])
