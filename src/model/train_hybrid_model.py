# src/model/train_hybrid_model.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from src.utils.config import CONFIG
import joblib    # pip install joblib (often installed with sklearn)


# ----------------------------
# Hybrid Model Architecture
# ----------------------------
class HybridModel(nn.Module):
    def __init__(self, input_dim_seq, input_dim_fund, hidden_dim=128, num_layers=2, dropout=0.2):
        super(HybridModel, self).__init__()

        # LSTM branch for technical (time-series)
        self.lstm = nn.LSTM(input_dim_seq, hidden_dim, num_layers=num_layers,
                            dropout=dropout, batch_first=True)

        # Feedforward branch for fundamentals
        self.fund_net = nn.Sequential(
            nn.Linear(input_dim_fund, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Combined layer
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_seq, x_fund):
        # LSTM branch
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = lstm_out[:, -1, :]  # last timestep output

        # Fundamentals branch
        fund_out = self.fund_net(x_fund)

        # Concatenate both representations
        combined = torch.cat((lstm_out, fund_out), dim=1)
        out = self.fc_combined(combined)
        return out

    def predict(self, X_seq, X_fund, device="cpu"):
        """Wrapper for inference: accepts numpy arrays or tensors, returns numpy preds."""
        self.eval()
        with torch.no_grad():
            if not isinstance(X_seq, torch.Tensor):
                X_seq = torch.tensor(X_seq, dtype=torch.float32)
            if not isinstance(X_fund, torch.Tensor):
                X_fund = torch.tensor(X_fund, dtype=torch.float32)
            X_seq = X_seq.to(device)
            X_fund = X_fund.to(device)
            out = self.forward(X_seq, X_fund)
            return out.cpu().numpy().flatten()


# ----------------------------
# Data Preparation
# ----------------------------
def _safe_to_numeric_df(df, cols):
    """coerce columns to numeric, impute column means for NaNs, drop columns completely NaN."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    # drop columns that are all NaN
    for c in cols:
        if df[c].isna().all():
            df.drop(columns=[c], inplace=True)
    # impute column mean
    for c in df.select_dtypes(include=[np.number]).columns:
        col = df[c].values
        if np.isfinite(col).any():
            mean = np.nanmean(col)
            col = np.nan_to_num(col, nan=mean)
            df[c] = col
        else:
            df[c] = 0.0
    return df


def prepare_data(symbol: str, lookback_window=60, forecast_horizon=5, train_split=0.8):
    """
    Loads merged features, cleans, scales, and returns train/test splits.
    Automatically ignores missing or non-numeric fundamental columns.
    """
    path = f"data/processed/{symbol}_merged_features.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # expected technical columns (these may or may not exist)
    tech_cols = [c for c in ["Open", "High", "Low", "Close", "Volume",
                             "RSI", "EMA_short", "EMA_long", "MACD", "ATR"]
                 if c in df.columns]

    if not tech_cols:
        raise ValueError("No valid technical columns found in dataset.")

    # candidate fundamental columns = all numeric columns not in tech_cols + ['Target']
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    fund_cols = [c for c in numeric_cols if c not in tech_cols + ["Target"]]

    # ✅ Only keep columns that actually exist in df
    fund_cols = [c for c in fund_cols if c in df.columns]

    print(f"[INFO] Found {len(tech_cols)} technical and {len(fund_cols)} fundamental features.")
    if len(fund_cols) == 0:
        print("[WARN] No fundamental columns found — will train using only technical features.")

    # Clean numeric
    df = df.copy()
    for c in tech_cols + fund_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.fillna(method="ffill", inplace=True)
    df.fillna(0, inplace=True)

    # Scale
    scaler_tech = StandardScaler()
    scaler_fund = StandardScaler()

    df_tech = scaler_tech.fit_transform(df[tech_cols])
    df_fund = scaler_fund.fit_transform(df[fund_cols]) if fund_cols else np.zeros((len(df), 1))

    os.makedirs("models/checkpoints", exist_ok=True)
    joblib.dump(scaler_tech, "models/checkpoints/scaler_tech.pkl")
    joblib.dump(scaler_fund, "models/checkpoints/scaler_fund.pkl")

    # Target: forecast next-day close
    if "Close" not in df.columns:
        raise KeyError("Expected 'Close' column not found for target creation.")
    y = df["Close"].shift(-forecast_horizon).dropna().values

    # Align shapes
    df_tech = df_tech[:len(y)]
    df_fund = df_fund[:len(y)]

    # Sequence construction
    X_seq, X_fund, Y = [], [], []
    for i in range(len(y) - lookback_window):
        X_seq.append(df_tech[i:i + lookback_window])
        X_fund.append(df_fund[i + lookback_window])
        Y.append(y[i + lookback_window])

    X_seq, X_fund, Y = np.array(X_seq), np.array(X_fund), np.array(Y)

    # Split
    split = int(len(X_seq) * train_split)
    X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
    X_train_fund, X_test_fund = X_fund[:split], X_fund[split:]
    y_train, y_test = Y[:split], Y[split:]

    return X_train_seq, X_test_seq, X_train_fund, X_test_fund, y_train, y_test


# ----------------------------
# Training Loop
# ----------------------------
def train_hybrid(model, loaders, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_seq, X_fund, y in loaders:
        X_seq, X_fund, y = X_seq.to(device), X_fund.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X_seq, X_fund)
        loss = criterion(preds, y)
        if not torch.isfinite(loss):
            print("[ERROR] Encountered non-finite loss; stopping training step.")
            return float("nan")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


def evaluate(model, X_seq, X_fund, y, device):
    model.eval()
    with torch.no_grad():
        preds = model(X_seq.to(device), X_fund.to(device)).cpu().numpy().flatten()
    y_true = y.numpy().flatten()
    rmse = np.sqrt(np.mean((y_true - preds) ** 2))
    mae = np.mean(np.abs(y_true - preds))
    return rmse, mae


# ----------------------------
# Iterative multi-step forecast
# ----------------------------
def iterative_forecast(model, df, tech_cols, fund_cols, lookback_window=60, horizon=1, device="cpu"):
    """
    Naive iterative forecast:
      - uses last `lookback_window` technical rows (scaled using saved scaler_tech)
      - uses latest fundamentals (scaled using saved scaler_fund)
      - for each step, predict next close, construct a synthetic bar, append to df and continue
    Returns: df_future (DataFrame), preds (list of predicted closes)
    NOTE: For best results you MUST use the same scalers used during training (we saved them).
    """
    # load scalers
    scaler_tech_path = "models/checkpoints/scaler_tech.pkl"
    scaler_fund_path = "models/checkpoints/scaler_fund.pkl"
    if not os.path.exists(scaler_tech_path) or not os.path.exists(scaler_fund_path):
        raise FileNotFoundError("Scalers not found. Train the model first to create scalers.")

    scaler_tech = joblib.load(scaler_tech_path)
    scaler_fund = joblib.load(scaler_fund_path)

    df_local = df.copy().reset_index(drop=True)
    # ensure numeric
    for c in tech_cols + fund_cols:
        if c in df_local.columns:
            df_local[c] = pd.to_numeric(df_local[c], errors="coerce")
    df_local.fillna(method="ffill", inplace=True)
    df_local.fillna(0, inplace=True)

    preds = []
    rows = []

    for step in range(horizon):
        tail = df_local[tech_cols].tail(lookback_window).values.astype(float)
        if tail.shape[0] < lookback_window:
            raise ValueError("Not enough rows for lookback window.")
        tail_s = scaler_tech.transform(tail)  # (lookback_window, n_feats)
        X_seq = np.expand_dims(tail_s, axis=0)  # (1, lookback_window, n_feats)

        last_fund = df_local[fund_cols].iloc[-1].values.astype(float).reshape(1, -1)
        last_fund_s = scaler_fund.transform(last_fund)

        # predict
        try:
            pred = model.predict(X_seq, last_fund_s, device=device)[0]
        except Exception as e:
            print("[ERROR] Forecast step failed:", e)
            pred = np.nan

        preds.append(pred)

        # create synthetic next bar:
        last_close = float(df_local["Close"].iloc[-1])
        # naive bar construction:
        next_open = last_close
        next_close = float(pred)
        next_high = max(next_open, next_close) * (1 + 0.001)  # tiny wiggle
        next_low = min(next_open, next_close) * (1 - 0.001)
        next_volume = float(df_local["Volume"].iloc[-5:].mean()) if "Volume" in df_local.columns else 0.0

        next_date = pd.to_datetime(df_local["Date"].iloc[-1]) + pd.Timedelta(1, unit="D")
        new_row = {c: 0.0 for c in df_local.columns}
        if "Date" in df_local.columns:
            new_row["Date"] = next_date
        if "Open" in df_local.columns:
            new_row["Open"] = next_open
        if "High" in df_local.columns:
            new_row["High"] = next_high
        if "Low" in df_local.columns:
            new_row["Low"] = next_low
        if "Close" in df_local.columns:
            new_row["Close"] = next_close
        if "Volume" in df_local.columns:
            new_row["Volume"] = next_volume

        # for fund cols, keep same last value (could be improved by modeling fundamentals)
        for fc in fund_cols:
            if fc in df_local.columns:
                new_row[fc] = float(df_local[fc].iloc[-1])

        rows.append(new_row)
        df_local = pd.concat([df_local, pd.DataFrame([new_row])], ignore_index=True)

    df_future = pd.DataFrame(rows)
    return df_future, preds


# ----------------------------
# Main Training Execution
# ----------------------------
if __name__ == "__main__":
    cfg = CONFIG["hybrid_model"]
    symbol = CONFIG["stock_symbol"]

    print("[INFO] Loading and preparing merged dataset...")
    X_train_seq, X_test_seq, X_train_fund, X_test_fund, y_train, y_test = prepare_data(
        symbol,
        lookback_window=10,   # smaller for example/debug
        forecast_horizon=1
    )

    # Convert to tensors
    X_train_seq = torch.tensor(X_train_seq, dtype=torch.float32)
    X_test_seq = torch.tensor(X_test_seq, dtype=torch.float32)
    X_train_fund = torch.tensor(X_train_fund, dtype=torch.float32)
    X_test_fund = torch.tensor(X_test_fund, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print("\n[DEBUG] Shape summary:")
    print("X_train_seq:", X_train_seq.shape)
    print("X_train_fund:", X_train_fund.shape)
    print("y_train:", y_train.shape)

    # Model setup
    input_dim_seq = X_train_seq.shape[2]
    input_dim_fund = X_train_fund.shape[1]
    model = HybridModel(input_dim_seq, input_dim_fund, hidden_dim=cfg["hidden_dim"],
                        num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)

    lr = cfg.get("learning_rate", cfg.get("learning", 1e-3))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Dataloader
    if len(X_train_seq) == 0:
        raise ValueError("No training samples available after preprocessing.")

    train_data = TensorDataset(X_train_seq, X_train_fund, y_train)
    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)

    print("[INFO] Training hybrid model...")
    for epoch in range(cfg["epochs"]):
        loss = train_hybrid(model, train_loader, criterion, optimizer, device)
        if np.isnan(loss):
            print("[ERROR] Training produced NaN loss — stopping training. Inspect data/scalers.")
            break
        rmse, mae = evaluate(model, X_test_seq, X_test_fund, y_test, device)
        print(f"Epoch [{epoch+1}/{cfg['epochs']}]: Loss={loss:.6f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    os.makedirs(os.path.dirname(cfg["model_save_path"]), exist_ok=True)
    torch.save(model.state_dict(), cfg["model_save_path"])
    print(f"[SUCCESS] Hybrid model saved to {cfg['model_save_path']}")
