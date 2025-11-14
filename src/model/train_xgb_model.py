import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils.config import CONFIG


# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------

def compute_technical_features(df: pd.DataFrame):
    """Recalculate core indicators for iterative forecasting."""
    df["EMA_short"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_long"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["MACD"] = df["EMA_short"] - df["EMA_long"]
    df["ATR"] = compute_atr(df)
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = high_low.combine(high_close, np.maximum).combine(low_close, np.maximum)
    return tr.rolling(period).mean()


# ----------------------------------------------------------
# Load and Prepare Data
# ----------------------------------------------------------
def load_tabular_data(symbol: str):
    """Load processed dataset and return numeric X, y, scaler, and original df (raw)"""
    data_path = os.path.join(CONFIG["save_paths"]["processed_data"], f"{symbol}_merged_features.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] Processed data not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    if "Target" not in df.columns:
        raise ValueError("Processed dataset must contain a 'Target' column.")

    # Keep a copy of original (with price columns) for iterative feature recomputation
    df_orig = df.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df_orig.columns:
            df_orig[c] = pd.to_numeric(df_orig[c], errors="coerce")

    # Build X,y using numeric columns only (the model expects these)
    df_num = df.select_dtypes(include=[np.number]).copy()
    df_num.fillna(df_num.median(), inplace=True)

    X = df_num.drop(columns=["Target"])
    y = df_num["Target"]

    # Fit scaler on X so we can re-scale future feature rows later
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X, y, scaler, df_orig

def autoregressive_forecast(model, df_orig, feature_cols, scaler, forecast_days=5, volatility_scale=0.5):
    """
    model: trained xgboost Booster
    df_orig: original DataFrame with OHLCV and other columns (not scaled)
    feature_cols: list of columns used by model (in the order model expects)
    scaler: fitted StandardScaler used at training-time for X
    returns: list of predicted closes, df_future (synthetic appended rows)
    """
    df_work = df_orig.copy().reset_index(drop=True)
    # ensure numeric price columns
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df_work.columns:
            df_work[c] = pd.to_numeric(df_work[c], errors="coerce")

    # helper to recompute minimal indicators that your model uses (adjust to real indicators in your features)
    def recompute_indicators(df):
        # EMA short/long used in your features
        if "Close" in df.columns:
            df["EMA_short"] = df["Close"].ewm(span=9, adjust=False).mean()
            df["EMA_long"] = df["Close"].ewm(span=21, adjust=False).mean()
            # RSI
            delta = df["Close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            df["RSI"] = 100 - (100 / (1 + rs))
            # MACD simple
            df["MACD"] = df["EMA_short"] - df["EMA_long"]
            # ATR (basic)
            high_low = df["High"] - df["Low"]
            high_close = (df["High"] - df["Close"].shift()).abs()
            low_close = (df["Low"] - df["Close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df["ATR"] = tr.rolling(14).mean()
        return df

    recompute_indicators(df_work)

    last_index = df_work.index[-1]
    preds = []
    synthetic_rows = []

    for step in range(forecast_days):
        last_row = df_work.iloc[-1]
        last_close = float(last_row["Close"])
        last_atr = float(last_row["ATR"]) if not pd.isna(last_row.get("ATR", np.nan)) else max(0.01, 0.01 * last_close)

        # Simulate next raw OHLCV bar:
        # open = last close; we create a high/low around predicted movement scale
        # Use atr * volatility_scale to create realistic high/low range
        open_price = last_close
        # set a small drift for exploration; actual predicted close will substitute
        high_sim = last_close + abs(last_atr) * (1.2 * volatility_scale)
        low_sim = last_close - abs(last_atr) * (1.2 * volatility_scale)
        vol_sim = max(1.0, last_row.get("Volume", 1) * (1.0 + np.random.randn() * 0.01))

        # create placeholder new row
        new_row = last_row.copy()
        new_row["Date"] = pd.to_datetime(last_row.get("Date")) + pd.Timedelta(days=1)
        new_row["Open"] = open_price
        # temporarily set Close equal to open; we'll replace by predicted close
        new_row["Close"] = open_price
        new_row["High"] = high_sim
        new_row["Low"] = low_sim
        new_row["Volume"] = vol_sim

        # append and recompute indicators so feature vector evolves
        df_work = pd.concat([df_work, pd.DataFrame([new_row])], ignore_index=True)
        df_work = recompute_indicators(df_work)

        # build feature row using last appended row (after recompute)
        feat_row = df_work[feature_cols].iloc[[-1]].copy()  # keep shape (1, n_features)

        # Some models expect no NaNs. Fill conservatively:
        feat_row = feat_row.fillna(method="ffill").fillna(0)

        # scale using scaler (scaler was fitted on training X)
        feat_scaled = scaler.transform(feat_row.values)

        # Create DMatrix with correct feature names
        dmat = xgb.DMatrix(pd.DataFrame(feat_scaled, columns=feature_cols))

        # predict next close
        next_close = float(model.predict(dmat)[0])
        preds.append(next_close)

        # update the appended synthetic row's Close (and adjust High/Low around predicted close)
        df_work.loc[df_work.index[-1], "Close"] = next_close
        # adjust high/low to include the predicted close if necessary
        df_work.loc[df_work.index[-1], "High"] = max(df_work.loc[df_work.index[-1], "High"], next_close)
        df_work.loc[df_work.index[-1], "Low"] = min(df_work.loc[df_work.index[-1], "Low"], next_close)

        synthetic_rows.append(df_work.iloc[-1].to_dict())

        # recompute indicators again to propagate the predicted close into next iteration
        df_work = recompute_indicators(df_work)

    df_future = pd.DataFrame(synthetic_rows)
    return preds, df_future


# ----------------------------------------------------------
# XGBoost Training
# ----------------------------------------------------------
def train_xgb_model(X, y, params, save_path, forecast_days: int = 5):
    """Train an XGBoost regressor and generate autoregressive forecasts."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.get("test_size", 0.2), shuffle=False)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Training setup
    train_params = params.get("train_params", {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.02,
        "max_depth": 8,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "alpha": 0.1,
        "seed": 42,
        "tree_method": "hist",
    })

    print(f"[INFO] Training XGBoost model with {params.get('num_boost_round', 800)} rounds...")
    model = xgb.train(
        params=train_params,
        dtrain=dtrain,
        num_boost_round=params.get("num_boost_round", 800),
        evals=[(dtrain, "Train"), (dtest, "Test")],
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        verbose_eval=100,
    )

    print("[INFO] Evaluating model performance...")
    preds = model.predict(dtest)

    try:
        rmse = mean_squared_error(y_test, preds, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_test, preds) ** 0.5

    mae = mean_absolute_error(y_test, preds)
    direction_acc = np.mean(
        np.sign(preds[1:] - preds[:-1]) == np.sign(y_test.values[1:] - y_test.values[:-1])
    )

    print(f"[RESULT] RMSE={rmse:.4f} | MAE={mae:.4f} | Directional Accuracy={direction_acc:.3f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path)
    print(f"[SUCCESS] Model saved to {save_path}")

    # ----------------------------------------------------------
    # Autoregressive Future Forecasting
    # ----------------------------------------------------------
    print(f"[INFO] Generating {forecast_days}-day rolling forecast...")
    future_preds = []
    df_future = X.copy()
    last_close = float(y.iloc[-1])

    for i in range(forecast_days):
        dmatrix = xgb.DMatrix(df_future.tail(1))
        next_close = float(model.predict(dmatrix)[0])
        future_preds.append(next_close)

        # simulate new bar
        new_row = df_future.tail(1).copy()
        new_row.iloc[0, 0] = next_close  # assume first feature = close
        df_future = pd.concat([df_future, new_row], ignore_index=True)

    print(f"[FORECAST] Next {forecast_days} predicted closes: {np.round(future_preds, 4)}")

    return model, rmse, mae, direction_acc, future_preds


# ----------------------------------------------------------
# Main Entry Point
# ----------------------------------------------------------
if __name__ == "__main__":
    cfg = CONFIG["xgboost"]
    symbol = CONFIG["stock_symbol"]

    print(f"[INFO] Loading processed data for {symbol}...")
    X, y, scaler, df_orig = load_tabular_data(symbol)

    model, rmse, mae, acc, preds = train_xgb_model(
        X, y, cfg, cfg["model_save_path"], forecast_days=5
    )

    print("\n[SUMMARY]")
    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ MAE: {mae:.4f}")
    print(f"✅ Directional Accuracy: {acc:.3f}")
    print(f"📈 Forecast (next 5 days): {np.round(preds, 3)}")
    print("[DONE] XGBoost training and dynamic forecasting complete ✅")
