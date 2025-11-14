# src/model/forecast.py
import numpy as np
import pandas as pd
import torch
from datetime import timedelta

def make_synthetic_bar(prev_row, predicted_close, atr=None, volatility_factor=0.5):
    """
    Create a synthetic OHLC bar from previous real bar and predicted close.
    - prev_row: Series with Date, Open, High, Low, Close, Volume
    - predicted_close: float
    - atr: optional ATR value to size high/low
    - volatility_factor: fraction of ATR used to create high/low
    Returns a dict that can be appended to a DataFrame.
    """
    o = float(prev_row["Close"])  # use last close as 'open' for next bar
    c = float(predicted_close)
    # approximate high/low from ATR or by small fraction of price movement
    if atr is None or np.isnan(atr) or atr <= 0:
        base = abs(c - o)
        hl_buffer = max(base * 0.5, o * 0.002)  # fallback: 0.2% of price
    else:
        hl_buffer = float(atr) * volatility_factor

    high = max(o, c) + hl_buffer
    low = min(o, c) - hl_buffer
    vol = float(prev_row.get("Volume", 0))  # keep same volume as last bar (caller may scale)
    date = pd.to_datetime(prev_row["Date"]) + timedelta(days=1)
    return {"Date": date, "Open": o, "High": high, "Low": low, "Close": c, "Volume": vol}

# src/model/forecast.py

def iterative_forecast(model, df, tech_cols, fund_cols, lookback_window=60, horizon=5, scaler=None):
    """
    Iteratively predicts next 'horizon' days beyond the available df.
    Supports both LSTM sequence input and tabular fundamental input.
    """
    df = df.copy().reset_index(drop=True)
    preds = []
    last_close = df["Close"].iloc[-1]

    for step in range(horizon):
        if len(df) < lookback_window:
            print(f"[WARN] Not enough data for step {step}")
            break

        # Prepare inputs
        seq_data = df[tech_cols].iloc[-lookback_window:].values
        seq_data = np.expand_dims(seq_data, axis=0)  # (1, lookback, features)
        tab_data = df[fund_cols].iloc[-1:].values   # (1, features)

        # Normalize if scaler is provided
        if scaler is not None:
            seq_data = scaler.transform(seq_data.reshape(-1, seq_data.shape[-1])).reshape(seq_data.shape)

        # Run model prediction
        try:
            pred_price = float(model.predict(seq_data, tab_data)[-1])
        except Exception as e:
            print(f"[ERROR] Forecast step {step} failed: {e}")
            pred_price = np.nan

        # Fallback to last known close
        if np.isnan(pred_price) or pred_price == 0:
            pred_price = last_close

        preds.append(pred_price)
        last_close = pred_price

        # Append pseudo-future row (extend dataframe)
        next_date = pd.to_datetime(df["Date"].iloc[-1]) + pd.Timedelta(days=1)
        new_row = {col: np.nan for col in df.columns}
        new_row["Date"] = next_date
        new_row["Close"] = pred_price
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    future_df = df.iloc[-horizon:].copy()
    future_df["Predicted_Close"] = preds
    return future_df, preds
