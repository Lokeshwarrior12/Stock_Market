# src/features/support_resistance.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from src.utils.config import CONFIG


# ------------------------------------------------------------
# 🔹 1. Detect Support/Resistance Levels (Pivot Logic)
# ------------------------------------------------------------
def detect_pivot_levels(df: pd.DataFrame, left: int = 10, right: int = 10, tolerance: float = 0.005):
    """
    Detect support/resistance levels using local pivot highs and lows
    — similar to TradingView pivot logic.
    """
    highs = df["High"].values
    lows = df["Low"].values

    local_max = argrelextrema(highs, np.greater_equal, order=left)[0]
    local_min = argrelextrema(lows, np.less_equal, order=left)[0]

    levels = []

    for i in local_max:
        level = highs[i]
        if not any(abs(level - l) / level < tolerance for l in levels):
            levels.append(level)

    for i in local_min:
        level = lows[i]
        if not any(abs(level - l) / level < tolerance for l in levels):
            levels.append(level)

    return sorted(levels)


# ------------------------------------------------------------
# 🔹 2. Trend Reversal Detection (MACD + RSI)
# ------------------------------------------------------------
def detect_trend_reversals(df: pd.DataFrame, lookback: int = 100):
    """
    Detect bullish/bearish reversals based on TradingView MACD+RSI concept:
    - Bullish when close crosses below support (oversold RSI)
    - Bearish when close crosses above resistance (overbought RSI)
    """
    df = df.copy()

    # Compute support/resistance levels
    df["resistance"] = df["High"].rolling(window=lookback).max()
    df["support"] = df["Low"].rolling(window=lookback).min()

    # Compute bullish/bearish reversals
    df["bullish_rev"] = (df["Close"].shift(1) > df["support"].shift(1)) & (df["Close"] < df["support"])
    df["bearish_rev"] = (df["Close"].shift(1) < df["resistance"].shift(1)) & (df["Close"] > df["resistance"])

    # Confirm with RSI if available
    if "RSI" in df.columns:
        df.loc[df["bullish_rev"] & (df["RSI"] > 30), "bullish_rev"] = False  # cancel false bullish
        df.loc[df["bearish_rev"] & (df["RSI"] < 70), "bearish_rev"] = False  # cancel false bearish

    # Confirm with MACD if available
    if "MACD" in df.columns and "MACD_signal" in df.columns:
        df.loc[df["bullish_rev"] & (df["MACD"] < df["MACD_signal"]), "bullish_rev"] = False
        df.loc[df["bearish_rev"] & (df["MACD"] > df["MACD_signal"]), "bearish_rev"] = False

    return df


# ------------------------------------------------------------
# 🔹 3. Plot Support/Resistance with Trend Reversals
# ------------------------------------------------------------
def plot_support_resistance(df: pd.DataFrame, chart_days: int = 180):
    """Plot price with support/resistance + reversal markers."""
    df_recent = df.tail(chart_days)
    levels = detect_pivot_levels(df_recent)

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_recent["Date"],
        open=df_recent["Open"],
        high=df_recent["High"],
        low=df_recent["Low"],
        close=df_recent["Close"],
        name="Candles"
    ))

    # Add support/resistance lines
    for level in levels:
        fig.add_hline(y=level, line_dash="dash", line_color="gray", opacity=0.5)

    # Add bullish/bearish reversal markers
    bull = df_recent[df_recent["bullish_rev"]]
    bear = df_recent[df_recent["bearish_rev"]]

    fig.add_trace(go.Scatter(
        x=bull["Date"], y=bull["Close"], mode="markers", name="Bullish Reversal",
        marker=dict(color="green", size=10, symbol="triangle-up")
    ))

    fig.add_trace(go.Scatter(
        x=bear["Date"], y=bear["Close"], mode="markers", name="Bearish Reversal",
        marker=dict(color="red", size=10, symbol="triangle-down")
    ))

    fig.update_layout(
        title=f"Support & Resistance + Reversal Zones ({CONFIG['stock_symbol']})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=750
    )

    fig.show()


# ------------------------------------------------------------
# 🔹 4. Run standalone (for testing)
# ------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(f"{CONFIG['save_paths']['raw_data']}{CONFIG['stock_symbol']}_technical_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)

    print("[INFO] Detecting reversals and levels...")
    df = detect_trend_reversals(df)
    plot_support_resistance(df)
