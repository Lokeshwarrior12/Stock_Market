# src/signals/support_resistance_strategy.py

import pandas as pd
import numpy as np

def detect_sr_levels(df, lookback=100):
    """Find simple support & resistance using highest/lowest of last N candles."""
    df["resistance"] = df["High"].rolling(window=lookback).max()
    df["support"] = df["Low"].rolling(window=lookback).min()
    return df

def detect_pivots(df, left=10, right=10):
    """Pivot-based support/resistance (TradingView style)."""
    pivots_high = (df["High"].shift(left) == df["High"].rolling(window=left+right+1, center=True).max())
    pivots_low = (df["Low"].shift(left) == df["Low"].rolling(window=left+right+1, center=True).min())

    df["pivot_resistance"] = np.where(pivots_high, df["High"].shift(left), np.nan)
    df["pivot_support"] = np.where(pivots_low, df["Low"].shift(left), np.nan)
    df["pivot_resistance"] = df["pivot_resistance"].ffill()
    df["pivot_support"] = df["pivot_support"].ffill()
    return df

def sr_strategy(df, lookback=100, left=10, right=10, tol=0.005):
    """
    Strategy based on support/resistance retests, reversals, and breakouts.
    """
    df = detect_sr_levels(df, lookback)
    df = detect_pivots(df)

    df["signal"] = "HOLD"
    df["confidence"] = 0.0

    for i in range(1, len(df)):
        price = df["Close"].iloc[i]
        support = df["support"].iloc[i]
        resistance = df["resistance"].iloc[i]
        pivot_s = df["pivot_support"].iloc[i]
        pivot_r = df["pivot_resistance"].iloc[i]

        # --- Long conditions ---
        if price <= support * 1.01 or price <= pivot_s * 1.01:
            df.loc[i, "signal"] = "BUY"
            df.loc[i, "confidence"] = 0.8

        # --- Short / Sell conditions ---
        elif price >= resistance * 0.99 or price >= pivot_r * 0.99:
            df.loc[i, "signal"] = "SELL"
            df.loc[i, "confidence"] = 0.8

        # --- Breakout Confirmation ---
        elif (price > resistance * 1.01):
            df.loc[i, "signal"] = "BUY_BREAKOUT"
            df.loc[i, "confidence"] = 0.9
        elif (price < support * 0.99):
            df.loc[i, "signal"] = "SELL_BREAKOUT"
            df.loc[i, "confidence"] = 0.9

        # --- Neutral Zone ---
        else:
            df.loc[i, "signal"] = "HOLD"
            df.loc[i, "confidence"] = 0.2

    return df
