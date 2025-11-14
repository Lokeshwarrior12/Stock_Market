# src/data/fetch_price_data.py

import os
import yfinance as yf
import pandas as pd
import ta  # Technical analysis library
from src.utils.config import CONFIG


def fetch_stock_data(symbol: str, start: str, end: str, interval: str = "1d"):
    """Download historical data from Yahoo Finance"""
    df = yf.download(symbol, start=start, end=end, interval=interval)
    df.reset_index(inplace=True)
    return df


def add_technical_indicators(df: pd.DataFrame, params: dict):
    """Add base technical indicators using configurable parameters"""

    # --- Ensure columns are 1D Series ---
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(
        close, window=params["rsi_period"]
    ).rsi()

    # EMA Short and Long
    df["EMA_short"] = ta.trend.EMAIndicator(
        close, window=params["ema_short"]
    ).ema_indicator()
    df["EMA_long"] = ta.trend.EMAIndicator(
        close, window=params["ema_long"]
    ).ema_indicator()

    # MACD
    macd = ta.trend.MACD(
        close,
        window_slow=params["ema_long"],
        window_fast=params["ema_short"],
        window_sign=params["macd_signal"],
    )
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(
        close,
        window=params["bollinger_window"],
        window_dev=params["bollinger_std_dev"],
    )
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    # ATR (Volatility)
    df["ATR"] = ta.volatility.AverageTrueRange(
        high, low, close, window=params["atr_period"]
    ).average_true_range()

    return df


def save_data(df: pd.DataFrame, path: str, symbol: str):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{symbol}_technical_data.csv")
    df.to_csv(file_path, index=False)
    print(f"[INFO] Saved {symbol} data to {file_path}")


if __name__ == "__main__":
    symbol = CONFIG["stock_symbol"]
    data_cfg = CONFIG["data"]
    tech_cfg = CONFIG["technical_indicators"]

    print(f"[INFO] Fetching data for {symbol}...")
    df = fetch_stock_data(symbol, data_cfg["start_date"], data_cfg["end_date"], data_cfg["interval"])

    print(f"[INFO] Adding indicators...")
    df = add_technical_indicators(df, tech_cfg)

    save_data(df, CONFIG["save_paths"]["raw_data"], symbol)
    print("[SUCCESS] Data fetching and feature generation complete ✅")
