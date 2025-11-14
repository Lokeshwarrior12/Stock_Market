# src/features/indicators.py
import pandas as pd
import numpy as np

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int):
    return series.rolling(window).mean()

def rsi(series: pd.Series, window: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, window=20, n_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + (std * n_std)
    lower = ma - (std * n_std)
    return upper, lower, ma

def atr(df: pd.DataFrame, window=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def vwap(df: pd.DataFrame):
    pv = (df["Close"] * df["Volume"])
    return pv.cumsum() / df["Volume"].cumsum()

def supertrend(df: pd.DataFrame, period=10, multiplier=3.0):
    """Basic SuperTrend (common on TradingView)."""
    hl2 = (df['High'] + df['Low']) / 2
    atr_vals = atr(df, window=period)
    upperband = hl2 + multiplier * atr_vals
    lowerband = hl2 - multiplier * atr_vals
    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    supertrend = pd.Series(index=df.index, dtype='float64')
    direction = pd.Series(index=df.index, dtype='int')
    for i in range(len(df)):
        if i == 0:
            final_upper.iat[i] = upperband.iat[i]
            final_lower.iat[i] = lowerband.iat[i]
            supertrend.iat[i] = np.nan
            direction.iat[i] = 1
        else:
            final_upper.iat[i] = upperband.iat[i] if upperband.iat[i] < final_upper.iat[i-1] or df['Close'].iat[i-1] > final_upper.iat[i-1] else final_upper.iat[i-1]
            final_lower.iat[i] = lowerband.iat[i] if lowerband.iat[i] > final_lower.iat[i-1] or df['Close'].iat[i-1] < final_lower.iat[i-1] else final_lower.iat[i-1]
            if df['Close'].iat[i] > final_upper.iat[i-1]:
                supertrend.iat[i] = final_lower.iat[i]
                direction.iat[i] = 1
            elif df['Close'].iat[i] < final_lower.iat[i-1]:
                supertrend.iat[i] = final_upper.iat[i]
                direction.iat[i] = -1
            else:
                supertrend.iat[i] = supertrend.iat[i-1]
                direction.iat[i] = direction.iat[i-1]
    return supertrend, direction
