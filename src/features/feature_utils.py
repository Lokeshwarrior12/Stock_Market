# src/features/feature_utils.py

import numpy as np
import pandas as pd


# --------------------------------------------
# 📊 1. Basic Technical Indicators
# --------------------------------------------

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_ema(data: pd.Series, span: int) -> pd.Series:
    """Compute Exponential Moving Average (EMA)."""
    return data.ewm(span=span, adjust=False).mean()


def calculate_macd(data: pd.Series, short_span=12, long_span=26, signal_span=9):
    """Compute MACD (Moving Average Convergence Divergence)."""
    ema_short = calculate_ema(data, short_span)
    ema_long = calculate_ema(data, long_span)
    macd_line = ema_short - ema_long
    signal_line = calculate_ema(macd_line, signal_span)
    return macd_line, signal_line


def calculate_atr(df: pd.DataFrame, period: int = 14):
    """Average True Range (volatility indicator)."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()


# --------------------------------------------
# 🧠 2. Support & Resistance Detection
# --------------------------------------------

def detect_support_resistance(df: pd.DataFrame, window: int = 20, tolerance: float = 0.02):
    """
    Detect support and resistance levels in price data.
    """
    supports, resistances = [], []
    for i in range(window, len(df) - window):
        low_range = df['Low'][i - window:i + window]
        high_range = df['High'][i - window:i + window]
        if df['Low'][i] == low_range.min():
            supports.append((i, df['Low'][i]))
        if df['High'][i] == high_range.max():
            resistances.append((i, df['High'][i]))

    df['Support'] = np.nan
    df['Resistance'] = np.nan
    for i, level in supports:
        df.loc[i, 'Support'] = level
    for i, level in resistances:
        df.loc[i, 'Resistance'] = level

    df['Support'] = df['Support'].ffill()
    df['Resistance'] = df['Resistance'].ffill()
    return df


# --------------------------------------------
# ⚙️ 3. Feature Scaling & Smoothing
# --------------------------------------------

def normalize_series(series: pd.Series) -> pd.Series:
    """Min-Max normalize a pandas Series."""
    return (series - series.min()) / (series.max() - series.min())


def smooth_series(series: pd.Series, window: int = 5) -> pd.Series:
    """Apply a rolling mean to smooth the data."""
    return series.rolling(window=window, min_periods=1).mean()


# --------------------------------------------
# 📈 4. Correlation & Feature Importance
# --------------------------------------------

def correlation_heatmap(df: pd.DataFrame, target_col: str = "Target"):
    """Return sorted correlation values with respect to the target."""
    corr = df.corr()[target_col].sort_values(ascending=False)
    return corr


def feature_selection_by_correlation(df: pd.DataFrame, target_col: str, threshold: float = 0.1):
    """
    Select features highly correlated with the target variable.
    """
    corr = df.corr()[target_col].abs()
    selected = corr[corr > threshold].index.tolist()
    selected.remove(target_col)
    return selected


# --------------------------------------------
# 🧰 5. Combined Feature Engineering Function
# --------------------------------------------

def add_technical_features(df: pd.DataFrame):
    """Compute a standard set of technical indicators."""
    df['RSI'] = calculate_rsi(df['Close'])
    df['EMA_short'] = calculate_ema(df['Close'], 12)
    df['EMA_long'] = calculate_ema(df['Close'], 26)
    macd, signal = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_signal'] = signal
    df['ATR'] = calculate_atr(df)
    df = detect_support_resistance(df)
    return df


# --------------------------------------------
# 🧩 6. Data Health Checks
# --------------------------------------------

def check_missing_values(df: pd.DataFrame):
    """Print missing value report."""
    missing = df.isnull().sum()
    print("[INFO] Missing Values:")
    print(missing[missing > 0])
    return missing


def drop_outliers(df: pd.DataFrame, cols=None, z_thresh=3):
    """Drop outliers using Z-score filtering."""
    from scipy.stats import zscore
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns
    z_scores = np.abs(zscore(df[cols]))
    return df[(z_scores < z_thresh).all(axis=1)]
