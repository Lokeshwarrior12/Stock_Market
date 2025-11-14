# src/signals/signal_generator.py
import numpy as np
import pandas as pd
# src/signals/signal_generator.py
import pandas as pd
import numpy as np
from src.features.indicators import ema, rsi, macd, bollinger, atr, supertrend, vwap
from src.signals.support_resistance_strategy import sr_strategy



# --- helpers ---
def _ensure_numeric(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    return df

def is_pin_bar(row, body_threshold=0.3, wick_ratio=2.5):
    """
    row: Series with Open, High, Low, Close
    body_threshold: max(body_size / candle_range)
    wick_ratio: min(long_wick / body_size to qualify as pin)
    returns (bool, confidence)
    """
    o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
    if pd.isna([o,h,l,c]).any():
        return False, 0.0
    body = abs(c - o)
    candle_range = h - l
    if candle_range <= 0: 
        return False, 0.0
    body_ratio = body / candle_range
    # determine upper and lower wick
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    # bullish pin: long lower wick; bearish: long upper wick
    if lower_wick > upper_wick and lower_wick / max(body, 1e-9) > wick_ratio and body_ratio < body_threshold:
        conf = min(1.0, (lower_wick / candle_range))  # > means strong
        return True, float(conf)
    if upper_wick > lower_wick and upper_wick / max(body, 1e-9) > wick_ratio and body_ratio < body_threshold:
        conf = min(1.0, (upper_wick / candle_range))
        return True, float(conf)
    return False, 0.0

def is_engulfing(prev, curr):
    """
    prev and curr: rows with Open, High, Low, Close
    Returns (bool, direction, confidence)
    direction: 'bull' or 'bear'
    """
    if pd.isna([prev["Open"], prev["Close"], curr["Open"], curr["Close"]]).any():
        return False, None, 0.0
    # curr body engulfs prev body
    prev_low, prev_high = min(prev["Open"], prev["Close"]), max(prev["Open"], prev["Close"])
    curr_low, curr_high = min(curr["Open"], curr["Close"]), max(curr["Open"], curr["Close"])
    if curr_low <= prev_low and curr_high >= prev_high and (curr_high - curr_low) > (prev_high - prev_low):
        direction = "bull" if curr["Close"] > curr["Open"] else "bear"
        conf = min(1.0, ( (curr_high - curr_low) / max(prev_high - prev_low, 1e-9) ) / 2.0)
        return True, direction, float(conf)
    return False, None, 0.0

def is_inside_bar(prev, curr):
    """
    Inside bar: curr high < prev high and curr low > prev low
    """
    if pd.isna([prev["High"], prev["Low"], curr["High"], curr["Low"]]).any():
        return False
    return (curr["High"] < prev["High"]) and (curr["Low"] > prev["Low"])

# Very small Doji detect
def is_doji(row, tol=0.02):
    o, c, h, l = row["Open"], row["Close"], row["High"], row["Low"]
    if pd.isna([o,c,h,l]).any(): return False
    body = abs(c - o)
    rng = h - l
    if rng == 0: return False
    return (body / rng) < tol

# --- volume confirmation helper ---
def volume_confirmation(df, idx, lookback=20, multiplier=1.0):
    """
    Return True if volume at idx is >= multiplier * avg(volume of lookback)
    """
    if "Volume" not in df.columns:
        return False, 0.0
    start = max(0, idx - lookback)
    avg_vol = df["Volume"].iloc[start:idx].replace(0, np.nan).mean()
    v = df["Volume"].iloc[idx]
    if pd.isna(avg_vol) or avg_vol == 0:
        return False, 0.0
    conf = float(min(1.0, v / (avg_vol * multiplier)))
    return (v >= avg_vol * multiplier), conf

def enrich(df):
    """Add core technical indicators (EMA, RSI, MACD, ATR, VWAP, SuperTrend) to df."""
    df = df.copy()
    # make sure price columns are numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["EMA_short"] = ema(df["Close"], span=9)
    df["EMA_long"] = ema(df["Close"], span=21)
    df["RSI"] = rsi(df["Close"], window=14)
    macd_val, macd_sig, macd_hist = macd(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd_val, macd_sig, macd_hist
    df["ATR"] = atr(df, window=14)
    df["VWAP"] = vwap(df)
    st, st_dir = supertrend(df, period=10, multiplier=3)
    df["ST"], df["ST_dir"] = st, st_dir
    return df

# --- higher-level generator ---

def generate_signals(df, atr_mult=1.5, rr=2.0, min_vol_mult=1.0, require_confirm=True):

    """
    Scans df for patterns and returns trades with entry/exit/stop/target/confidence/trade_type.
    - df must include Date, Open, High, Low, Close, Volume, ATR (or calculates ATR outside)
    - intraday vs swing vs investment classification:
        - intraday: exit within same day (we mark if exit_idx has same date)
        - swing: hold > 1 day <= 14 days
        - investment: > 365 days
    """
    df = df.copy().reset_index(drop=True)
    df = _ensure_numeric(df, ["Open", "High", "Low", "Close", "Volume", "ATR"])
    trades = []
    in_trade = False
    trade = None

    for i in range(2, len(df)):
        row = df.loc[i]
        prev = df.loc[i-1]
        prev2 = df.loc[i-2]

        # detect patterns
        pin, pin_conf = is_pin_bar(row)
        engulf, eng_dir, eng_conf = is_engulfing(prev, row)
        inside = is_inside_bar(prev, row)
        doji = is_doji(row)

        # volume confirmation
        vol_ok, vol_conf = volume_confirmation(df, i, lookback=20, multiplier=min_vol_mult)

        # MACD & EMA checks if available (confluence)
        macd_ok = False
        ema_ok = False
        if "MACD" in df.columns and "MACD_signal" in df.columns:
            macd_ok = row["MACD"] > row["MACD_signal"]
        if "EMA_short" in df.columns and "EMA_long" in df.columns:
            ema_ok = row["EMA_short"] > row["EMA_long"]

        # combined confidence
        pattern_conf = max(pin_conf, eng_conf, 1.0 if inside else 0.0, 0.5 if doji else 0.0)
        conf = 0.4 * pattern_conf + 0.3 * float(vol_conf) + 0.15 * float(macd_ok) + 0.15 * float(ema_ok)

        # ENTRY rules (long example). Mirror for short when pattern is bearish.
        entry_long = ((pin and row["Close"] > row["Open"] ) or (engulf and eng_dir=="bull") or (inside and macd_ok)) and (not in_trade)
        entry_short = ((pin and row["Close"] < row["Open"] ) or (engulf and eng_dir=="bear") or (inside and not macd_ok)) and (not in_trade)

        # optionally require volume confirmation
        if require_confirm:
            entry_long = entry_long and vol_ok
            entry_short = entry_short and vol_ok

        if not in_trade and (entry_long or entry_short):
            entry_price = float(row["Close"])
            atr = float(row["ATR"]) if not np.isnan(row["ATR"]) and row["ATR"]>0 else (df["High"] - df["Low"]).rolling(10).mean().iloc[i]
            stop = entry_price - atr_mult * atr if entry_long else entry_price + atr_mult * atr
            # target: R:R
            if entry_long:
                target = entry_price + rr * (entry_price - stop)
                trade_type = "LONG"
            else:
                target = entry_price - rr * (stop - entry_price)
                trade_type = "SHORT"
            trade = dict(
                entry_idx = i,
                entry_time = row.get("Date"),
                entry_price = entry_price,
                stop = float(stop),
                target = float(target),
                exit_idx = None,
                exit_time = None,
                exit_price = None,
                pnl = None,
                exit_reason = None,
                confidence = float(conf),
                pattern = "pin" if pin else ("engulf" if engulf else ("inside" if inside else ("doji" if doji else "unknown"))),
                trade_type = trade_type
            )
            in_trade = True
            continue

        # trade management
        if in_trade:
            high, low, close = row["High"], row["Low"], row["Close"]
            # check stop/target
            if trade["trade_type"] == "LONG":
                if low <= trade["stop"]:
                    exit_price = trade["stop"]; reason="stop"
                elif high >= trade["target"]:
                    exit_price = trade["target"]; reason="target"
                elif ("EMA_short" in df.columns and df["EMA_short"].iloc[i] < df["EMA_long"].iloc[i]) or ("ST_dir" in df.columns and df["ST_dir"].iloc[i] != 1):
                    exit_price = float(close); reason="signal_exit"
                else:
                    exit_price = None; reason=None
            else:
                # SHORT
                if high >= trade["stop"]:
                    exit_price = trade["stop"]; reason="stop"
                elif low <= trade["target"]:
                    exit_price = trade["target"]; reason="target"
                elif ("EMA_short" in df.columns and df["EMA_short"].iloc[i] > df["EMA_long"].iloc[i]) or ("ST_dir" in df.columns and df["ST_dir"].iloc[i] != -1):
                    exit_price = float(close); reason="signal_exit"
                else:
                    exit_price = None; reason=None

            if exit_price is not None:
                trade["exit_idx"] = i
                trade["exit_time"] = row.get("Date")
                trade["exit_price"] = float(exit_price)
                trade["pnl"] = float(exit_price - trade["entry_price"]) if trade["trade_type"]=="LONG" else float(trade["entry_price"] - exit_price)
                trade["exit_reason"] = reason
                # classify tenure
                try:
                    t0 = pd.to_datetime(trade["entry_time"])
                    t1 = pd.to_datetime(trade["exit_time"])
                    days = (t1 - t0).days
                except Exception:
                    days = None
                if days is None:
                    trade["tenure"] = "unknown"
                elif days == 0:
                    trade["tenure"] = "intraday"
                elif 1 <= days <= 14:
                    trade["tenure"] = "swing"
                elif days > 365:
                    trade["tenure"] = "investment"
                else:
                    trade["tenure"] = "position"
                trades.append(trade)
                in_trade = False
                trade = None

    return pd.DataFrame(trades)
# --------------------------------------------------------------------
# --- v2: Rule-based Indicator Confluence Strategy (SMA + RSI + MACD)
# --------------------------------------------------------------------

def generate_signals_v2(
    df,
    atr_mult=1.5,
    rr=2.0,
    sma_short=20,
    sma_long=50,
    rsi_buy=30,
    rsi_sell=70,
    min_vol_mult=1.0,
    require_confirm=True
):
    """
    Advanced rule-based signal generator using trend, momentum, and volume confirmation.
    Does NOT interfere with candle-pattern generator above.

    Rules:
    - Trend: SMA(20) > SMA(50) and EMA_short > EMA_long
    - Momentum: RSI between rsi_buy and rsi_sell
    - Confirmation: MACD > MACD_signal and Volume > avg*min_vol_mult
    - Exit on stop, target, or trend reversal
    """

    df = df.copy()
    df["SMA_short"] = df["Close"].rolling(window=sma_short).mean()
    df["SMA_long"] = df["Close"].rolling(window=sma_long).mean()

    trades = []
    in_trade = False
    trade = None

    for i in range(max(sma_long, 14), len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # --- Volume confirmation
        vol_ok, vol_conf = volume_confirmation(df, i, lookback=20, multiplier=min_vol_mult)

        # --- Trend & momentum
        trend_up = (row["SMA_short"] > row["SMA_long"]) and (row["EMA_short"] > row["EMA_long"])
        macd_up = row.get("MACD", 0) > row.get("MACD_signal", 0)
        rsi_ok = (rsi_buy < row.get("RSI", 50) < rsi_sell)

        # --- Entry condition
        entry_long = trend_up and macd_up and rsi_ok and vol_ok and not in_trade
        if require_confirm:
            entry_long = entry_long and (row["Close"] > row["EMA_short"])

        if entry_long:
            entry_price = float(row["Close"])
            atr_val = float(row.get("ATR", np.nan))
            if np.isnan(atr_val):
                atr_val = (df["High"] - df["Low"]).rolling(14).mean().iloc[i]
            stop = entry_price - atr_mult * atr_val
            target = entry_price + rr * (entry_price - stop)
            trade = dict(
                entry_idx=i,
                entry_time=row.get("Date"),
                entry_price=entry_price,
                stop=stop,
                target=target,
                exit_idx=None,
                exit_time=None,
                exit_price=None,
                pnl=None,
                exit_reason=None,
                confidence=float(0.6 + 0.4 * vol_conf),
                pattern="IndicatorConfluence",
                trade_type="LONG"
            )
            in_trade = True
            continue

        # --- Manage trade
        if in_trade and trade:
            high, low, close = row["High"], row["Low"], row["Close"]

            if low <= trade["stop"]:
                exit_price = trade["stop"]; reason = "stop"
            elif high >= trade["target"]:
                exit_price = trade["target"]; reason = "target"
            elif (row["SMA_short"] < row["SMA_long"]) or (row.get("RSI", 50) > rsi_sell):
                exit_price = close; reason = "trend_reversal"
            else:
                exit_price = None; reason = None

            if exit_price is not None:
                trade["exit_idx"] = i
                trade["exit_time"] = row.get("Date")
                trade["exit_price"] = float(exit_price)
                trade["pnl"] = float(exit_price - trade["entry_price"])
                trade["exit_reason"] = reason
                # classify tenure
                try:
                    t0 = pd.to_datetime(trade["entry_time"])
                    t1 = pd.to_datetime(trade["exit_time"])
                    days = (t1 - t0).days
                except Exception:
                    days = None
                if days is None:
                    trade["tenure"] = "unknown"
                elif days == 0:
                    trade["tenure"] = "intraday"
                elif 1 <= days <= 14:
                    trade["tenure"] = "swing"
                elif days > 365:
                    trade["tenure"] = "investment"
                else:
                    trade["tenure"] = "position"
                trades.append(trade)
                in_trade = False
                trade = None

    return pd.DataFrame(trades)
# --------------------------------------------------------------------
# --- v3: Hybrid Indicator + Candlestick Confirmation Strategy
# --------------------------------------------------------------------

def generate_signals_v3(
    df,
    atr_mult=1.5,
    rr=2.0,
    sma_short=20,
    sma_long=50,
    rsi_buy=30,
    rsi_sell=70,
    min_vol_mult=1.0,
    require_confirm=True
):
    """
    Hybrid signal generator combining:
    - Trend indicators (SMA/EMA/MACD)
    - Momentum (RSI)
    - Volatility (ATR)
    - Volume confirmation
    - Candlestick confirmation (pin bar, engulfing, doji)
    
    Rules:
    BUY when:
      - SMA(20) > SMA(50)
      - EMA_short > EMA_long
      - MACD > MACD_signal
      - RSI between rsi_buy and rsi_sell
      - Bullish candle pattern detected (Pin Bar / Engulfing / Doji)
      - Volume > average * min_vol_mult

    SELL/EXIT when:
      - Opposite trend conditions
      - RSI > rsi_sell or < rsi_buy
      - Target/Stop hit
    """

    df = df.copy().reset_index(drop=True)
    df["SMA_short"] = df["Close"].rolling(window=sma_short).mean()
    df["SMA_long"] = df["Close"].rolling(window=sma_long).mean()

    trades = []
    in_trade = False
    trade = None

    for i in range(max(sma_long, 14), len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        prev2 = df.iloc[i - 2] if i >= 2 else prev

        # --- Volume confirmation
        vol_ok, vol_conf = volume_confirmation(df, i, lookback=20, multiplier=min_vol_mult)

        # --- Candlestick patterns
        pin, pin_conf = is_pin_bar(row)
        engulf, eng_dir, eng_conf = is_engulfing(prev, row)
        doji = is_doji(row)

        # Combine pattern strength
        pattern_conf = max(pin_conf, eng_conf, 0.5 if doji else 0.0)
        has_bullish_pattern = (pin and row["Close"] > row["Open"]) or (engulf and eng_dir == "bull") or (doji and row["Close"] > row["Open"])
        has_bearish_pattern = (pin and row["Close"] < row["Open"]) or (engulf and eng_dir == "bear") or (doji and row["Close"] < row["Open"])

        # --- Trend & momentum confirmation
        trend_up = (row["SMA_short"] > row["SMA_long"]) and (row["EMA_short"] > row["EMA_long"])
        trend_down = (row["SMA_short"] < row["SMA_long"]) and (row["EMA_short"] < row["EMA_long"])
        macd_up = row.get("MACD", 0) > row.get("MACD_signal", 0)
        macd_down = row.get("MACD", 0) < row.get("MACD_signal", 0)
        rsi_ok = (rsi_buy < row.get("RSI", 50) < rsi_sell)

        # --- Entry conditions
        entry_long = trend_up and macd_up and rsi_ok and vol_ok and has_bullish_pattern and not in_trade
        entry_short = trend_down and macd_down and rsi_ok and vol_ok and has_bearish_pattern and not in_trade

        # Add confirmation (price above short EMA)
        if require_confirm:
            entry_long = entry_long and (row["Close"] > row["EMA_short"])
            entry_short = entry_short and (row["Close"] < row["EMA_short"])

        # --- Long entry
        if entry_long:
            entry_price = float(row["Close"])
            atr_val = float(row.get("ATR", np.nan))
            if np.isnan(atr_val):
                atr_val = (df["High"] - df["Low"]).rolling(14).mean().iloc[i]
            stop = entry_price - atr_mult * atr_val
            target = entry_price + rr * (entry_price - stop)
            trade = dict(
                entry_idx=i,
                entry_time=row.get("Date"),
                entry_price=entry_price,
                stop=float(stop),
                target=float(target),
                exit_idx=None,
                exit_time=None,
                exit_price=None,
                pnl=None,
                exit_reason=None,
                confidence=float(0.5 * pattern_conf + 0.3 * vol_conf + 0.2),
                pattern="bullish_pattern",
                trade_type="LONG"
            )
            in_trade = True
            continue

        # --- Short entry
        if entry_short:
            entry_price = float(row["Close"])
            atr_val = float(row.get("ATR", np.nan))
            if np.isnan(atr_val):
                atr_val = (df["High"] - df["Low"]).rolling(14).mean().iloc[i]
            stop = entry_price + atr_mult * atr_val
            target = entry_price - rr * (stop - entry_price)
            trade = dict(
                entry_idx=i,
                entry_time=row.get("Date"),
                entry_price=entry_price,
                stop=float(stop),
                target=float(target),
                exit_idx=None,
                exit_time=None,
                exit_price=None,
                pnl=None,
                exit_reason=None,
                confidence=float(0.5 * pattern_conf + 0.3 * vol_conf + 0.2),
                pattern="bearish_pattern",
                trade_type="SHORT"
            )
            in_trade = True
            continue

        # --- Manage trade
        if in_trade and trade:
            high, low, close = row["High"], row["Low"], row["Close"]

            if trade["trade_type"] == "LONG":
                if low <= trade["stop"]:
                    exit_price = trade["stop"]; reason = "stop"
                elif high >= trade["target"]:
                    exit_price = trade["target"]; reason = "target"
                elif trend_down or (row.get("RSI", 50) > rsi_sell):
                    exit_price = close; reason = "trend_reversal"
                else:
                    exit_price = None; reason = None
            else:
                if high >= trade["stop"]:
                    exit_price = trade["stop"]; reason = "stop"
                elif low <= trade["target"]:
                    exit_price = trade["target"]; reason = "target"
                elif trend_up or (row.get("RSI", 50) < rsi_buy):
                    exit_price = close; reason = "trend_reversal"
                else:
                    exit_price = None; reason = None

            if exit_price is not None:
                trade["exit_idx"] = i
                trade["exit_time"] = row.get("Date")
                trade["exit_price"] = float(exit_price)
                trade["pnl"] = (
                    float(exit_price - trade["entry_price"])
                    if trade["trade_type"] == "LONG"
                    else float(trade["entry_price"] - exit_price)
                )
                trade["exit_reason"] = reason
                try:
                    t0 = pd.to_datetime(trade["entry_time"])
                    t1 = pd.to_datetime(trade["exit_time"])
                    days = (t1 - t0).days
                except Exception:
                    days = None
                if days is None:
                    trade["tenure"] = "unknown"
                elif days == 0:
                    trade["tenure"] = "intraday"
                elif 1 <= days <= 14:
                    trade["tenure"] = "swing"
                elif days > 365:
                    trade["tenure"] = "investment"
                else:
                    trade["tenure"] = "position"
                trades.append(trade)
                in_trade = False
                trade = None

    return pd.DataFrame(trades)

# add to src/signals/signal_generator.py (end of file)

def find_future_signals(df_hist, df_future, lookback_window=60, require_confirm=True, **kwargs):
    """
    Append df_future to history, enrich indicators, and return only the signals with entry_time after last hist date.
    df_hist: historical DataFrame (sorted by Date)
    df_future: synthetic future bars DataFrame (Date, Open, High, Low, Close, Volume)
    Returns: trades_future_df (DataFrame)
    """
    df_combined = pd.concat([df_hist, df_future], ignore_index=True).sort_values("Date").reset_index(drop=True)
    df_combined = enrich(df_combined)  # recompute indicators (enrich is your function)
    trades = generate_signals(df_combined, require_confirm=require_confirm, **kwargs)
    # filter trades whose entry_time > last historical date
    last_hist_date = pd.to_datetime(df_hist["Date"].iloc[-1])
    if trades.empty:
        return trades
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    future_trades = trades[trades["entry_time"] > last_hist_date].copy()
    return future_trades.reset_index(drop=True)

# Add SR-based strategy signals
# --- SR Integration Helper ---
def merge_sr_strategy(df):
    """Integrate Support & Resistance strategy with enriched dataset."""
    df_enriched = enrich(df)
    df_sr = sr_strategy(df_enriched, lookback=100)

    df_final = df_enriched.copy()
    df_final["SR_signal"] = df_sr["signal"]
    df_final["SR_confidence"] = df_sr["confidence"]

    # Reinforce trades when SR and pattern agree
    df_final["final_signal"] = np.where(
        (df_final["SR_signal"] == "BUY") & (df_final["EMA_short"] > df_final["EMA_long"]),
        "STRONG_BUY",
        np.where(
            (df_final["SR_signal"] == "SELL") & (df_final["EMA_short"] < df_final["EMA_long"]),
            "STRONG_SELL",
            df_final["SR_signal"]
        )
    )

    return df_final
