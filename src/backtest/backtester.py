# src/backtest/backtester.py
import pandas as pd
from src.signals.signal_generator import enrich, generate_signals

def backtest_file(csv_path, atr_mult=1.5, rr=2.0):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = enrich(df)
    trades = generate_signals(df, atr_mult=atr_mult, rr=rr)
    if trades.empty:
        print("[WARN] No trades found")
    else:
        trades["duration_mins"] = (pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])).dt.total_seconds() / 60.0
        trades["return_pct"] = trades["pnl"] / trades["entry_price"] * 100.0
        trades.to_csv(csv_path.replace(".csv", f"_trades_atr{atr_mult}_rr{rr}.csv"), index=False)
        print(f"[INFO] Saved trades: {len(trades)} to CSV")
    return trades
