# src/data/fetch_intraday_and_daily.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime

def fetch(symbol: str, interval: str, start=None, end=None):
    """
    Fetch OHLCV using yfinance. Note: intraday history is limited by yfinance.
    interval: '5m', '15m', '1d'
    """
    print(f"[INFO] Fetching {symbol} {interval} data...")
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, prepost=False, auto_adjust=False)
    df.reset_index(inplace=True)
    # unify column names
    df = df.rename(columns={"Datetime":"Date", "Date":"Date", "Adj Close":"Adj_Close"})
    # ensure timezone-naive/local consistent
    return df

def save_df(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved {out_path} (rows={len(df)})")

if __name__ == "__main__":
    symbol = "AAPL"
    # example: last 60 days intraday; yfinance limits apply
    df5 = fetch(symbol, "5m")
    save_df(df5, f"data/processed/{symbol}_5m.csv")

    df15 = fetch(symbol, "15m")
    save_df(df15, f"data/processed/{symbol}_15m.csv")

    df1d = fetch(symbol, "1d", start="2018-01-01", end=datetime.today().strftime("%Y-%m-%d"))
    save_df(df1d, f"data/processed/{symbol}_1d.csv")
