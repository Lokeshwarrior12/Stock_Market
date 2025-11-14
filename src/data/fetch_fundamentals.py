import yfinance as yf
import pandas as pd
import os
from src.utils.config import CONFIG

def fetch_fundamentals(symbol: str):
    stock = yf.Ticker(symbol)
    info = stock.info
    df = pd.DataFrame([info])
    return df

def save_fundamentals(df, path, symbol):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{symbol}_fundamentals.csv")
    df.to_csv(file_path, index=False)
    print(f"[SUCCESS] Fundamentals saved to {file_path}")

if __name__ == "__main__":
    symbol = CONFIG["stock_symbol"]
    print(f"[INFO] Fetching fundamentals for {symbol} from Yahoo Finance...")
    df = fetch_fundamentals(symbol)
    save_fundamentals(df, CONFIG["save_paths"]["raw_data"], symbol)
