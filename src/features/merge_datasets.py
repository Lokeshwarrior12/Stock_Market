import pandas as pd
import os
from src.utils.config import CONFIG


def merge_datasets(symbol: str):
    tech_path = os.path.join(CONFIG["save_paths"]["raw_data"], f"{symbol}_technical_data.csv")
    fund_path = os.path.join(CONFIG["save_paths"]["raw_data"], f"{symbol}_fundamentals.csv")

    df_tech = pd.read_csv(tech_path)
    df_fund = pd.read_csv(fund_path)

    print("[INFO] Technical dataset columns:", df_tech.columns.tolist())
    print("[INFO] Fundamental dataset columns:", df_fund.columns.tolist())

    # --- Clean Technical Data ---
    if "Date" not in df_tech.columns:
        raise KeyError("Expected 'Date' column in technical dataset.")

    df_tech["Date"] = pd.to_datetime(df_tech["Date"], errors="coerce")
    df_tech.dropna(subset=["Date"], inplace=True)
    df_tech = df_tech.sort_values("Date").reset_index(drop=True)

    # --- Prepare Fundamental Data ---
    if "date" in df_fund.columns:
        df_fund["date"] = pd.to_datetime(df_fund["date"], errors="coerce")
    elif "reportDate" in df_fund.columns:
        df_fund.rename(columns={"reportDate": "date"}, inplace=True)
        df_fund["date"] = pd.to_datetime(df_fund["date"], errors="coerce")
    else:
        print("[WARN] No date column found in fundamentals, assigning static date.")
        df_fund["date"] = df_tech["Date"].max()

    df_fund.dropna(subset=["date"], inplace=True)
    df_fund = df_fund.sort_values("date").reset_index(drop=True)

    # --- Merge (safe now) ---
    df_merged = pd.merge_asof(
        df_tech,
        df_fund,
        left_on="Date",
        right_on="date",
        direction="backward"
    )

    df_merged.drop(columns=["date"], inplace=True, errors="ignore")
    df_merged.fillna(method="ffill", inplace=True)

    # --- Add Target Column (Next-day Close Price) ---
    if "Close" not in df_merged.columns:
        raise KeyError("Expected 'Close' column in merged dataset to create Target.")

    df_merged["Target"] = df_merged["Close"].shift(-1)
    df_merged.dropna(subset=["Target"], inplace=True)

    # --- Save Final Dataset ---
    save_path = os.path.join(CONFIG["save_paths"]["processed_data"], f"{symbol}_merged_features.csv")
    os.makedirs(CONFIG["save_paths"]["processed_data"], exist_ok=True)
    df_merged.to_csv(save_path, index=False)
    print(f"[SUCCESS] Merged dataset with Target saved to {save_path}")

    return df_merged


if __name__ == "__main__":
    symbol = CONFIG["stock_symbol"]
    merge_datasets(symbol)
