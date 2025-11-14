# app.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from src.model.train_hybrid_model import HybridModel
from src.genai.market_explainer import generate_prompt, generate_analysis
from src.utils.config import CONFIG
from collections.abc import Iterable
from src.signals.signal_generator import enrich, generate_signals, generate_signals_v2, generate_signals_v3, find_future_signals
from src.model.forecast import iterative_forecast
from src.signals.signal_generator import find_future_signals
from src.features.support_resistance import detect_trend_reversals, plot_support_resistance
from src.model.train_hybrid_model import iterative_forecast
from src.model.forecast import iterative_forecast




# UI: choose forecast horizon
horizon_type = st.selectbox("Forecast horizon", ["Next day", "Next 7 days", "Next 30 days", "Next 90 days"])
horizon_map = {"Next day": 1, "Next 7 days": 7, "Next 30 days": 30, "Next 90 days": 90}
horizon = horizon_map[horizon_type]


# -----------------------
# Load Model and Data
# -----------------------

@st.cache_resource
def load_model(model_path, input_dim_seq, input_dim_fund):
    cfg = CONFIG["hybrid_model"]
    model = HybridModel(input_dim_seq, input_dim_fund,
                        hidden_dim=cfg["hidden_dim"],
                        num_layers=cfg["num_layers"],
                        dropout=cfg["dropout"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


@st.cache_data
def load_data(symbol):
    df = pd.read_csv(f"data/processed/{symbol}_merged_features.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    # Only drop rows where *critical* columns (like price) are missing
    critical_cols = ["Date", "Close", "Open", "High", "Low", "Volume"]
    df.dropna(subset=critical_cols, inplace=True)
    # ensure Date parsed & sorted
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

# enrich indicators (if not already saved)
    df_enriched = enrich(df)
    trades_df = generate_signals(df_enriched, atr_mult=1.5, rr=2.0, min_vol_mult=1.0, require_confirm=True)
    # Use the new version (v2) for rule-based indicator strategy
    trades_df = generate_signals_v2(df_enriched, atr_mult=1.5, rr=2.0, min_vol_mult=1.2)
    trades_df = generate_signals_v3(df_enriched, atr_mult=1.5, rr=2.0, min_vol_mult=1.2)

    st.subheader("Generated Signals (latest)")
    st.dataframe(trades_df.sort_values("entry_time", ascending=False).head(10))


    # For all other columns (especially fundamentals), fill missing with 0 or last value
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    return df


# -----------------------
# Plotting Utilities
# -----------------------

def plot_predictions(df, preds, trades_df=None, lookback_days=180):
    df_recent = df.tail(lookback_days)
    fig = go.Figure(data=[
        go.Candlestick(
            x=df_recent["Date"],
            open=df_recent["Open"],
            high=df_recent["High"],
            low=df_recent["Low"],
            close=df_recent["Close"],
            name="Candles"
        )
    ])

    fig.add_trace(go.Scatter(
        x=df_recent["Date"].iloc[-len(preds):],
        y=preds,
        mode="lines",
        name="Predicted Price",
        line=dict(color="orange")
    ))

    if trades_df is not None and not trades_df.empty:
        # Mark entries and exits
        fig.add_trace(go.Scatter(
            x=trades_df["entry_time"], y=trades_df["entry_price"],
            mode="markers", name="Entry", marker=dict(color="green", size=10, symbol="triangle-up")
        ))
        fig.add_trace(go.Scatter(
            x=trades_df["exit_time"], y=trades_df["exit_price"],
            mode="markers", name="Exit", marker=dict(color="red", size=10, symbol="triangle-down")
        ))

    fig.update_layout(template="plotly_dark", height=600, title="Predicted vs Actual Price + Signals")
    st.plotly_chart(fig, use_container_width=True)
    
def compose_trade_call(df_enriched, pred_price, current_price, trades_df):
    # find latest high-confidence pattern within last N bars
    recent = trades_df.sort_values("entry_time", ascending=False).iloc[:5]
    strong = recent[recent["confidence"] >= 0.55]
    suggestion = {}
    suggestion["predicted_price"] = float(pred_price)
    suggestion["current_price"] = float(current_price)
    suggestion["delta_pct"] = 100 * (pred_price - current_price) / max(1e-9, current_price)
    if not strong.empty:
        t = strong.iloc[0]
        suggestion["signal"] = t["trade_type"]
        suggestion["entry_time"] = t["entry_time"]
        suggestion["entry_price"] = t["entry_price"]
        suggestion["stop"] = t["stop"]
        suggestion["target"] = t["target"]
        suggestion["confidence"] = float(t["confidence"])
        suggestion["note"] = f"Pattern: {t['pattern']}, tenure: {t.get('tenure')}"
    else:
        suggestion["signal"] = "no_confluence"
        suggestion["confidence"] = 0.25
        suggestion["note"] = "ML prediction only — no high-confidence price action pattern"
    return suggestion


import plotly.graph_objects as go

def plot_candles_with_signals(df, trades=None, chart_days=200):
    """Plot candlestick chart with trade entries and SR Buy/Sell signals."""
    df_plot = df.tail(chart_days)

    fig = go.Figure(data=[go.Candlestick(
        x=df_plot['Date'],
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close'],
        name='Candles'
    )])

    # --- SR-based arrows ---
    if "final_signal" in df_plot.columns:
        buy_signals = df_plot[df_plot["final_signal"].isin(["BUY", "STRONG_BUY"])]
        sell_signals = df_plot[df_plot["final_signal"].isin(["SELL", "STRONG_SELL"])]

        fig.add_trace(go.Scatter(
            x=buy_signals["Date"],
            y=buy_signals["Low"] * 0.995,  # slightly below candle
            mode="markers+text",
            marker_symbol="triangle-up",
            marker=dict(size=12, color="lime"),
            text=["BUY" if s == "BUY" else "STRONG BUY" for s in buy_signals["final_signal"]],
            textposition="bottom center",
            name="BUY Signal"
        ))

        fig.add_trace(go.Scatter(
            x=sell_signals["Date"],
            y=sell_signals["High"] * 1.005,  # slightly above candle
            mode="markers+text",
            marker_symbol="triangle-down",
            marker=dict(size=12, color="red"),
            text=["SELL" if s == "SELL" else "STRONG SELL" for s in sell_signals["final_signal"]],
            textposition="top center",
            name="SELL Signal"
        ))

    # --- Trade markers (optional) ---
    if trades is not None and not trades.empty:
        fig.add_trace(go.Scatter(
            x=trades["entry_time"],
            y=trades["entry_price"],
            mode="markers",
            marker_symbol="triangle-up",
            marker=dict(size=10, color="cyan"),
            name="Trade Entry"
        ))
        fig.add_trace(go.Scatter(
            x=trades["exit_time"],
            y=trades["exit_price"],
            mode="markers",
            marker_symbol="x",
            marker=dict(size=10, color="white"),
            name="Trade Exit"
        ))

    fig.update_layout(template='plotly_dark', height=750, title="Price + SR & Pattern Signals")
    st.plotly_chart(fig, use_container_width=True)

from src.model.train_lstm_model import LSTMModel
import torch, json, os

def load_lstm_model():
    model_path = CONFIG["training"]["model_save_path"]
    meta_path = os.path.splitext(model_path)[0] + "_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    input_dim = meta["input_dim"]

    model = LSTMModel(input_dim, hidden_dim=CONFIG["training"]["hidden_dim"], num_layers=CONFIG["training"]["num_layers"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"[INFO] LSTM model loaded (input_dim={input_dim}) ✅")
    return model

def load_model(model_path, input_dim_seq, input_dim_fund):
    model = HybridModel(
        input_dim_seq=input_dim_seq,
        input_dim_fund=input_dim_fund,
        hidden_dim=CONFIG["hybrid_model"]["hidden_dim"],
        num_layers=CONFIG["hybrid_model"]["num_layers"],
        dropout=CONFIG["hybrid_model"]["dropout"]
    )

    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("[INFO] Loaded pretrained HybridModel successfully.")
    except RuntimeError as e:
        print(f"[WARN] Model weight mismatch detected. Skipping load. ({e})")

    model.eval()
    return model



# -----------------------
# Streamlit App Layout
# -----------------------

def main():
    st.set_page_config(page_title="AI Stock Predictor", layout="wide")
    st.title(" AI-Powered Stock Prediction Dashboard")
    symbol = CONFIG["stock_symbol"]

    # Sidebar
    st.sidebar.header(" Settings")
    forecast_days = st.sidebar.slider("Forecast Horizon (days)", 1, 20, 5)
    lookback_window = st.sidebar.slider("Lookback Window", 30, 120, 60)

    st.sidebar.markdown("---")
    st.sidebar.write("Using Model:")
    st.sidebar.code("HybridModel (Technical + Fundamentals)")
    

    # Load data
    df = load_data(symbol)
    st.subheader(f"Stock: {symbol}")
    st.write(" Data shape after loading:", df.shape)
    st.write(" Columns:", df.columns.tolist())

    if df.empty:
        st.error("Loaded dataset is empty. Please check your CSV file or preprocessing pipeline.")
        st.stop()

    # Prepare features
    tech_cols = ["Open", "High", "Low", "Close", "Volume", "RSI", 
             "EMA_short", "EMA_long", "MACD", "MACD_signal", 
             "BB_high", "ATR"]
    fund_cols = [c for c in df.columns if c not in tech_cols + ["Date", "Target"] and df[c].dtype != 'O']
    st.write("Number of technical features in dataset:", len(tech_cols))
    st.write("Number of fundamental features:", len(fund_cols))


    input_dim_seq = len(tech_cols)
    input_dim_fund = len(fund_cols)

    model = load_model(CONFIG["hybrid_model"]["model_save_path"], input_dim_seq, input_dim_fund)

    df_scaled = df.copy().fillna(method="ffill").fillna(0)
    if df_scaled.empty:
        st.error("No valid rows available after cleaning data.")
        st.stop()

    if len(df_scaled) == 0:
        st.error("No data available for prediction — check input CSV.")
        st.stop()

    if len(fund_cols) > 0:
        X_fund = torch.tensor(
            df_scaled[fund_cols].iloc[-1].values,
            dtype=torch.float32
        ).unsqueeze(0)
    else:
        st.warning("No fundamental features found — using zeros.")
        X_fund = torch.zeros((1, input_dim_fund))


    # Prepare tensors
    # --- Prepare model inputs ---
    # Make sure there are enough rows for the lookback window
    if len(df) < lookback_window:
        st.warning(f"Not enough data ({len(df)} rows) for a {lookback_window}-day lookback.")
        st.stop()

    # Use numeric columns only, fill NaNs defensively
    df_scaled = df.copy().fillna(method="ffill").fillna(0)

    # Prepare the technical sequence (last `lookback_window` days)
    X_seq = torch.tensor(
        df_scaled[tech_cols].tail(lookback_window).values,
        dtype=torch.float32
    ).unsqueeze(0)   # shape: (1, lookback_window, n_tech_features)

    # Prepare the fundamental snapshot (latest fundamentals)
    if len(fund_cols) > 0:
        X_fund = torch.tensor(
            df_scaled[fund_cols].iloc[-1].values,
            dtype=torch.float32
        ).unsqueeze(0)
    else:
    # if no fundamental features, just use zeros
        X_fund = torch.zeros((1, input_dim_fund))
    # shape: (1, n_fund_features)

    # --- Run prediction ---
    with torch.no_grad():
        pred_price = model(X_seq, X_fund).item()

    # --- Get latest actual close price ---
    current_price = float(df["Close"].iloc[-1])

    # Predict
    with torch.no_grad():
        pred_price = model(X_seq, X_fund).item()
    current_price = df["Close"].iloc[-1]

    delta = pred_price - current_price
    direction = " Up" if delta > 0 else " Down"
    
    delta_color = "normal" if delta > 0 else "inverse"
    st.metric(label="Predicted Movement", value=f"${pred_price:.2f}", delta=f"{delta:.2f}", delta_color=delta_color)

    st.markdown(f"**Prediction Direction:** {direction}")
    from src.signals.signal_generator import enrich, generate_signals
    
    # Required variables for forecasting
    lookback_window = st.sidebar.slider("Forecast lookback (bars)", 30, 180, 60)
    tech_cols = ["Open", "High", "Low", "Close", "Volume", "RSI", "EMA_short", "EMA_long", "MACD", "MACD_signal", "BB_high", "ATR"]
    fund_cols = [c for c in df.columns if c not in tech_cols + ["Date", "Target"] and df[c].dtype != 'O']

    df = pd.read_csv("data/processed/AAPL_merged_features.csv")
    df_future, preds = iterative_forecast(
        model,
        df,
        tech_cols=tech_cols,
        fund_cols=fund_cols,
        lookback_window=60,
        horizon=5
    )
    
    # Forecast

    st.info(f"Running iterative forecast for {forecast_days} steps...")
    future_df, future_preds = iterative_forecast(
        model=model,
        df=df,
        tech_cols=tech_cols,
        fund_cols=fund_cols,
        lookback_window=lookback_window,
        horizon=forecast_days
    )

    if len(future_preds) > 0:
        st.success(f"✅ Predicted next {forecast_days} close(s): {', '.join([f'${p:.2f}' for p in future_preds])}")
    else:
        st.warning("⚠️ No valid predictions generated. Check model weights or input shapes.")


    if len(future_preds) > 0:
        st.success(f"Predicted next {horizon} close(s): {', '.join([f'${p:.2f}' for p in future_preds])}")
        # append synthetic future to df for plotting/signals
        # Force both original and future DataFrames to use datetime type
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        future_df["Date"] = pd.to_datetime(future_df["Date"], errors="coerce")

# Now combine and sort
        df_combined = (
           pd.concat([df, future_df], ignore_index=True)
           .sort_values("Date")
           .reset_index(drop=True)
        )

        df_combined = enrich(df_combined)  # recompute indicators on entire series (fast for small horizon)
        # find future trades (only those with entry after historical last date)
        future_trades = find_future_signals(df, future_df, lookback_window=lookback_window, require_confirm=True,
                                       atr_mult=1.5, rr=2.0, min_vol_mult=1.0)
        st.subheader("🔔 Future Signals (based on forecasted bars)")
        if future_trades.empty:
            st.info("No high-confidence future signals found for selected horizon.")
        else:
            st.dataframe(future_trades[["entry_time","trade_type","entry_price","stop","target","confidence","tenure"]])
            st.metric("Future Trades", len(future_trades))
            # plot combined with markers
            plot_candles_with_signals(df_combined, future_trades, chart_days=200)
    else:
        st.warning("No predictions generated.")
    
    # After df is enriched
    df = detect_trend_reversals(df)
    plot_support_resistance(df)
# -----------------------------
# Inside main() AFTER df is loaded
# -----------------------------
    df = load_data(symbol)
    st.subheader(f"Stock: {symbol}")
    st.write("Data shape after loading:", df.shape)

    if df.empty:
        st.error("Loaded dataset is empty. Please check your CSV file or preprocessing pipeline.")
        st.stop()

    # ✅ Enrich data and generate signals here
    df_enriched = enrich(df)
    trades_df = generate_signals(df_enriched, atr_mult=1.5, rr=2.0, min_vol_mult=1.0, require_confirm=True)

    st.subheader("📈 Generated Signals / Trades")
    if trades_df.empty:
        st.info("No trades found with current parameters.")
    else:
        st.dataframe(
            trades_df[["entry_time", "entry_price", "exit_time", "exit_price", "pnl", "confidence", "trade_type"]]
            .sort_values("entry_time", ascending=False)
            .head(20)
        )
    st.metric("Total trades (period)", len(trades_df))

# Save trades for chart markers
    trades_csv_path = f"data/processed/{symbol}_trades.csv"
    trades_df.to_csv(trades_csv_path, index=False)


    # Chart
    preds = np.linspace(current_price, pred_price, num=lookback_window)
    plot_predictions(df, preds)

    # GenAI Analysis
    st.subheader(" AI Market Analysis")

    indicators = {
        "RSI": round(df["RSI"].iloc[-1], 2),
        "MACD": round(df["MACD"].iloc[-1], 2),
        "EMA Trend": "Short EMA above Long EMA" if df["EMA_short"].iloc[-1] > df["EMA_long"].iloc[-1] else "Short EMA below Long EMA",
        "ATR": round(df["ATR"].iloc[-1], 2)
    }

    fundamentals = {k: round(df[k].iloc[-1], 2) for k in fund_cols[:5]}  # summarize top 5 metrics

    if st.button(" Generate AI Explanation"):
        with st.spinner("Analyzing with GenAI..."):
            prompt = generate_prompt(symbol, current_price, pred_price, indicators, fundamentals)
            analysis = generate_analysis(prompt)
            st.success("Analysis Ready!")
            st.markdown(f"**{analysis}**")

    st.markdown("---")
    st.caption("Built by Lokeshwar • Powered by PyTorch, Streamlit, and GPT")
# =====================================
# 📊 SIGNAL DASHBOARD VIEW
# =====================================

from datetime import timedelta

from src.signals.signal_generator import (
    enrich,
    generate_signals,
    generate_signals_v3
)

def summarize_trades(trades_df):
    """Compute summary metrics for the signal dashboard."""
    if trades_df.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "avg_tenure_days": 0.0
        }
    total = len(trades_df)
    wins = (trades_df["pnl"] > 0).sum()
    avg_pnl = trades_df["pnl"].mean()
    # tenure
    avg_tenure = np.mean([
        (pd.to_datetime(r["exit_time"]) - pd.to_datetime(r["entry_time"])).days
        for _, r in trades_df.iterrows()
        if r["exit_time"] is not None
    ]) if "exit_time" in trades_df else 0

    return {
        "total_trades": total,
        "win_rate": round(100 * wins / total, 2),
        "avg_pnl": round(avg_pnl, 2),
        "avg_tenure_days": round(avg_tenure, 2)
    }

def plot_trades_chart(df, trades_df, title="Signal Visualization"):
    """Render candle chart with trade markers."""
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candlestick"
    )])

    if not trades_df.empty:
        fig.add_trace(go.Scatter(
            x=trades_df["entry_time"], y=trades_df["entry_price"],
            mode="markers", marker=dict(size=10, color="green", symbol="triangle-up"),
            name="Entry"
        ))
        fig.add_trace(go.Scatter(
            x=trades_df["exit_time"], y=trades_df["exit_price"],
            mode="markers", marker=dict(size=10, color="red", symbol="triangle-down"),
            name="Exit"
        ))

    fig.update_layout(template="plotly_dark", height=700, title=title)
    st.plotly_chart(fig, use_container_width=True)

def signal_dashboard(symbol, df):
    st.markdown("## 🧭 Signal Dashboard")
    st.caption("Compare different trading signal engines")

    # --- Select Engine ---
    engine = st.selectbox(
        "Choose Signal Engine",
        ["Candlestick (v1)", "Indicator (v2)", "Hybrid (v3)"],
        index=2
    )

    df_enriched = enrich(df)

    if engine == "Candlestick (v1)":
        from src.signals.signal_generator import generate_signals as signal_fn
        trades_df = signal_fn(df_enriched, atr_mult=1.5, rr=2.0)
    elif engine == "Indicator (v2)":
        from src.signals.signal_generator import generate_trading_signals as signal_fn
        trades_df = signal_fn(df_enriched)
    else:
        from src.signals.signal_generator import generate_signals_v3 as signal_fn
        trades_df = signal_fn(df_enriched, atr_mult=1.5, rr=2.0, min_vol_mult=1.2)

    # --- Show Summary ---
    stats = summarize_trades(trades_df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", stats["total_trades"])
    col2.metric("Win Rate", f"{stats['win_rate']}%")
    col3.metric("Avg PnL", f"{stats['avg_pnl']:.2f}")
    col4.metric("Avg Tenure (days)", stats["avg_tenure_days"])

    # --- Display Trades Table ---
    st.markdown("### 📋 Recent Trades")
    st.dataframe(
        trades_df.sort_values("entry_time", ascending=False)
        .head(20)[["entry_time", "trade_type", "entry_price", "exit_price", "pnl", "confidence", "tenure"]]
    )

    # --- Candle Chart ---
    plot_trades_chart(df, trades_df, title=f"{engine} Signals for {symbol}")


if __name__ == "__main__":
    main()
