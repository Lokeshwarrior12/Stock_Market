# src/visualization/plot_candles.py
import plotly.graph_objects as go
import pandas as pd

def plot_with_signals(csv_path, trades_csv=None, chart_title=None, show_indicators=True):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")])
    if show_indicators:
        if "EMA_short" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_short"], name="EMA short"))
        if "EMA_long" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_long"], name="EMA long"))
        if "VWAP" in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["VWAP"], name="VWAP"))
    if trades_csv:
        trades = pd.read_csv(trades_csv, parse_dates=["entry_time", "exit_time"])
        for _, t in trades.iterrows():
            fig.add_trace(go.Scatter(x=[t["entry_time"]], y=[t["entry_price"]], mode="markers", marker=dict(symbol="triangle-up", color="green", size=10), name="Entry"))
            fig.add_trace(go.Scatter(x=[t["exit_time"]], y=[t["exit_price"]], mode="markers", marker=dict(symbol="x", color="red", size=10), name="Exit"))
    fig.update_layout(title=chart_title or csv_path, xaxis_rangeslider_visible=False, template="plotly_dark", height=700)
    fig.show()
