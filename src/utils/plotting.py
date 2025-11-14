# src/utils/plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# --------------------------------------------
# 📈 1. Training Metrics
# --------------------------------------------
def plot_training_loss(train_losses, val_losses=None):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# --------------------------------------------
# 💹 2. Actual vs Predicted
# --------------------------------------------
def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted Prices"):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='orange')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


# --------------------------------------------
# 🧩 3. Correlation Heatmap
# --------------------------------------------
def plot_correlation_heatmap(df, figsize=(10, 8)):
    corr = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap='coolwarm', annot=False, center=0)
    plt.title('Feature Correlation Heatmap')
    plt.show()


# --------------------------------------------
# 🧠 4. Support & Resistance (Candlestick + Lines)
# --------------------------------------------
def plot_support_resistance(df, title="Support & Resistance"):
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name='Candlestick'
    )])

    if 'Support' in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Support"], mode='lines',
                                 name='Support', line=dict(color='green', width=1.5)))
    if 'Resistance' in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Resistance"], mode='lines',
                                 name='Resistance', line=dict(color='red', width=1.5)))

    fig.update_layout(title=title, xaxis_rangeslider_visible=False,
                      template='plotly_dark', height=600)
    fig.show()


# --------------------------------------------
# ⚙️ 5. Feature Trend Plot
# --------------------------------------------
def plot_feature_trends(df, features, n_points=200):
    plt.figure(figsize=(12, 6))
    subset = df.tail(n_points)
    for feat in features:
        plt.plot(subset[feat], label=feat)
    plt.title(f"Recent Trends ({n_points} points)")
    plt.legend()
    plt.grid(True)
    plt.show()
