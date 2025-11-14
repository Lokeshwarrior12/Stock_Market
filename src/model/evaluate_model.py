import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils.config import CONFIG
from src.model.train_lstm_model import LSTMModel


def load_data_and_model():
    """Load saved datasets and trained LSTMModel"""
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    cfg = CONFIG["training"]

    input_dim = X_test.shape[2]
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"]
    )
    model.load_state_dict(torch.load(cfg["model_save_path"], map_location="cpu"))
    model.eval()

    return model, X_test, y_test


def predict(model, X_test):
    """Run predictions on the test set"""
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy().flatten()
    return preds


def evaluate_metrics(y_true, preds):
    """Compute MAE, RMSE, and Directional Accuracy"""
    mae = mean_absolute_error(y_true, preds)
    rmse = mean_squared_error(y_true, preds) ** 0.5  # Manual sqrt for older sklearn
    direction_acc = np.mean(
        np.sign(preds[1:] - preds[:-1]) == np.sign(y_true[1:] - y_true[:-1])
    )
    return mae, rmse, direction_acc



def plot_predictions(y_true, preds, cfg):
    """Plot Actual vs Predicted prices"""
    days = np.arange(len(y_true))
    fig = go.Figure()

    # Actual & Predicted Lines
    fig.add_trace(go.Scatter(x=days, y=y_true, mode='lines', name='Actual Price', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=days, y=preds, mode='lines', name='Predicted Price', line=dict(color='orange')))

    # Mark bullish & bearish signals
    diff = np.sign(np.diff(preds))
    up_idx = np.where(diff > 0)[0]
    down_idx = np.where(diff < 0)[0]

    fig.add_trace(go.Scatter(x=days[up_idx], y=preds[up_idx], mode='markers',
                             name='Bullish Signal', marker=dict(color='green', size=6, symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=days[down_idx], y=preds[down_idx], mode='markers',
                             name='Bearish Signal', marker=dict(color='red', size=6, symbol='triangle-down')))

    fig.update_layout(
        title=f"Actual vs Predicted Prices ({CONFIG['stock_symbol']})",
        xaxis_title="Test Day Index",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=700
    )

    fig.show()


if __name__ == "__main__":
    print("[INFO] Loading model and data...")
    model, X_test, y_test = load_data_and_model()

    print("[INFO] Running predictions...")
    preds = predict(model, X_test)

    print("[INFO] Evaluating model performance...")
    mae, rmse, acc = evaluate_metrics(y_test, preds)
    print(f"[RESULT] MAE={mae:.4f} | RMSE={rmse:.4f} | Directional Accuracy={acc:.3f}")

    print("[INFO] Plotting results...")
    plot_predictions(y_test, preds, CONFIG)
    print("[SUCCESS] Evaluation complete")
