# src/model/train_lstm_model.py
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.utils.config import CONFIG


# --------------------------------------------
# ⚙️ LSTM Model
# --------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# --------------------------------------------
# 🧹 Load Preprocessed Data
# --------------------------------------------
def load_data():

    def safe_load(path):
        arr = np.load(path, allow_pickle=True)
        # Convert all string / object arrays to numeric safely
        if arr.dtype.kind in {"U", "S", "O"}:
            arr = pd.to_numeric(arr.flatten(), errors="coerce")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.reshape(-1, 1)


    try:
        X_train = np.load("data/processed/X_train.npy", allow_pickle=True)
        y_train = safe_load("data/processed/y_train.npy")
        X_test = np.load("data/processed/X_test.npy", allow_pickle=True)
        y_test = safe_load("data/processed/y_test.npy")
    except FileNotFoundError:
        raise FileNotFoundError("Missing preprocessed .npy files in data/processed/. Run feature extraction first.")

    # Ensure all arrays are numeric
    X_train = np.nan_to_num(X_train.astype(float), nan=0.0)
    X_test = np.nan_to_num(X_test.astype(float), nan=0.0)
    y_train = np.nan_to_num(y_train.astype(float), nan=0.0)
    y_test = np.nan_to_num(y_test.astype(float), nan=0.0)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Reshape targets if needed
    if y_train.ndim == 1:
        y_train = y_train.unsqueeze(1)
    if y_test.ndim == 1:
        y_test = y_test.unsqueeze(1)

    return X_train, y_train, X_test, y_test



# --------------------------------------------
# 🧮 Training and Evaluation
# --------------------------------------------
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        if torch.isnan(loss):
            continue
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(train_loader))


def evaluate_model(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        y_true = y_test.numpy()

        preds = np.nan_to_num(preds, nan=0.0)
        y_true = np.nan_to_num(y_true, nan=0.0)

        mae = mean_absolute_error(y_true, preds)
        rmse = mean_squared_error(y_true, preds) ** 0.5
        direction_acc = np.mean(np.sign(np.diff(preds.flatten())) == np.sign(np.diff(y_true.flatten())))

    return mae, rmse, direction_acc


# --------------------------------------------
# 🧾 Metadata Saver (prevents mismatched shapes later)
# --------------------------------------------
def save_metadata(model_path, input_dim):
    metadata = {"model_type": "LSTM", "input_dim": input_dim}
    meta_path = os.path.splitext(model_path)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[INFO] Saved metadata to {meta_path}")


# --------------------------------------------
# 🚀 Main Execution
# --------------------------------------------
if __name__ == "__main__":
    cfg = CONFIG["training"]
    print("[INFO] Loading dataset...")
    X_train, y_train, X_test, y_test = load_data()
    input_dim = X_train.shape[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    print("[INFO] Starting training...")
    for epoch in range(cfg["epochs"]):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        mae, rmse, acc = evaluate_model(model, X_test, y_test, device)
        print(f"Epoch [{epoch+1}/{cfg['epochs']}]: Loss={loss:.6f} | RMSE={rmse:.4f} | MAE={mae:.4f} | DirACC={acc:.3f}")

    os.makedirs(os.path.dirname(cfg["model_save_path"]), exist_ok=True)
    torch.save(model.state_dict(), cfg["model_save_path"])
    save_metadata(cfg["model_save_path"], input_dim)
    print(f"[SUCCESS] Model saved to {cfg['model_save_path']}")
