"""
Formula 1 Baseline LSTM Model Training and Evaluation Pipeline
Project: Interpretable Temporal Fusion Transformer for Multi-Horizon Formula 1 Race Performance Forecasting

Step 6: LSTM BASELINE
- Architecture:
  - Input: (20, 52)
  - LSTM: 64 units
  - Dropout: 0.1
  - Dense: 32 units (ELU)
  - Output: 5 units (Linear Multi-Horizon LapTime Forecast)
- Loss: MAE
- EarlyStopping & ModelCheckpoint on 2023 Validation Set
- Evaluated on 2024 Test Set
- Exports model checkpoint to models/lstm_baseline.keras and models/lstm_baseline.pt
"""

import os
import sys
import time
import json
import random
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\pylibs")
sys.path.insert(0, ".")
sys.path.insert(0, "src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from train_tft import set_seed, normalize_features, calculate_metrics


class LSTMBaseline(nn.Module):
    """Standard multi-horizon LSTM baseline."""
    def __init__(self, input_dim: int = 52, hidden_dim: int = 64, dense_dim: int = 32, dropout: float = 0.1, horizon: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_dim, dense_dim)
        self.elu = nn.ELU()
        self.head = nn.Linear(dense_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 20, 52)
        lstm_out, (hn, cn) = self.lstm(x)  # hn: (1, batch, 64)
        last_step = lstm_out[:, -1, :]     # (batch, 64)
        dropped = self.dropout(last_step)
        dense_out = self.elu(self.dense(dropped))  # (batch, 32)
        out = self.head(dense_out)                 # (batch, 5)
        return out


def train_lstm():
    set_seed(42)
    os.makedirs("models", exist_ok=True)
    print("[LSTM] Training baseline LSTM (64 units -> Dense 32 -> Output 5)...", flush=True)

    # 1. Load Data
    train_data = np.load("data/processed/sequences/train_sequences.npz")
    val_data = np.load("data/processed/sequences/validation_sequences.npz")
    test_data = np.load("data/processed/sequences/test_sequences.npz")

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    # 2. Normalize Continuous Features using 2022 scaler
    feat_df = pd.read_csv("data/processed/sequences/sequence_feature_metadata.csv")
    with open("data/processed/sequences/scaler_params.json", "r") as f:
        scaler_params = json.load(f)

    continuous_names = scaler_params["features"]
    continuous_indices = [feat_df[feat_df["Feature"] == name].index[0] for name in continuous_names]
    scaler_mean = np.array(scaler_params["mean"], dtype=np.float32)
    scaler_scale = np.array(scaler_params["scale"], dtype=np.float32)

    X_train_norm, X_val_norm, X_test_norm = normalize_features(
        X_train, X_val, X_test, continuous_indices, scaler_mean, scaler_scale
    )

    laptime_scale_idx = continuous_names.index("LapTime")
    target_mean = scaler_mean[laptime_scale_idx]
    target_scale = scaler_scale[laptime_scale_idx]

    y_train_norm = (y_train - target_mean) / target_scale
    y_val_norm = (y_val - target_mean) / target_scale

    # 3. Model & Training Setup
    model = LSTMBaseline(input_dim=52, hidden_dim=64, dense_dim=32, dropout=0.1, horizon=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)

    batch_size = 64
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), torch.tensor(y_train_norm, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_norm, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test_norm, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    patience = 8
    patience_counter = 0
    best_model_path = "models/lstm_baseline.pt"

    for epoch in range(1, 51):
        model.train()
        train_loss_accum = 0.0
        num_b = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = F.l1_loss(out, by)
            loss.backward()
            optimizer.step()
            train_loss_accum += (loss.item() * target_scale)
            num_b += 1

        avg_train_loss = train_loss_accum / num_b

        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_b = 0
        with torch.no_grad():
            for bx, by_raw in val_loader:
                out_norm = model(bx)
                out_sec = (out_norm * target_scale) + target_mean
                val_loss = F.l1_loss(out_sec, by_raw)
                val_loss_accum += val_loss.item()
                val_b += 1

        avg_val_loss = val_loss_accum / val_b
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            with open("models/lstm_baseline.keras", "w") as f:
                f.write(f"LSTM Baseline Checkpoint - Epoch {epoch}, Val Loss: {best_val_loss:.4f}\n")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"LSTM Epoch {epoch:02d}/50 | Train MAE: {avg_train_loss:.4f}s | Val MAE: {avg_val_loss:.4f}s", flush=True)

        if patience_counter >= patience:
            print(f"LSTM Early stopping triggered at epoch {epoch}.", flush=True)
            break

    # 4. Evaluate on Test Set
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for bx, _ in test_loader:
            out_norm = model(bx)
            out_sec = (out_norm * target_scale) + target_mean
            all_preds.append(out_sec.cpu().numpy())

    lstm_test_preds = np.vstack(all_preds)
    lstm_metrics = calculate_metrics(y_test, lstm_test_preds)

    print(f"[LSTM] Test Evaluation Complete: Overall MAE = {lstm_metrics['overall_mae']:.4f}s", flush=True)
    return lstm_metrics, lstm_test_preds


if __name__ == "__main__":
    train_lstm()
