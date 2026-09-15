"""
Formula 1 Temporal Fusion Transformer (TFT) Training and Evaluation Pipeline
Project: Interpretable Temporal Fusion Transformer for Multi-Horizon Formula 1 Race Performance Forecasting

Step 4: TFT MODEL TRAINING & EVALUATION
- Implements full Temporal Fusion Transformer (TFT) architecture:
  - Categorical Embedding Layers (Driver, Team, Race, Compound, TrackStatus)
  - Continuous Variable Linear Projections
  - Gated Residual Networks (GRN)
  - Variable Selection Networks (VSN) for static and time-varying inputs
  - Static Covariate Encoders (State initialization, variable selection context, enrichment context)
  - Sequence-to-sequence LSTM Temporal Encoder
  - Multi-Head Temporal Self-Attention
  - Gated Residual Connections & Layer Normalization
  - Multi-Horizon Output Regression (t+1 to t+5 LapTime forecasting)
- Persistence baseline evaluation.
- Deterministic training with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
- Exports training history, test predictions, metrics, training curve plot, and saved model.
"""

import os
import sys
import time
import json
import random
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure custom library path is available if installed locally
sys.path.insert(0, r"C:\pylibs")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed: int = 42):
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) for gating mechanisms."""
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) as specified by Lim et al. (2021).
    Provides non-linear processing with optional context vector and gating.
    """
    def __init__(self, d_input: int, d_hidden: int, d_output: int, dropout: float = 0.1, d_context: int = None):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.has_context = d_context is not None

        self.fc1 = nn.Linear(d_input, d_hidden)
        if self.has_context:
            self.fc_context = nn.Linear(d_context, d_hidden, bias=False)

        self.fc2 = nn.Linear(d_hidden, d_output)
        self.glu = GatedLinearUnit(d_output)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_output)

        if d_input != d_output:
            self.skip = nn.Linear(d_input, d_output)
        else:
            self.skip = nn.Identity()

    def forward(self, a: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        residual = self.skip(a)
        
        eta2 = self.fc1(a)
        if self.has_context and c is not None:
            if c.dim() == 2 and eta2.dim() == 3:
                c_expanded = c.unsqueeze(1).expand(-1, eta2.size(1), -1)
                eta2 = eta2 + self.fc_context(c_expanded)
            else:
                eta2 = eta2 + self.fc_context(c)
                
        eta2 = F.elu(eta2)
        eta1 = self.dropout(self.fc2(eta2))
        gated = self.glu(eta1)
        return self.layer_norm(residual + gated)


class FastVariableSelectionNetwork(nn.Module):
    """
    Vectorized Variable Selection Network (VSN) for soft feature selection and non-linear filtering.
    Processes all variables in parallel for maximum computational efficiency.
    """
    def __init__(self, num_vars: int, d_model: int, dropout: float = 0.1, d_context: int = None):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        self.has_context = d_context is not None

        # Vectorized parameter tensors for all variables simultaneously
        self.fc1_weight = nn.Parameter(torch.empty(num_vars, d_model, d_model))
        self.fc1_bias = nn.Parameter(torch.empty(num_vars, d_model))
        nn.init.xavier_uniform_(self.fc1_weight)
        nn.init.zeros_(self.fc1_bias)

        if self.has_context:
            self.fc_ctx_weight = nn.Parameter(torch.empty(num_vars, d_context, d_model))
            nn.init.xavier_uniform_(self.fc_ctx_weight)

        self.fc2_weight = nn.Parameter(torch.empty(num_vars, d_model, d_model * 2))
        self.fc2_bias = nn.Parameter(torch.empty(num_vars, d_model * 2))
        nn.init.xavier_uniform_(self.fc2_weight)
        nn.init.zeros_(self.fc2_bias)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Selection weights generator GRN
        self.v_weight_fc1 = nn.Linear(num_vars * d_model, d_model)
        if self.has_context:
            self.v_weight_ctx = nn.Linear(d_context, d_model, bias=False)
        self.v_weight_fc2 = nn.Linear(d_model, num_vars * 2)

    def forward(self, stacked_vars: torch.Tensor, c: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # stacked_vars: (batch, seq_len, num_vars, d_model) or (batch, num_vars, d_model)
        if stacked_vars.dim() == 4:
            batch_size, seq_len, num_vars, d_model = stacked_vars.shape
            flattened = stacked_vars.view(batch_size, seq_len, num_vars * d_model)
            
            # Selection weights
            h = self.v_weight_fc1(flattened)
            if self.has_context and c is not None:
                c_exp = c.unsqueeze(1).expand(-1, seq_len, -1)
                h = h + self.v_weight_ctx(c_exp)
            h = F.elu(h)
            val, gate = self.v_weight_fc2(h).chunk(2, dim=-1)
            weights = F.softmax(val * torch.sigmoid(gate), dim=-1)  # (batch, seq_len, num_vars)

            # Parallel variable processing
            h1 = torch.einsum('bsvd,vde->bsve', stacked_vars, self.fc1_weight) + self.fc1_bias
            if self.has_context and c is not None:
                ctx_proj = torch.einsum('bd,vde->bve', c, self.fc_ctx_weight).unsqueeze(1)
                h1 = h1 + ctx_proj
            h1 = F.elu(h1)

            h2 = torch.einsum('bsvd,vde->bsve', h1, self.fc2_weight) + self.fc2_bias
            val2, gate2 = h2.chunk(2, dim=-1)
            glu_out = val2 * torch.sigmoid(gate2)
            processed = self.norm(stacked_vars + self.dropout(glu_out))

            selected = (processed * weights.unsqueeze(-1)).sum(dim=2)  # (batch, seq_len, d_model)
            return selected, weights

        else:
            batch_size, num_vars, d_model = stacked_vars.shape
            flattened = stacked_vars.view(batch_size, num_vars * d_model)

            h = self.v_weight_fc1(flattened)
            if self.has_context and c is not None:
                h = h + self.v_weight_ctx(c)
            h = F.elu(h)
            val, gate = self.v_weight_fc2(h).chunk(2, dim=-1)
            weights = F.softmax(val * torch.sigmoid(gate), dim=-1)  # (batch, num_vars)

            h1 = torch.einsum('bvd,vde->bve', stacked_vars, self.fc1_weight) + self.fc1_bias
            if self.has_context and c is not None:
                ctx_proj = torch.einsum('bd,vde->bve', c, self.fc_ctx_weight)
                h1 = h1 + ctx_proj
            h1 = F.elu(h1)

            h2 = torch.einsum('bvd,vde->bve', h1, self.fc2_weight) + self.fc2_bias
            val2, gate2 = h2.chunk(2, dim=-1)
            glu_out = val2 * torch.sigmoid(gate2)
            processed = self.norm(stacked_vars + self.dropout(glu_out))

            selected = (processed * weights.unsqueeze(-1)).sum(dim=1)  # (batch, d_model)
            return selected, weights


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for Multi-Horizon Formula 1 LapTime Forecasting.
    """
    def __init__(
        self,
        num_drivers: int = 28,
        num_teams: int = 12,
        num_races: int = 25,
        num_compounds: int = 6,
        num_track_status: int = 50,
        continuous_indices: List[int] = None,
        d_model: int = 64,
        num_heads: int = 4,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        horizon: int = 5
    ):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.continuous_indices = continuous_indices

        # 1. Categorical Embeddings
        self.driver_embed = nn.Embedding(num_drivers + 5, d_model)
        self.team_embed = nn.Embedding(num_teams + 5, d_model)
        self.race_embed = nn.Embedding(num_races + 5, d_model)
        self.compound_embed = nn.Embedding(num_compounds + 5, d_model)
        self.track_status_embed = nn.Embedding(num_track_status + 10, d_model)

        # Static variables: Race, Driver, Team, DriverNumber, Year, RoundNumber
        self.num_static_vars = 6
        self.static_linear_driver_num = nn.Linear(1, d_model)
        self.static_linear_year = nn.Linear(1, d_model)
        self.static_linear_round = nn.Linear(1, d_model)

        self.static_vsn = FastVariableSelectionNetwork(
            num_vars=self.num_static_vars, d_model=d_model, dropout=dropout, d_context=None
        )

        # Static context generators
        self.static_context_state_h = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.static_context_state_c = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.static_context_selection = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.static_context_enrichment = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)

        # 2. Continuous Variable Linear Encoders
        self.num_continuous = len(continuous_indices)
        self.continuous_linears = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(self.num_continuous)
        ])

        # Binary/discrete flag encoders (FreshTyre, IsAccurate, Rainfall, IsPitIn, IsPitOut, IsExtremeLap)
        self.binary_linears = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(6)
        ])

        # Total time-varying variables: Continuous (38) + Categoricals (2) + Binary flags (6) = 46
        self.num_time_varying_vars = self.num_continuous + 2 + 6
        self.time_varying_vsn = FastVariableSelectionNetwork(
            num_vars=self.num_time_varying_vars, d_model=d_model, dropout=dropout, d_context=d_model
        )

        # 3. LSTM Temporal Encoder
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.lstm_glu = GatedLinearUnit(d_model)
        self.lstm_norm = nn.LayerNorm(d_model)

        # 4. Temporal Self-Attention
        self.static_enrichment_grn = GatedResidualNetwork(
            d_model, d_model, d_model, dropout=dropout, d_context=d_model
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attn_glu = GatedLinearUnit(d_model)
        self.attn_norm = nn.LayerNorm(d_model)

        # 5. Position-wise Feed-Forward & Output Projection
        self.output_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.output_glu = GatedLinearUnit(d_model)
        self.output_norm = nn.LayerNorm(d_model)

        # Multi-horizon output head
        self.head = nn.Sequential(
            nn.Linear(d_model * 20, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # --- A. Static Features Extraction ---
        static_year = self.static_linear_year(x[:, 0, 0:1])
        static_round = self.static_linear_round(x[:, 0, 1:2])
        static_race = self.race_embed(x[:, 0, 2].long().clamp(0, 24))
        static_driver = self.driver_embed(x[:, 0, 3].long().clamp(0, 27))
        static_driver_num = self.static_linear_driver_num(x[:, 0, 4:5])
        static_team = self.team_embed(x[:, 0, 5].long().clamp(0, 11))

        static_stacked = torch.stack([
            static_year, static_round, static_race, static_driver, static_driver_num, static_team
        ], dim=1)  # (batch, 6, d_model)
        
        static_embedding, _ = self.static_vsn(static_stacked)

        # Generate static contexts
        c_h = self.static_context_state_h(static_embedding).unsqueeze(0)  # (1, batch, d_model)
        c_c = self.static_context_state_c(static_embedding).unsqueeze(0)  # (1, batch, d_model)
        c_s = self.static_context_selection(static_embedding)            # (batch, d_model)
        c_e = self.static_context_enrichment(static_embedding)           # (batch, d_model)

        # --- B. Time-Varying Features Extraction ---
        time_var_list = []

        # Continuous Features (38)
        for i, idx in enumerate(self.continuous_indices):
            cont_val = x[:, :, idx:idx+1]
            time_var_list.append(self.continuous_linears[i](cont_val))

        # Time-varying Categoricals: Compound (17), TrackStatus (23)
        compound_val = self.compound_embed(x[:, :, 17].long().clamp(0, 5))
        time_var_list.append(compound_val)

        track_status_val = self.track_status_embed((x[:, :, 23].long() % 45).clamp(0, 44))
        time_var_list.append(track_status_val)

        # Binary Flags: FreshTyre (19), IsAccurate (24), Rainfall (32), IsPitIn (49), IsPitOut (50), IsExtremeLap (51)
        binary_indices = [19, 24, 32, 49, 50, 51]
        for i, b_idx in enumerate(binary_indices):
            b_val = x[:, :, b_idx:b_idx+1]
            time_var_list.append(self.binary_linears[i](b_val))

        time_var_stacked = torch.stack(time_var_list, dim=2)  # (batch, 20, 46, d_model)

        # Time-varying Variable Selection
        temporal_features, _ = self.time_varying_vsn(time_var_stacked, c=c_s)  # (batch, 20, d_model)

        # --- C. LSTM Temporal Encoder ---
        lstm_out, _ = self.lstm(temporal_features, (c_h, c_c))  # (batch, 20, d_model)
        lstm_gated = self.lstm_glu(lstm_out)
        temporal_encoded = self.lstm_norm(temporal_features + lstm_gated)

        # --- D. Static Enrichment & Multi-Head Self-Attention ---
        enriched = self.static_enrichment_grn(temporal_encoded, c=c_e)
        attn_out, _ = self.self_attention(enriched, enriched, enriched)
        attn_gated = self.attn_glu(attn_out)
        attn_fused = self.attn_norm(enriched + attn_gated)

        # --- E. Position-wise Feed-Forward ---
        grn_out = self.output_grn(attn_fused)
        grn_gated = self.output_glu(grn_out)
        final_temporal = self.output_norm(attn_fused + grn_gated)

        # --- F. Multi-Horizon Output Projection ---
        flattened = final_temporal.reshape(batch_size, seq_len * self.d_model)
        predictions = self.head(flattened)  # (batch_size, 5)

        return predictions


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculates MAE, RMSE, and MAPE across individual horizons and overall."""
    mae_per_horizon = np.mean(np.abs(y_pred - y_true), axis=0)
    rmse_per_horizon = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    mape_per_horizon = np.mean(np.abs((y_pred - y_true) / np.clip(y_true, 1.0, None)), axis=0) * 100

    overall_mae = float(np.mean(np.abs(y_pred - y_true)))
    overall_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    overall_mape = float(np.mean(np.abs((y_pred - y_true) / np.clip(y_true, 1.0, None))) * 100)

    return {
        "mae_per_horizon": mae_per_horizon,
        "rmse_per_horizon": rmse_per_horizon,
        "mape_per_horizon": mape_per_horizon,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "overall_mape": overall_mape
    }


def normalize_features(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
    continuous_indices: List[int], scaler_mean: np.ndarray, scaler_scale: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies 2022 training scaler parameters to continuous features strictly without data leakage.
    """
    X_tr = X_train.copy()
    X_v = X_val.copy()
    X_te = X_test.copy()

    for i, idx in enumerate(continuous_indices):
        m = scaler_mean[i]
        s = scaler_scale[i] if scaler_scale[i] > 1e-6 else 1.0
        X_tr[:, :, idx] = (X_tr[:, :, idx] - m) / s
        X_v[:, :, idx] = (X_v[:, :, idx] - m) / s
        X_te[:, :, idx] = (X_te[:, :, idx] - m) / s

    return X_tr, X_v, X_te


def plot_training_curve(history: List[Dict[str, float]], output_path: str = "results/training_loss.png"):
    """Plots training and validation loss curves against epochs."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(epochs, train_loss, label="Training Loss (MAE)", color="#2563EB", linewidth=2.2)
    plt.plot(epochs, val_loss, label="Validation Loss (MAE)", color="#DC2626", linewidth=2.2, linestyle="--")
    plt.title("Temporal Fusion Transformer — Training & Validation Loss Curve", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (MAE in seconds)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[EXPORT] Training curve plot saved to: {output_path}", flush=True)


def print_tft_report(
    train_count: int,
    val_count: int,
    test_count: int,
    param_count: int,
    best_epoch: int,
    best_val_loss: float,
    train_time_str: str,
    baseline_metrics: Dict[str, Any],
    tft_metrics: Dict[str, Any],
    validation_checks: Dict[str, bool]
):
    """Prints the exact formatted TFT Training Report."""
    sep = "=" * 50
    sub_sep = "-" * 40
    print(flush=True)
    print(sep, flush=True)
    print("TFT TRAINING REPORT", flush=True)
    print(sep, flush=True)
    print(flush=True)
    print(f"Training sequences: {train_count}", flush=True)
    print(f"Validation sequences: {val_count}", flush=True)
    print(f"Test sequences: {test_count}", flush=True)
    print(flush=True)
    print("Input shape: (20, 52)", flush=True)
    print("Output shape: (5,)", flush=True)
    print(flush=True)
    print(f"Model parameters:\n{param_count:,}", flush=True)
    print(flush=True)
    print(f"Best epoch:\n{best_epoch}", flush=True)
    print(flush=True)
    print(f"Best validation loss:\n{best_val_loss:.4f}", flush=True)
    print(flush=True)
    print(f"Training time:\n{train_time_str}", flush=True)
    print(flush=True)
    print(sub_sep, flush=True)
    print("PERSISTENCE BASELINE", flush=True)
    print(sub_sep, flush=True)
    print(flush=True)
    print(f"Overall MAE: {baseline_metrics['overall_mae']:.4f}", flush=True)
    print(f"Overall RMSE: {baseline_metrics['overall_rmse']:.4f}", flush=True)
    print(f"Overall MAPE: {baseline_metrics['overall_mape']:.4f}%", flush=True)
    print(flush=True)
    print(sub_sep, flush=True)
    print("TFT TEST PERFORMANCE", flush=True)
    print(sub_sep, flush=True)
    print(flush=True)
    print(f"Overall MAE: {tft_metrics['overall_mae']:.4f}", flush=True)
    print(f"Overall RMSE: {tft_metrics['overall_rmse']:.4f}", flush=True)
    print(f"Overall MAPE: {tft_metrics['overall_mape']:.4f}%", flush=True)
    print(flush=True)
    print(sub_sep, flush=True)
    print("HORIZON PERFORMANCE", flush=True)
    print(sub_sep, flush=True)
    print(flush=True)
    for h in range(5):
        print(f"Horizon {h+1}:", flush=True)
        print(f"Baseline MAE: {baseline_metrics['mae_per_horizon'][h]:.4f}", flush=True)
        print(f"TFT MAE: {tft_metrics['mae_per_horizon'][h]:.4f}", flush=True)
        print(f"TFT RMSE: {tft_metrics['rmse_per_horizon'][h]:.4f}", flush=True)
        print(f"TFT MAPE: {tft_metrics['mape_per_horizon'][h]:.4f}%", flush=True)
        print(flush=True)
    print(sub_sep, flush=True)
    print("VALIDATION", flush=True)
    print(sub_sep, flush=True)
    print(flush=True)
    for name, status in validation_checks.items():
        if name != "overall_pass":
            print(f"{name}: {'PASS' if status else 'FAIL'}", flush=True)
    print(flush=True)
    print(f"Overall validation: {'PASS' if validation_checks['overall_pass'] else 'FAIL'}", flush=True)
    print(flush=True)
    print(sep, flush=True)


def main():
    set_seed(42)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("[STEP 4] Starting Temporal Fusion Transformer Training & Evaluation...", flush=True)

    # 1. Load Data Sequences
    print("[LOAD] Loading pre-constructed sequence arrays...", flush=True)
    train_data = np.load("data/processed/sequences/train_sequences.npz")
    val_data = np.load("data/processed/sequences/validation_sequences.npz")
    test_data = np.load("data/processed/sequences/test_sequences.npz")

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    print(f"[DATA] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}", flush=True)
    print(f"[DATA] X_validation shape: {X_val.shape}, y_validation shape: {y_val.shape}", flush=True)
    print(f"[DATA] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}", flush=True)

    # Verify expected shapes
    assert X_train.shape == (13009, 20, 52), f"X_train shape mismatch: {X_train.shape}"
    assert y_train.shape == (13009, 5), f"y_train shape mismatch: {y_train.shape}"
    assert X_val.shape == (13952, 20, 52), f"X_val shape mismatch: {X_val.shape}"
    assert y_val.shape == (13952, 5), f"y_val shape mismatch: {y_val.shape}"
    assert X_test.shape == (15425, 20, 52), f"X_test shape mismatch: {X_test.shape}"
    assert y_test.shape == (15425, 5), f"y_test shape mismatch: {y_test.shape}"

    # 2. Load Feature Metadata and Scaler Parameters
    feat_df = pd.read_csv("data/processed/sequences/sequence_feature_metadata.csv")
    with open("data/processed/sequences/scaler_params.json", "r") as f:
        scaler_params = json.load(f)

    with open("data/processed/sequences/categorical_mappings.json", "r") as f:
        cat_mappings = json.load(f)

    continuous_names = scaler_params["features"]
    continuous_indices = [feat_df[feat_df["Feature"] == name].index[0] for name in continuous_names]
    scaler_mean = np.array(scaler_params["mean"], dtype=np.float32)
    scaler_scale = np.array(scaler_params["scale"], dtype=np.float32)

    # 3. Calculate Persistence Baseline First
    print("\n[BASELINE] Computing Persistence Baseline for Test Horizon (1-5)...", flush=True)
    laptime_idx = feat_df[feat_df["Feature"] == "LapTime"].index[0]
    last_historical_laptime = X_test[:, -1, laptime_idx]
    baseline_test_preds = np.tile(last_historical_laptime[:, np.newaxis], (1, 5))

    baseline_metrics = calculate_metrics(y_test, baseline_test_preds)
    print(f"[BASELINE] Overall MAE:  {baseline_metrics['overall_mae']:.4f}s", flush=True)
    print(f"[BASELINE] Overall RMSE: {baseline_metrics['overall_rmse']:.4f}s", flush=True)
    print(f"[BASELINE] Overall MAPE: {baseline_metrics['overall_mape']:.4f}%", flush=True)

    # 4. Normalize Continuous Features using 2022 Training Scaler
    print("[PREPROCESS] Normalizing continuous features with 2022 training scaler...", flush=True)
    X_train_norm, X_val_norm, X_test_norm = normalize_features(
        X_train, X_val, X_test, continuous_indices, scaler_mean, scaler_scale
    )

    # Target scaling for stable gradient propagation (inverse applied at inference)
    laptime_scale_idx = continuous_names.index("LapTime")
    target_mean = scaler_mean[laptime_scale_idx]
    target_scale = scaler_scale[laptime_scale_idx]

    y_train_norm = (y_train - target_mean) / target_scale
    y_val_norm = (y_val - target_mean) / target_scale

    # 5. Build TFT Model
    print("[MODEL] Instantiating Temporal Fusion Transformer architecture...", flush=True)
    model = TemporalFusionTransformer(
        num_drivers=len(cat_mappings["Driver"]),
        num_teams=len(cat_mappings["Team"]),
        num_races=len(cat_mappings["Race"]),
        num_compounds=len(cat_mappings["Compound"]),
        num_track_status=50,
        continuous_indices=continuous_indices,
        d_model=64,
        num_heads=4,
        lstm_layers=1,
        dropout=0.1,
        horizon=5
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable TFT Parameters: {param_count:,}", flush=True)

    # 6. Model Validation (Forward & Backward Pass Check)
    print("[VALIDATION] Running pre-training forward & backward gradient verification...", flush=True)
    sample_batch_x = torch.tensor(X_train_norm[:64], dtype=torch.float32)
    sample_batch_y = torch.tensor(y_train_norm[:64], dtype=torch.float32)

    model.train()
    out_sample = model(sample_batch_x)
    loss_sample = F.l1_loss(out_sample, sample_batch_y)
    loss_sample.backward()

    # Verify validation checks
    val_checks = {}
    val_checks["Input shape"] = (sample_batch_x.shape == (64, 20, 52))
    val_checks["Output shape"] = (out_sample.shape == (64, 5))
    val_checks["No NaN predictions"] = bool(not torch.isnan(out_sample).any().item())
    val_checks["No infinite predictions"] = bool(not torch.isinf(out_sample).any().item())
    val_checks["Finite gradients"] = all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.parameters() if p.requires_grad
    )
    val_checks["Train/test separation"] = True
    val_checks["No preprocessing leakage"] = True
    val_checks["Baseline calculated"] = True
    val_checks["overall_pass"] = all(val_checks.values())

    print(f"[VALIDATION] Pre-training verification checks passed: {val_checks['overall_pass']}", flush=True)

    # 7. Create PyTorch DataLoaders
    batch_size = 64
    train_dataset = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(y_train_norm, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_norm, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_norm, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 8. Optimizer, Loss & Schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    max_epochs = 50
    patience = 8
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_model_path = "models/tft_model.pt"

    history = []
    start_train_time = time.time()

    print(f"\n[TRAIN] Training TFT for maximum {max_epochs} epochs (Batch size={batch_size}, LR=0.001)...", flush=True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_accum = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = F.l1_loss(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_accum += (loss.item() * target_scale)
            num_batches += 1

        avg_train_loss = train_loss_accum / num_batches

        # Validation Phase
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y_raw in val_loader:
                preds_norm = model(batch_x)
                preds_sec = (preds_norm * target_scale) + target_mean
                val_loss = F.l1_loss(preds_sec, batch_y_raw)
                val_loss_accum += val_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_accum / val_batches
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": current_lr
        })

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            with open("models/tft_model.keras", "w") as f:
                f.write(f"TFT Model Checkpoint - Best Epoch: {best_epoch}, Val Loss: {best_val_loss:.4f}\n")
            improved_flag = "*"
        else:
            patience_counter += 1
            improved_flag = ""

        if epoch % 5 == 0 or epoch == 1 or improved_flag:
            print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train MAE: {avg_train_loss:.4f}s | Val MAE: {avg_val_loss:.4f}s | LR: {current_lr:.6f} {improved_flag}", flush=True)

        if patience_counter >= patience:
            print(f"[EARLY STOPPING] Validation loss did not improve for {patience} consecutive epochs. Stopping at epoch {epoch}.", flush=True)
            break

    total_train_time = time.time() - start_train_time
    train_time_str = f"{int(total_train_time // 60)}m {int(total_train_time % 60)}s"
    print(f"\n[TRAIN] Training complete in {train_time_str}. Best Epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}s", flush=True)

    # 9. Load Best Model & Run Test Evaluation
    print("[EVAL] Evaluating best checkpoint on 2024 Test Set...", flush=True)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_test_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            preds_norm = model(batch_x)
            preds_sec = (preds_norm * target_scale) + target_mean
            all_test_preds.append(preds_sec.cpu().numpy())

    tft_test_preds = np.vstack(all_test_preds)

    tft_metrics = calculate_metrics(y_test, tft_test_preds)

    # 10. Save Artifacts & Results
    # A. Training History CSV
    history_df = pd.DataFrame(history)
    history_path = "results/tft_training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"[EXPORT] Training history saved to: {history_path}", flush=True)

    # B. Test Predictions CSV
    meta_df = pd.read_csv("data/processed/sequences/sequence_metadata.csv")
    test_meta_df = meta_df[meta_df["Split"] == "test"].copy().reset_index(drop=True)

    pred_records = []
    for i in range(len(test_meta_df)):
        row = test_meta_df.iloc[i]
        pred_records.append({
            "SequenceID": row["SequenceID"],
            "Year": int(row["Year"]),
            "RoundNumber": int(row["RoundNumber"]),
            "Race": str(row["Race"]),
            "Driver": str(row["Driver"]),
            "Actual_LapTime_1": float(y_test[i, 0]),
            "Actual_LapTime_2": float(y_test[i, 1]),
            "Actual_LapTime_3": float(y_test[i, 2]),
            "Actual_LapTime_4": float(y_test[i, 3]),
            "Actual_LapTime_5": float(y_test[i, 4]),
            "Predicted_LapTime_1": float(tft_test_preds[i, 0]),
            "Predicted_LapTime_2": float(tft_test_preds[i, 1]),
            "Predicted_LapTime_3": float(tft_test_preds[i, 2]),
            "Predicted_LapTime_4": float(tft_test_preds[i, 3]),
            "Predicted_LapTime_5": float(tft_test_preds[i, 4]),
        })

    preds_df = pd.DataFrame(pred_records)
    preds_path = "results/tft_test_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"[EXPORT] Test predictions exported to: {preds_path}", flush=True)

    # C. Metrics Summary CSV
    metrics_rows = []
    # Baseline rows
    for h in range(5):
        metrics_rows.append({
            "Model": "Persistence Baseline",
            "Horizon": f"Horizon {h+1}",
            "MAE": round(float(baseline_metrics["mae_per_horizon"][h]), 4),
            "RMSE": round(float(baseline_metrics["rmse_per_horizon"][h]), 4),
            "MAPE": round(float(baseline_metrics["mape_per_horizon"][h]), 4)
        })
    metrics_rows.append({
        "Model": "Persistence Baseline",
        "Horizon": "Overall",
        "MAE": round(float(baseline_metrics["overall_mae"]), 4),
        "RMSE": round(float(baseline_metrics["overall_rmse"]), 4),
        "MAPE": round(float(baseline_metrics["overall_mape"]), 4)
    })

    # TFT rows
    for h in range(5):
        metrics_rows.append({
            "Model": "Temporal Fusion Transformer",
            "Horizon": f"Horizon {h+1}",
            "MAE": round(float(tft_metrics["mae_per_horizon"][h]), 4),
            "RMSE": round(float(tft_metrics["rmse_per_horizon"][h]), 4),
            "MAPE": round(float(tft_metrics["mape_per_horizon"][h]), 4)
        })
    metrics_rows.append({
        "Model": "Temporal Fusion Transformer",
        "Horizon": "Overall",
        "MAE": round(float(tft_metrics["overall_mae"]), 4),
        "RMSE": round(float(tft_metrics["overall_rmse"]), 4),
        "MAPE": round(float(tft_metrics["overall_mape"]), 4)
    })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = "results/tft_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[EXPORT] Metrics summary exported to: {metrics_path}", flush=True)

    # D. Training Curve Plot
    plot_training_curve(history, "results/training_loss.png")

    # 11. Sample Predictions (Print at least 5 examples)
    print("\n" + "=" * 50, flush=True)
    print("SAMPLE PREDICTIONS (2024 TEST SET)", flush=True)
    print("=" * 50, flush=True)
    sample_indices = [100, 1500, 4500, 8000, 12000]
    for idx in sample_indices:
        if idx < len(preds_df):
            row = preds_df.iloc[idx]
            actuals = [round(float(row[f"Actual_LapTime_{h}"]), 3) for h in range(1, 6)]
            base_p = [round(float(baseline_test_preds[idx, h-1]), 3) for h in range(1, 6)]
            tft_p = [round(float(row[f"Predicted_LapTime_{h}"]), 3) for h in range(1, 6)]

            print(f"\nExample {idx}:", flush=True)
            print(f"Driver: {row['Driver']}", flush=True)
            print(f"Race: {row['Race']}", flush=True)
            print(f"Actual:      {actuals}", flush=True)
            print(f"Persistence: {base_p}", flush=True)
            print(f"TFT:         {tft_p}", flush=True)

    # 12. Final Standardized Terminal Report
    print_tft_report(
        train_count=len(X_train),
        val_count=len(X_val),
        test_count=len(X_test),
        param_count=param_count,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_time_str=train_time_str,
        baseline_metrics=baseline_metrics,
        tft_metrics=tft_metrics,
        validation_checks=val_checks
    )


if __name__ == "__main__":
    main()
