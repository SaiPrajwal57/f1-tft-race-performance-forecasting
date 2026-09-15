"""
Script: generate_ieee_figures.py
Generates the 5 IEEE publication-quality figures for the research paper:
- Fig 1: Proposed TFT Framework Architecture
- Fig 2: Sequence Construction & Sliding Window Setup
- Fig 3: Overall Model Comparison (MAE)
- Fig 4: Horizon-Wise MAE Comparison
- Fig 5: TFT Interpretability (Temporal Attention & VSN Saturation Analysis)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Ensure output directory exists
os.makedirs("figures", exist_ok=True)

# Set IEEE-style global plotting aesthetics
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["axes.edgecolor"] = "#2D3748"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.color"] = "#2D3748"
plt.rcParams["ytick.color"] = "#2D3748"
plt.rcParams["grid.color"] = "#E2E8F0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

# Color palette: Academic / IEEE restrained colors
CLR_PRIMARY = "#1E40AF"     # Deep Blue (TFT)
CLR_SECONDARY = "#DC2626"   # Crimson / Red (Persistence)
CLR_TERTIARY = "#0D9488"    # Teal / Green (LSTM)
CLR_ACCENT1 = "#D97706"     # Amber (Linear Regression)
CLR_ACCENT2 = "#4B5563"     # Slate Gray (Random Forest)
CLR_BG_CARD = "#F8FAFC"
CLR_BORDER = "#CBD5E1"


# ==============================================================================
# FIGURE 1: PROPOSED TFT FRAMEWORK ARCHITECTURE
# ==============================================================================
def generate_fig1():
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(-8, 102)
    ax.axis("off")

    def draw_box(x, y, w, h, text, subtitle=None, bg_color="#FFFFFF", border_color="#334155", text_color="#0F172A", radius=1.5, fontsize=10, bold=True):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={radius}",
                                     facecolor=bg_color, edgecolor=border_color, linewidth=1.4, zorder=3)
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, text, ha="center", va="center", fontsize=fontsize, fontweight="bold" if bold else "normal", color=text_color, zorder=4)
            ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", fontsize=fontsize-2, color="#475569", zorder=4, style="italic")
        else:
            ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold" if bold else "normal", color=text_color, zorder=4)

    def draw_arrow(x1, y1, x2, y2, color="#475569", width=1.5, text=None, text_pos=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=width, mutation_scale=14), zorder=2)
        if text and text_pos:
            ax.text(text_pos[0], text_pos[1], text, ha="center", va="center", fontsize=8, color="#64748B", backgroundcolor="#FFFFFF", zorder=5)

    # 1. Inputs Level
    draw_box(4, 88, 26, 8, "Static Metadata (6)", "Driver, Team, Race, Round, Year", bg_color="#EFF6FF", border_color="#3B82F6", text_color="#1E3A8A")
    draw_box(37, 88, 26, 8, "Observed Telemetry (26)", "Sectors, Speeds, Status, Weather", bg_color="#F0FDF4", border_color="#22C55E", text_color="#14532D")
    draw_box(70, 88, 26, 8, "Lags & Dynamics (20)", "Target Lags, Rolling Stats, Flags", bg_color="#FEF2F2", border_color="#EF4444", text_color="#7F1D1D")

    # Grouping Box for Inputs (t-19 to t)
    input_group = patches.FancyBboxPatch((2, 85), 96, 13, boxstyle="round,pad=0.5,rounding_size=2",
                                         facecolor="none", edgecolor="#94A3B8", linewidth=1.2, linestyle="--", zorder=1)
    ax.add_patch(input_group)
    ax.text(50, 97.2, "Historical Input Laps [t-19 ... t] (52 Features)", ha="center", va="center", fontsize=11, fontweight="bold", color="#334155")

    # 2. Embedding & Projections
    draw_arrow(17, 85, 17, 76)
    draw_arrow(50, 85, 50, 76)
    draw_arrow(83, 85, 83, 76)

    draw_box(4, 69, 26, 7, "Categorical Embeddings", "d_model = 64 (Lookup)", bg_color="#FFFFFF", border_color="#64748B")
    draw_box(37, 69, 59, 7, "Linear Feature Projections", "46 continuous & binary variables -> d_model = 64", bg_color="#FFFFFF", border_color="#64748B")

    # 3. Variable Selection Networks (VSN)
    draw_arrow(17, 69, 17, 60)
    draw_arrow(66.5, 69, 66.5, 60)

    draw_box(4, 52, 26, 8, "Static VSN", "Softmax over 6 variables", bg_color="#EFF6FF", border_color="#2563EB", text_color="#1E40AF")
    draw_box(37, 52, 59, 8, "Time-Varying VSN (per step)", "Softmax over 46 variables (t-19...t)", bg_color="#F8FAFC", border_color="#2563EB", text_color="#1E40AF")

    # Static context routing arrows
    draw_arrow(17, 52, 17, 43, color="#2563EB", text="c_s, c_e, c_h, c_c", text_pos=(23, 47.5))
    draw_arrow(17, 43, 37, 43, color="#2563EB")

    # 4. Temporal Processing Layers
    draw_arrow(66.5, 52, 66.5, 43)

    # LSTM Temporal Encoder
    draw_box(37, 36, 59, 7, "Sequence-to-Sequence LSTM Encoder", "Temporal dynamics & state conditioning (1 Layer, d=64)", bg_color="#FEF3C7", border_color="#D97706", text_color="#92400E")
    
    draw_arrow(66.5, 36, 66.5, 28)

    # Static Enrichment + Temporal Self-Attention
    draw_box(37, 21, 59, 7, "Interpretable Multi-Head Self-Attention", "4 Heads, d_model=64 with Gated Residual Connection (GLU)", bg_color="#EDE9FE", border_color="#7C3AED", text_color="#5B21B6")

    draw_arrow(66.5, 21, 66.5, 14)

    # 5. Position-wise Output & Multi-Horizon Decoder
    draw_box(37, 7, 59, 7, "Multi-Horizon Dense Projection Head", "Flattened Temporal Repr. (20 x 64) -> Dense -> 5 Horizons", bg_color="#F1F5F9", border_color="#475569", text_color="#0F172A")

    draw_arrow(66.5, 7, 66.5, 0.5)

    # 6. Forecast Outputs (t+1 to t+5)
    output_group = patches.FancyBboxPatch((25, -6.5), 50, 5.5, boxstyle="round,pad=0.4,rounding_size=1.5",
                                          facecolor="#ECFDF5", edgecolor="#10B981", linewidth=1.5, zorder=3)
    ax.add_patch(output_group)
    ax.text(50, -3.7, "Multi-Horizon LapTime Forecasts: [t+1,  t+2,  t+3,  t+4,  t+5]", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#065F46", zorder=4)

    plt.tight_layout()
    plt.savefig("figures/fig1_tft_architecture.png", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig1_tft_architecture.pdf", bbox_inches="tight")
    plt.close()
    print("[SUCCESS] Figure 1 generated: figures/fig1_tft_architecture.png & .pdf")


# ==============================================================================
# FIGURE 2: SEQUENCE CONSTRUCTION & PARTITION ISOLATION
# ==============================================================================
def generate_fig2():
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title & Partition Container
    ax.text(50, 95, "Chronological Sliding-Window Sequence Construction", ha="center", va="center", fontsize=13, fontweight="bold", color="#0F172A")
    ax.text(50, 90, "Enforces Strict Isolation across Race, Driver, and Season Boundaries", ha="center", va="center", fontsize=9.5, color="#64748B", style="italic")

    container = patches.FancyBboxPatch((4, 8), 92, 78, boxstyle="round,pad=0.5,rounding_size=2",
                                      facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.2, linestyle=":")
    ax.add_patch(container)
    ax.text(8, 82, "Driver-Race Sequence Group: (Year, RoundNumber, Driver)", fontsize=10, fontweight="bold", color="#334155")

    # Visualizing Sequence i
    ax.text(8, 68, "Sequence i (e.g., Lap 1 to 25):", fontsize=9.5, fontweight="bold", color="#1E293B")
    
    # 20 Historical Laps (Encoder)
    enc_rect1 = patches.Rectangle((8, 54), 56, 11, facecolor="#DBEAFE", edgecolor="#2563EB", linewidth=1.3, zorder=3)
    ax.add_patch(enc_rect1)
    ax.text(36, 59.5, "20 Historical Encoder Laps: [t-19 ... t]\n(Observed Telemetry, Weather, Lags, Rolling Stats)", 
            ha="center", va="center", fontsize=9, fontweight="bold", color="#1E40AF", zorder=4)

    # 5 Future Laps (Target Horizon)
    tar_rect1 = patches.Rectangle((67, 54), 25, 11, facecolor="#DCFCE7", edgecolor="#16A34A", linewidth=1.3, zorder=3)
    ax.add_patch(tar_rect1)
    ax.text(79.5, 59.5, "5 Target Horizons:\n[t+1 ... t+5] (LapTime)", 
            ha="center", va="center", fontsize=9, fontweight="bold", color="#14532D", zorder=4)

    # Sequence i+1 (Sliding forward by 1 lap)
    ax.text(8, 43, "Sequence i+1 (Sliding window step +1 lap):", fontsize=9.5, fontweight="bold", color="#1E293B")

    enc_rect2 = patches.Rectangle((11, 29), 56, 11, facecolor="#DBEAFE", edgecolor="#2563EB", linewidth=1.3, linestyle="--", zorder=3)
    ax.add_patch(enc_rect2)
    ax.text(39, 34.5, "Historical Laps: [t-18 ... t+1]", ha="center", va="center", fontsize=9, fontweight="bold", color="#1E40AF", zorder=4)

    tar_rect2 = patches.Rectangle((70, 29), 25, 11, facecolor="#DCFCE7", edgecolor="#16A34A", linewidth=1.3, linestyle="--", zorder=3)
    ax.add_patch(tar_rect2)
    ax.text(82.5, 34.5, "Target Horizons: [t+2 ... t+6]", ha="center", va="center", fontsize=9, fontweight="bold", color="#14532D", zorder=4)

    # Boundary Violation Guardrail
    guard = patches.FancyBboxPatch((8, 12), 84, 11, boxstyle="round,pad=0.2,rounding_size=1",
                                  facecolor="#FEF2F2", edgecolor="#EF4444", linewidth=1.2, zorder=3)
    ax.add_patch(guard)
    ax.text(50, 17.5, "Strict Anti-Leakage Boundary Rule: Sequences never cross race or driver borders.\nTail windows with < 5 future laps in the same race are strictly discarded (zero zero-padding).",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#991B1B", zorder=4)

    plt.tight_layout()
    plt.savefig("figures/fig2_sequence_construction.png", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig2_sequence_construction.pdf", bbox_inches="tight")
    plt.close()
    print("[SUCCESS] Figure 2 generated: figures/fig2_sequence_construction.png & .pdf")


# ==============================================================================
# FIGURE 3: OVERALL MODEL COMPARISON (MAE)
# ==============================================================================
def generate_fig3():
    models = ["TFT", "Persistence", "LSTM", "Linear Reg.", "Random Forest"]
    mae_values = [3.0957, 3.4899, 3.7860, 7.6437, 8.9038]
    colors = [CLR_PRIMARY, CLR_SECONDARY, CLR_TERTIARY, CLR_ACCENT1, CLR_ACCENT2]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    bars = ax.bar(models, mae_values, color=colors, width=0.55, edgecolor="#1E293B", linewidth=1.2, zorder=3)
    
    # Value annotations on top of bars
    for bar, val in zip(bars, mae_values):
        height = bar.get_height()
        ax.annotate(f"{val:.4f} s",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color="#0F172A")

    # Highlight relative improvement of TFT over Persistence & LSTM
    ax.annotate("Best Overall MAE\n(11.3% < Persistence)\n(18.2% < LSTM)", 
                xy=(0, 3.0957), xytext=(0.8, 5.0),
                arrowprops=dict(arrowstyle="->", color="#1E40AF", lw=1.5, connectionstyle="arc3,rad=-0.2"),
                fontsize=8.5, fontweight="bold", color="#1E40AF",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor="#3B82F6", lw=1))

    # Indicate "Lower is better"
    ax.annotate("Lower is better", xy=(4.2, 8.6), xytext=(4.2, 9.4),
                arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.2),
                fontsize=8.5, color="#64748B", style="italic", ha="center")

    ax.set_ylabel("Overall Mean Absolute Error (s)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Overall Multi-Horizon Lap Time Forecast Error (2024 Test Season)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, 10.5)
    ax.grid(axis="y", zorder=0)

    plt.tight_layout()
    plt.savefig("figures/fig3_model_comparison_mae.png", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig3_model_comparison_mae.pdf", bbox_inches="tight")
    plt.close()
    print("[SUCCESS] Figure 3 generated: figures/fig3_model_comparison_mae.png & .pdf")


# ==============================================================================
# FIGURE 4: HORIZON-WISE MAE COMPARISON
# ==============================================================================
def generate_fig4():
    horizons = ["Horizon 1\n(t+1)", "Horizon 2\n(t+2)", "Horizon 3\n(t+3)", "Horizon 4\n(t+4)", "Horizon 5\n(t+5)"]
    x = np.arange(len(horizons))

    persistence_mae = [2.4724, 3.2003, 3.5742, 3.9418, 4.2610]
    linear_mae = [4.3025, 7.3115, 8.6177, 8.6901, 9.2968]
    rf_mae = [3.6810, 6.4447, 9.8838, 12.1349, 12.3746]
    lstm_mae = [3.4530, 3.6791, 3.8140, 3.9263, 4.0576]
    tft_mae = [3.0916, 3.1311, 3.1028, 3.0678, 3.0855]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Plot lines with distinct markers and styling
    ax.plot(x, persistence_mae, marker="o", markersize=6, linewidth=2.0, color=CLR_SECONDARY, label="Persistence Baseline")
    ax.plot(x, lstm_mae, marker="^", markersize=6, linewidth=2.0, color=CLR_TERTIARY, label="LSTM Baseline")
    ax.plot(x, linear_mae, marker="s", markersize=5, linewidth=1.5, linestyle=":", color=CLR_ACCENT1, label="Linear Regression")
    ax.plot(x, rf_mae, marker="D", markersize=5, linewidth=1.5, linestyle=":", color=CLR_ACCENT2, label="Random Forest")
    ax.plot(x, tft_mae, marker="s", markersize=7, linewidth=2.6, color=CLR_PRIMARY, label="Temporal Fusion Transformer (TFT)", zorder=5)

    # Key callout annotations
    ax.annotate("Persistence wins H1\n(MAE: 2.4724 s)", xy=(0, 2.4724), xytext=(0.15, 1.4),
                arrowprops=dict(arrowstyle="->", color=CLR_SECONDARY, lw=1.3),
                fontsize=8.5, fontweight="bold", color=CLR_SECONDARY,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor=CLR_SECONDARY, lw=0.8))

    ax.annotate(r"TFT Dominates & Remains Stable (H2-H5)" + "\n" + r"(MAE: 3.07 - 3.13 s | $\Delta_{H1-H5} = -0.20\%$)", 
                xy=(3, 3.0678), xytext=(2.2, 5.2),
                arrowprops=dict(arrowstyle="->", color=CLR_PRIMARY, lw=1.3),
                fontsize=8.5, fontweight="bold", color=CLR_PRIMARY,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#EFF6FF", edgecolor=CLR_PRIMARY, lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (s)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Forecasting Error Across Step Horizons (H1 to H5)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(1.0, 13.5)
    ax.grid(True, zorder=0)
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig("figures/fig4_horizon_mae.png", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig4_horizon_mae.pdf", bbox_inches="tight")
    plt.close()
    print("[SUCCESS] Figure 4 generated: figures/fig4_horizon_mae.png & .pdf")


# ==============================================================================
# FIGURE 5: TFT INTERPRETABILITY (ATTENTION & VSN SATURATION)
# ==============================================================================
def generate_fig5():
    # Load verified attention data
    attn_df = pd.read_csv("results/attention_by_lag.csv")
    attn_df["LagNum"] = attn_df["HistoricalLag"].apply(lambda x: int(x.replace("Lap t-", "")))
    attn_sorted = attn_df.sort_values("LagNum", ascending=False).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300, gridspec_kw={"width_ratios": [1.25, 1.0]})

    # Panel A: Temporal Attention Across 20 Historical Lags
    lags = [f"t-{int(lag)}" for lag in attn_sorted["LagNum"]]
    x_indices = np.arange(len(lags))
    weights = attn_sorted["AttentionWeight"].values

    ax1.plot(x_indices, weights, marker="o", markersize=5.5, linewidth=2.0, color="#7C3AED", label="Mean Attention Weight")
    ax1.fill_between(x_indices, weights, alpha=0.15, color="#7C3AED")
    
    # Highlight t-0 and t-19
    ax1.scatter([x_indices[-1]], [weights[-1]], color="#DC2626", s=60, zorder=5)
    ax1.annotate(f"Max Recency (t-0):\n{weights[-1]:.3f}", xy=(19, weights[-1]), xytext=(14.0, 0.073),
                 arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2),
                 fontsize=8, fontweight="bold", color="#DC2626")

    ax1.scatter([x_indices[0]], [weights[0]], color="#2563EB", s=60, zorder=5)
    ax1.annotate(f"Window Anchor (t-19):\n{weights[0]:.3f}", xy=(0, weights[0]), xytext=(1.0, 0.070),
                 arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.2),
                 fontsize=8, fontweight="bold", color="#2563EB")

    ax1.set_title("(a) Mean Temporal Multi-Head Attention Across Historical Lags", fontsize=10, fontweight="bold", pad=10)
    ax1.set_xlabel("Historical Time Step", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Attention Weight", fontsize=9.5, fontweight="bold")
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(lags, rotation=45, ha="right", fontsize=8)
    ax1.grid(True)
    ax1.set_ylim(0.035, 0.088)

    # Panel B: Variable Selection Weight Distribution & Saturation
    vsn_labels = [
        "Static: Driver", 
        "Temporal: RollingLapTimeMean_5", 
        "Temporal: SpeedST", 
        "Temporal: WeatherTime", 
        "Remaining 48 Features\n(Tail Sum)"
    ]
    vsn_weights = [1.0, 0.999999, 1.77e-6, 3.01e-7, 2.92e-6]
    bar_colors = ["#2563EB", "#0D9488", "#94A3B8", "#94A3B8", "#E2E8F0"]

    ax2.barh(vsn_labels[::-1], [w if w > 1e-4 else 0.001 for w in vsn_weights[::-1]], 
             color=bar_colors[::-1], edgecolor="#334155", linewidth=1.0)

    ax2.text(0.5, 3.8, "Weight ≈ 1.000 (Saturated)", fontsize=8, fontweight="bold", color="#1E40AF", va="center")
    ax2.text(0.5, 2.8, "Weight ≈ 1.000 (Saturated)", fontsize=8, fontweight="bold", color="#0F766E", va="center")
    ax2.text(0.05, 1.8, "Weight = 1.77e-6 (< 0.001%)", fontsize=7.5, color="#64748B", va="center")
    ax2.text(0.05, 0.8, "Weight = 3.01e-7 (< 0.001%)", fontsize=7.5, color="#64748B", va="center")
    ax2.text(0.05, -0.2, "Sum < 3.0e-6 (Numerical Tail)", fontsize=7.5, color="#64748B", va="center")

    ax2.set_title("(b) Variable Selection Network (VSN) Saturation Profile", fontsize=10, fontweight="bold", pad=10)
    ax2.set_xlabel("Selection Weight (Softmax Output)", fontsize=9.5, fontweight="bold")
    ax2.set_xlim(0, 1.15)
    ax2.grid(axis="x")

    plt.tight_layout()
    plt.savefig("figures/fig5_tft_interpretability.png", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig5_tft_interpretability.pdf", bbox_inches="tight")
    plt.close()
    print("[SUCCESS] Figure 5 generated: figures/fig5_tft_interpretability.png & .pdf")


if __name__ == "__main__":
    print("[RUN] Generating all 5 IEEE publication figures...")
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    print("[COMPLETE] All 5 figures successfully created in 'figures/' directory.")
