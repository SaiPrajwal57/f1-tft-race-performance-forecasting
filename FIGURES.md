# IEEE Research Paper Figures Specification

This document details the 5 publication-quality figures generated for the research paper: *"Interpretable Temporal Fusion Transformer for Multi-Horizon Formula 1 Race Performance Forecasting"*.

All figures are rendered at 300 DPI high-resolution PNG format and vector PDF format inside [`figures/`](file:///e:/paper/figures).

---

## FIGURE 1 — PROPOSED TFT FRAMEWORK ARCHITECTURE

- **File Formats**: [`figures/fig1_tft_architecture.png`](file:///e:/paper/figures/fig1_tft_architecture.png) | [`figures/fig1_tft_architecture.pdf`](file:///e:/paper/figures/fig1_tft_architecture.pdf)
- **Caption**: *Fig. 1. End-to-end architecture of the proposed Temporal Fusion Transformer for multi-horizon Formula 1 lap time forecasting. The model ingests 52 features across 20 historical laps $[t-19 \dots t]$, processes them through categorical embeddings and linear projections, filters variables via Static and Time-Varying Variable Selection Networks (VSN), captures dynamic trajectory patterns with an LSTM temporal encoder and Interpretable Multi-Head Self-Attention, and predicts multi-step lap times across five future horizons $[t+1 \dots t+5]$.*
- **Source Code & Data**: [`src/train_tft.py`](file:///e:/paper/src/train_tft.py#L50-L362), [`src/sequence_construction.py`](file:///e:/paper/src/sequence_construction.py#L143-L165).
- **What It Demonstrates**:
  - The structural flow from raw 52-dimensional input tensors to multi-step predictions.
  - The explicit separation between historical context ($t-19 \dots t$) and future multi-step outputs ($t+1 \dots t+5$).
  - The interaction between static entity metadata (driver, team, circuit) and time-varying sequential telemetry via context vectors ($c_s, c_e, c_h, c_c$).
- **Concerns & Limitations**:
  - Pure architecture representation; does not display internal weight values (which are detailed in Fig. 5).

---

## FIGURE 2 — SEQUENCE CONSTRUCTION & BOUNDARY ISOLATION

- **File Formats**: [`figures/fig2_sequence_construction.png`](file:///e:/paper/figures/fig2_sequence_construction.png) | [`figures/fig2_sequence_construction.pdf`](file:///e:/paper/figures/fig2_sequence_construction.pdf)
- **Caption**: *Fig. 2. Multi-horizon sliding-window sequence construction protocol. Sequences are strictly partitioned within each individual (Year, RoundNumber, Driver) competition trajectory. An encoder window of $L=20$ historical laps predicts $H=5$ consecutive future laps without crossing driver, Grand Prix, or season boundaries. Sequences with incomplete future horizons at race conclusion are discarded to prevent padding artifacts.*
- **Source Code & Data**: [`src/sequence_construction.py`](file:///e:/paper/src/sequence_construction.py#L167-L259).
- **What It Demonstrates**:
  - How sliding windows move chronologically through each driver's race stint.
  - Enforces the anti-leakage guarantee: zero information bleeding across drivers or distinct Grand Prix events.
  - Zero zero-padding policy: incomplete tail windows are discarded rather than synthetically padded.
- **Concerns & Limitations**:
  - Conceptual schematic representing the algorithmic data extraction rule.

---

## FIGURE 3 — OVERALL MODEL COMPARISON (MAE)

- **File Formats**: [`figures/fig3_model_comparison_mae.png`](file:///e:/paper/figures/fig3_model_comparison_mae.png) | [`figures/fig3_model_comparison_mae.pdf`](file:///e:/paper/figures/fig3_model_comparison_mae.pdf)
- **Caption**: *Fig. 3. Overall Mean Absolute Error (MAE in seconds) across 15,425 multi-step test sequences (77,125 forecast instances) from the 2024 Formula 1 World Championship. The Temporal Fusion Transformer (TFT) achieves an overall MAE of 3.0957 s, outperforming the Persistence baseline (3.4899 s) by 11.3% and the recurrent LSTM baseline (3.7860 s) by 18.2%.*
- **Source Code & Data**: [`results/baseline_comparison.csv`](file:///e:/paper/results/baseline_comparison.csv), [`results/tft_metrics.csv`](file:///e:/paper/results/tft_metrics.csv).
- **Exact Values Plotted**:
  - TFT: **3.0957 s**
  - Persistence: **3.4899 s**
  - LSTM: **3.7860 s**
  - Linear Regression: **7.6437 s**
  - Random Forest: **8.9038 s**
- **What It Demonstrates**:
  - Clear ranking on the primary optimization metric (MAE).
  - Highlights the substantial gap between deep temporal models (TFT, LSTM, Persistence) and static tabular baselines (Linear Regression, Random Forest).
- **Concerns & Limitations**:
  - This figure explicitly plots **MAE only**. As established in the audit, LSTM achieves slightly lower RMSE (8.3401 s vs. 8.4769 s), which is appropriately documented in Table IV and the text.

---

## FIGURE 4 — HORIZON-WISE MAE COMPARISON (H1 TO H5)

- **File Formats**: [`figures/fig4_horizon_mae.png`](file:///e:/paper/figures/fig4_horizon_mae.png) | [`figures/fig4_horizon_mae.pdf`](file:///e:/paper/figures/fig4_horizon_mae.pdf)
- **Caption**: *Fig. 4. Step-by-step Mean Absolute Error (MAE) trajectory across forecasting horizons $t+1$ (H1) through $t+5$ (H5). While the naive Persistence baseline achieves lower error at Horizon 1 (2.4724 s) due to immediate lap-to-lap inertia, its error degrades by +72.3% by Horizon 5. In contrast, the TFT model dominates from Horizons 2 through 5 and displays remarkable horizon stability ($\Delta_{H1 \to H5} = -0.20\%$, maintaining $\approx 3.07 - 3.13\text{ s}$).*
- **Source Code & Data**: [`results/baseline_comparison.csv`](file:///e:/paper/results/baseline_comparison.csv), [`results/tft_metrics.csv`](file:///e:/paper/results/tft_metrics.csv).
- **Exact Values Plotted**:
  - Persistence: `[2.4724, 3.2003, 3.5742, 3.9418, 4.2610]`
  - LSTM: `[3.4530, 3.6791, 3.8140, 3.9263, 4.0576]`
  - Linear Regression: `[4.3025, 7.3115, 8.6177, 8.6901, 9.2968]`
  - Random Forest: `[3.6810, 6.4447, 9.8838, 12.1349, 12.3746]`
  - TFT: `[3.0916, 3.1311, 3.1028, 3.0677, 3.0855]`
- **What It Demonstrates**:
  - Persistence is superior at H1 ($t+1$).
  - TFT takes over from H2 to H5 ($t+2 \dots t+5$).
  - TFT exhibits unmatched horizon stability (flat error curve), unlike all other models whose error grows monotonically.
- **Concerns & Limitations**:
  - None; reflects exact empirical numbers verified in the adversarial audit.

---

## FIGURE 5 — TFT INTERPRETABILITY: TEMPORAL ATTENTION & VSN SATURATION

- **File Formats**: [`figures/fig5_tft_interpretability.png`](file:///e:/paper/figures/fig5_tft_interpretability.png) | [`figures/fig5_tft_interpretability.pdf`](file:///e:/paper/figures/fig5_tft_interpretability.pdf)
- **Caption**: *Fig. 5. Interpretability analysis of the trained Temporal Fusion Transformer. (a) Mean temporal multi-head attention weights across the 20 historical encoder steps, revealing peak attention at the most immediate historical lap ($t-0$, weight 0.079) alongside a long-term reference anchor at the sequence start ($t-19$, weight 0.066). (b) Variable Selection Network (VSN) weight distribution, illustrating softmax saturation where static context is dominated by Driver identity (weight $\approx 1.0$) and temporal dynamics are dominated by 5-lap rolling mean pace (weight $\approx 1.0$), with remaining features forming a numerical tail ($< 1.8 \times 10^{-6}$).*
- **Source Code & Data**: [`results/attention_by_lag.csv`](file:///e:/paper/results/attention_by_lag.csv), [`results/feature_importance.csv`](file:///e:/paper/results/feature_importance.csv), [`src/interpretability.py`](file:///e:/paper/src/interpretability.py).
- **Exact Values Plotted**:
  - Attention weights: `t-0`: 0.0794, `t-1`: 0.0599, `t-2`: 0.0538, `t-3`: 0.0503 $\dots$ `t-19`: 0.0663.
  - VSN weights: Static `Driver`: 1.000, Temporal `RollingLapTimeMean_5`: 0.999999, `SpeedST`: 1.77e-6, `WeatherTime`: 3.01e-7, Tail sum: 2.92e-6.
- **What It Demonstrates**:
  - Honest presentation of the multi-head attention mechanism showing non-trivial temporal dynamics (U-shaped attention curve).
  - Explicit and transparent visualization of softmax saturation in the VSN, preventing false fine-grained ranking claims.
- **Concerns & Limitations**:
  - Acknowledges that VSN weights are saturated rather than broadly distributed across all 52 features.
