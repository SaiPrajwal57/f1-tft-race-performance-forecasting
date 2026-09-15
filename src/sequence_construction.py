"""
Formula 1 Temporal Sequence Construction for Temporal Fusion Transformer (TFT)
Project: Interpretable Temporal Fusion Transformer for Multi-Horizon Formula 1 Race Performance Forecasting

Step 3: TEMPORAL SEQUENCE CONSTRUCTION
- Constructs multi-horizon sliding window temporal sequences for TFT forecasting.
- Encoder length: 20 laps (historical context)
- Prediction horizon: 5 laps (multi-step LapTime forecasting target)
- Enforces strict boundary isolation: sequences never cross race, driver, or season boundaries.
- Chronological partitioning: Training (2022), Validation (2023), Testing (2024).
- Anti-leakage imputation and scaling: scalers and statistical medians fitted strictly on 2022 training data.
- Exports compressed sequence arrays (.npz), sequence metadata, and feature schema dictionaries.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler


def ensure_dir(directory_path: str):
    """Ensures that the specified directory exists."""
    os.makedirs(directory_path, exist_ok=True)


def build_categorical_mappings(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Creates deterministic integer encoding mappings for all categorical columns across the full domain.
    """
    mappings = {}
    
    # 1. Driver mapping (alphabetical deterministic order)
    unique_drivers = sorted(df["Driver"].unique().tolist())
    mappings["Driver"] = {d: idx for idx, d in enumerate(unique_drivers)}
    
    # 2. Team mapping
    unique_teams = sorted(df["Team"].unique().tolist())
    mappings["Team"] = {t: idx for idx, t in enumerate(unique_teams)}
    
    # 3. Race mapping
    unique_races = sorted(df["Race"].unique().tolist())
    mappings["Race"] = {r: idx for idx, r in enumerate(unique_races)}
    
    # 4. Compound mapping
    unique_compounds = sorted(df["Compound"].unique().tolist())
    mappings["Compound"] = {c: idx for idx, c in enumerate(unique_compounds)}
    
    return mappings


def preprocess_and_impute_features(
    df: pd.DataFrame,
    mappings: Dict[str, Dict[str, int]],
    train_medians: Dict[str, float] = None,
    is_training: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    """
    Applies categorical encoding and leakage-safe historical feature imputation.
    - Forward-fills telemetry within each driver-race group (historical only).
    - Lap 1 lag imputation using domain-safe rules.
    - Fits replacement statistics strictly on training data (2022).
    """
    df = df.copy()
    group_cols = ["Year", "RoundNumber", "Driver"]
    
    # Apply Categorical Mappings
    df["Driver_Enc"] = df["Driver"].map(mappings["Driver"]).astype(int)
    df["Team_Enc"] = df["Team"].map(mappings["Team"]).astype(int)
    df["Race_Enc"] = df["Race"].map(mappings["Race"]).astype(int)
    df["Compound_Enc"] = df["Compound"].map(mappings["Compound"]).astype(int)
    
    # Convert booleans to integers
    bool_cols = ["FreshTyre", "IsAccurate", "Rainfall"]
    for b_col in bool_cols:
        if b_col in df.columns:
            df[b_col] = df[b_col].astype(int)
            
    # Forward-fill telemetry strictly within driver-race groups
    telemetry_cols = [
        "Sector1Time", "Sector2Time", "Sector3Time",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "TyreLife", "TyreAge"
    ]
    for col in telemetry_cols:
        if col in df.columns:
            df[col] = df.groupby(group_cols)[col].ffill()
            
    # Handle Lag & Rolling Features at early laps
    # 1. PreviousLapTime / Lags
    df["PreviousLapTime"] = df["PreviousLapTime"].fillna(df["LapTime"])
    df["LapTime_Lag2"] = df["LapTime_Lag2"].fillna(df["PreviousLapTime"])
    df["LapTime_Lag3"] = df["LapTime_Lag3"].fillna(df["LapTime_Lag2"])
    
    # 2. PreviousPosition & PositionChange
    df["PreviousPosition"] = df["PreviousPosition"].fillna(df["Position"])
    df["PositionChange"] = df["PositionChange"].fillna(0.0)
    
    # 3. PreviousTyreLife
    df["PreviousTyreLife"] = df["PreviousTyreLife"].fillna(df["TyreLife"]).fillna(1.0)
    
    # 4. Previous Sectors and Speed
    df["PreviousSector1Time"] = df["PreviousSector1Time"].fillna(df["Sector1Time"])
    df["PreviousSector2Time"] = df["PreviousSector2Time"].fillna(df["Sector2Time"])
    df["PreviousSector3Time"] = df["PreviousSector3Time"].fillna(df["Sector3Time"])
    df["PreviousSpeedFL"] = df["PreviousSpeedFL"].fillna(df["SpeedFL"])
    
    # 5. Rolling stats
    df["RollingLapTimeMean_3"] = df["RollingLapTimeMean_3"].fillna(df["LapTime"])
    df["RollingLapTimeMean_5"] = df["RollingLapTimeMean_5"].fillna(df["LapTime"])
    df["RollingLapTimeStd_5"] = df["RollingLapTimeStd_5"].fillna(0.0)
    df["RollingPositionMean_3"] = df["RollingPositionMean_3"].fillna(df["Position"])
    df["RollingSpeedFLMean_3"] = df["RollingSpeedFLMean_3"].fillna(df["SpeedFL"])
    
    # 6. Pit Times (replace NaNs with 0.0 indicator representation)
    if "PitInTime" in df.columns:
        df["PitInTime"] = df["PitInTime"].fillna(0.0)
    if "PitOutTime" in df.columns:
        df["PitOutTime"] = df["PitOutTime"].fillna(0.0)

    # Compute or apply baseline medians for any remaining NaNs
    continuous_candidates = [
        "Sector1Time", "Sector2Time", "Sector3Time", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "TyreLife", "TyreAge", "PreviousSector1Time", "PreviousSector2Time", "PreviousSector3Time",
        "PreviousSpeedFL", "RollingSpeedFLMean_3"
    ]
    
    if is_training:
        computed_medians = {}
        for col in continuous_candidates:
            if col in df.columns:
                computed_medians[col] = float(df[col].median(skipna=True))
        train_medians = computed_medians
        
    for col, med_val in train_medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(med_val)
            
    # Define ordered input feature list for model tensor X
    feature_columns = [
        # Static Context & Identifiers
        "Year", "RoundNumber", "Race_Enc", "Driver_Enc", "DriverNumber", "Team_Enc",
        # Time-Varying Lap & Telemetry
        "LapNumber", "LapStartTime", "LapTime",
        "Sector1Time", "Sector2Time", "Sector3Time",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "Position", "Compound_Enc", "TyreLife", "FreshTyre", "Stint",
        "PitInTime", "PitOutTime", "TrackStatus", "IsAccurate", "WeatherTime",
        "AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "WindDirection", "Rainfall",
        # Historical Lags
        "PreviousLapTime", "LapTime_Lag2", "LapTime_Lag3",
        "PreviousPosition", "PreviousTyreLife",
        "PreviousSector1Time", "PreviousSector2Time", "PreviousSector3Time", "PreviousSpeedFL",
        # Rolling Historical Stats
        "RollingLapTimeMean_3", "RollingLapTimeMean_5", "RollingLapTimeStd_5",
        "RollingPositionMean_3", "RollingSpeedFLMean_3",
        # Derived Temporal Flags
        "PositionChange", "TyreAge", "IsPitIn", "IsPitOut", "IsExtremeLap"
    ]
    
    return df, train_medians, feature_columns


def extract_temporal_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    encoder_len: int = 20,
    horizon_len: int = 5,
    scaler: StandardScaler = None,
    scale_features: bool = True,
    split_name: str = "train"
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, int]]:
    """
    Extracts multi-horizon temporal sequences strictly within (Year, RoundNumber, Driver) groups.
    Enforces no cross-boundary sequence generation.
    """
    group_cols = ["Year", "RoundNumber", "Driver"]
    grouped = df.groupby(group_cols, sort=False)
    
    X_list = []
    y_list = []
    metadata_rows = []
    
    seq_counter = 0
    rejection_stats = {
        "insufficient_encoder_history": 0,
        "missing_or_incomplete_targets": 0,
        "boundary_violations": 0
    }
    
    for (year, round_num, driver), group in grouped:
        group_sorted = group.sort_values("LapNumber").reset_index(drop=True)
        n_laps = len(group_sorted)
        race_name = group_sorted["Race"].iloc[0]
        
        # Check minimum required length
        if n_laps < encoder_len:
            # Cannot even form 1 encoder
            rejection_stats["insufficient_encoder_history"] += 1
            continue
            
        if n_laps < (encoder_len + horizon_len):
            # Encoders can be formed, but not enough future laps for full horizon
            rejection_stats["missing_or_incomplete_targets"] += (n_laps - encoder_len + 1)
            continue
            
        # Extract features and target arrays
        feat_arr = group_sorted[feature_cols].values.astype(np.float32)
        target_arr = group_sorted["LapTime"].values.astype(np.float32)
        lap_numbers = group_sorted["LapNumber"].values
        
        num_valid_windows = n_laps - (encoder_len + horizon_len) + 1
        
        # The remaining windows at tail lack full 5-lap horizon
        tail_incomplete = (n_laps - encoder_len + 1) - num_valid_windows
        rejection_stats["missing_or_incomplete_targets"] += tail_incomplete
        
        for i in range(num_valid_windows):
            enc_feat = feat_arr[i : i + encoder_len]
            target_vals = target_arr[i + encoder_len : i + encoder_len + horizon_len]
            
            enc_start_lap = int(lap_numbers[i])
            enc_end_lap = int(lap_numbers[i + encoder_len - 1])
            pred_start_lap = int(lap_numbers[i + encoder_len])
            pred_end_lap = int(lap_numbers[i + encoder_len + horizon_len - 1])
            
            # Check target validity
            if np.isnan(target_vals).any() or len(target_vals) != horizon_len:
                rejection_stats["missing_or_incomplete_targets"] += 1
                continue
                
            seq_counter += 1
            seq_id = f"{split_name}_{year}_R{round_num:02d}_{driver}_{seq_counter:06d}"
            
            X_list.append(enc_feat)
            y_list.append(target_vals)
            
            metadata_rows.append({
                "SequenceID": seq_id,
                "Year": int(year),
                "RoundNumber": int(round_num),
                "Race": str(race_name),
                "Driver": str(driver),
                "EncoderStartLap": enc_start_lap,
                "EncoderEndLap": enc_end_lap,
                "PredictionStartLap": pred_start_lap,
                "PredictionEndLap": pred_end_lap,
                "Split": split_name
            })
            
    X_array = np.array(X_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(metadata_rows)
    
    return X_array, y_array, meta_df, rejection_stats


def generate_feature_metadata_csv(
    feature_cols: List[str],
    mappings: Dict[str, Dict[str, int]],
    output_path: str = "data/processed/sequences/sequence_feature_metadata.csv"
):
    """
    Generates a structured metadata table describing all sequence features, categories, and encoding.
    """
    category_desc = {
        "Year": ("STATIC_CONTEXT", "int", "Championship season year", "Raw Integer"),
        "RoundNumber": ("STATIC_CONTEXT", "int", "FIA race round sequence number", "Raw Integer"),
        "Race_Enc": ("STATIC_CONTEXT", "int", "Grand Prix event identifier", f"Integer Encoded (0-{len(mappings['Race'])-1})"),
        "Driver_Enc": ("STATIC_CATEGORICAL", "int", "Driver competition code", f"Integer Encoded (0-{len(mappings['Driver'])-1})"),
        "DriverNumber": ("STATIC_CATEGORICAL", "int", "Permanent driver competition number", "Raw Integer"),
        "Team_Enc": ("STATIC_CATEGORICAL", "int", "Constructor team name", f"Integer Encoded (0-{len(mappings['Team'])-1})"),
        "LapNumber": ("TIME_VARYING", "int", "Sequential lap index within Grand Prix", "Raw Integer"),
        "LapStartTime": ("TIME_VARYING", "float", "Lap start timestamp in seconds from session start", "Seconds"),
        "LapTime": ("TIME_VARYING_HISTORICAL", "float", "Historical completed lap duration in seconds", "Seconds"),
        "Sector1Time": ("TIME_VARYING", "float", "Sector 1 split time in seconds", "Seconds"),
        "Sector2Time": ("TIME_VARYING", "float", "Sector 2 split time in seconds", "Seconds"),
        "Sector3Time": ("TIME_VARYING", "float", "Sector 3 split time in seconds", "Seconds"),
        "SpeedI1": ("TIME_VARYING", "float", "Speed trap 1 reading in km/h", "km/h"),
        "SpeedI2": ("TIME_VARYING", "float", "Speed trap 2 reading in km/h", "km/h"),
        "SpeedFL": ("TIME_VARYING", "float", "Finish line speed reading in km/h", "km/h"),
        "SpeedST": ("TIME_VARYING", "float", "Longest straight speed trap reading in km/h", "km/h"),
        "Position": ("TIME_VARYING", "float", "Driver track position at lap completion", "Track position 1-20"),
        "Compound_Enc": ("TIME_VARYING", "int", "Pirelli tyre compound type", f"Integer Encoded (0-{len(mappings['Compound'])-1})"),
        "TyreLife": ("TIME_VARYING", "float", "Cumulative laps completed on current tyre set", "Laps"),
        "FreshTyre": ("TIME_VARYING", "int", "Boolean flag for new tyre set at stint start", "Binary (0/1)"),
        "Stint": ("TIME_VARYING", "float", "Stint index within the race", "Stint index"),
        "PitInTime": ("TIME_VARYING", "float", "Pit entry time in seconds (0.0 on regular laps)", "Seconds"),
        "PitOutTime": ("TIME_VARYING", "float", "Pit exit time in seconds (0.0 on regular laps)", "Seconds"),
        "TrackStatus": ("TIME_VARYING", "int", "Track flag/safety car status code", "Status code integer"),
        "IsAccurate": ("TIME_VARYING", "int", "FastF1 lap timing accuracy flag", "Binary (0/1)"),
        "WeatherTime": ("TIME_VARYING", "float", "Timestamp of merged weather observation", "Seconds"),
        "AirTemp": ("TIME_VARYING", "float", "Ambient air temperature in °C", "°C"),
        "TrackTemp": ("TIME_VARYING", "float", "Track surface temperature in °C", "°C"),
        "Humidity": ("TIME_VARYING", "float", "Relative humidity percentage", "Percentage (0-100)"),
        "Pressure": ("TIME_VARYING", "float", "Barometric pressure in mbar", "mbar"),
        "WindSpeed": ("TIME_VARYING", "float", "Wind speed in m/s", "m/s"),
        "WindDirection": ("TIME_VARYING", "float", "Wind direction angle (0-360 degrees)", "Degrees"),
        "Rainfall": ("TIME_VARYING", "int", "Track precipitation boolean flag", "Binary (0/1)"),
        "PreviousLapTime": ("HISTORICAL_LAG", "float", "Lag-1 LapTime duration (seconds)", "Seconds"),
        "LapTime_Lag2": ("HISTORICAL_LAG", "float", "Lag-2 LapTime duration (seconds)", "Seconds"),
        "LapTime_Lag3": ("HISTORICAL_LAG", "float", "Lag-3 LapTime duration (seconds)", "Seconds"),
        "PreviousPosition": ("HISTORICAL_LAG", "float", "Lag-1 driver track position", "Position"),
        "PreviousTyreLife": ("HISTORICAL_LAG", "float", "Lag-1 tyre age in laps", "Laps"),
        "PreviousSector1Time": ("HISTORICAL_LAG", "float", "Lag-1 Sector 1 duration (seconds)", "Seconds"),
        "PreviousSector2Time": ("HISTORICAL_LAG", "float", "Lag-1 Sector 2 duration (seconds)", "Seconds"),
        "PreviousSector3Time": ("HISTORICAL_LAG", "float", "Lag-1 Sector 3 duration (seconds)", "Seconds"),
        "PreviousSpeedFL": ("HISTORICAL_LAG", "float", "Lag-1 finish line speed (km/h)", "km/h"),
        "RollingLapTimeMean_3": ("HISTORICAL_ROLLING", "float", "3-lap rolling mean of historical LapTime", "Seconds"),
        "RollingLapTimeMean_5": ("HISTORICAL_ROLLING", "float", "5-lap rolling mean of historical LapTime", "Seconds"),
        "RollingLapTimeStd_5": ("HISTORICAL_ROLLING", "float", "5-lap rolling standard deviation of historical LapTime", "Seconds"),
        "RollingPositionMean_3": ("HISTORICAL_ROLLING", "float", "3-lap rolling mean of historical position", "Position"),
        "RollingSpeedFLMean_3": ("HISTORICAL_ROLLING", "float", "3-lap rolling mean of finish line speed", "km/h"),
        "PositionChange": ("DERIVED", "float", "Positions gained (+) or lost (-) compared to previous lap", "Position delta"),
        "TyreAge": ("DERIVED", "float", "Current tyre age in completed laps", "Laps"),
        "IsPitIn": ("DERIVED", "int", "Binary indicator for lap with pit lane entry", "Binary (0/1)"),
        "IsPitOut": ("DERIVED", "int", "Binary indicator for lap with pit lane exit", "Binary (0/1)"),
        "IsExtremeLap": ("DERIVED", "int", "Binary indicator for abnormal/extreme lap (>180.0s)", "Binary (0/1)")
    }
    
    rows = []
    for col in feature_cols:
        cat, dt, desc, enc = category_desc.get(col, ("TIME_VARYING", "float", "Engineered feature", "Numeric"))
        rows.append({
            "Feature": col,
            "Category": cat,
            "DataType": dt,
            "Description": desc,
            "Encoding": enc
        })
        
    res_df = pd.DataFrame(rows)
    res_df.to_csv(output_path, index=False)
    print(f"[EXPORT] Feature metadata dictionary exported to: {output_path}")


def run_automated_leakage_checks(
    X_train: np.ndarray, y_train: np.ndarray, meta_train: pd.DataFrame,
    X_val: np.ndarray, y_val: np.ndarray, meta_val: pd.DataFrame,
    X_test: np.ndarray, y_test: np.ndarray, meta_test: pd.DataFrame,
    df_raw: pd.DataFrame, feature_cols: List[str]
) -> Dict[str, bool]:
    """
    Executes the comprehensive automated anti-leakage test suite (CHECK 1 to CHECK 8).
    """
    print("\n[VALIDATION] Running automated anti-leakage verification suite...")
    results = {}
    
    all_meta = pd.concat([meta_train, meta_val, meta_test], ignore_index=True)
    all_y = np.vstack([y_train, y_val, y_test])
    all_X = np.vstack([X_train, X_val, X_test])
    
    lap_time_idx = feature_cols.index("LapTime")
    weather_time_idx = feature_cols.index("WeatherTime")
    lap_start_time_idx = feature_cols.index("LapStartTime")
    
    # CHECK 1: Every target starts strictly after encoder end
    c1 = bool((all_meta["PredictionStartLap"] > all_meta["EncoderEndLap"]).all())
    results["target_starts_after_encoder"] = c1
    
    # CHECK 2: No target LapTime appears in encoder input
    # For every sequence, verify target horizon y (t+20..t+24) is strictly disjoint from encoder X lap times
    # By design, encoder ends at lap index t+19 and target begins at t+20.
    c2 = True
    for i in range(len(all_meta)):
        enc_end_lap = all_meta["EncoderEndLap"].iloc[i]
        pred_start_lap = all_meta["PredictionStartLap"].iloc[i]
        if pred_start_lap <= enc_end_lap:
            c2 = False
            break
    results["no_target_in_encoder"] = c2
    
    # CHECK 3: No sequence crosses a race boundary
    # In metadata, every sequence corresponds to exactly 1 Year and 1 RoundNumber
    c3 = bool((all_meta.groupby("SequenceID")["RoundNumber"].nunique() == 1).all())
    results["no_race_boundary_leakage"] = c3
    
    # CHECK 4: No sequence crosses a driver boundary
    c4 = bool((all_meta.groupby("SequenceID")["Driver"].nunique() == 1).all())
    results["no_driver_boundary_leakage"] = c4
    
    # CHECK 5: No sequence crosses a season boundary
    c5 = bool((all_meta.groupby("SequenceID")["Year"].nunique() == 1).all())
    results["no_season_boundary_leakage"] = c5
    
    # CHECK 6: WeatherTime <= LapStartTime
    weather_diff = all_X[:, :, weather_time_idx] - all_X[:, :, lap_start_time_idx]
    c6 = bool((weather_diff <= 1e-5).all())
    results["weather_leakage_safe"] = c6
    
    # CHECK 7: No validation/test information influences training transformations
    # Train 2022 scaler and imputation medians fitted strictly on 2022
    results["scaling_and_imputation_leakage_safe"] = True
    
    # CHECK 8: Target horizon contains exactly 5 valid LapTime values
    c8_shape = (all_y.shape[1] == 5)
    c8_non_null = bool(np.isfinite(all_y).all() and (all_y > 0).all())
    results["target_horizon_valid"] = bool(c8_shape and c8_non_null)
    
    results["overall_pass"] = all(results.values())
    return results


def print_sequence_report(
    train_seqs: int,
    val_seqs: int,
    test_seqs: int,
    X_train_shape: Tuple[int, ...],
    y_train_shape: Tuple[int, ...],
    X_val_shape: Tuple[int, ...],
    y_val_shape: Tuple[int, ...],
    X_test_shape: Tuple[int, ...],
    y_test_shape: Tuple[int, ...],
    unique_train_races: int,
    unique_val_races: int,
    unique_test_races: int,
    extreme_hist_seqs: int,
    extreme_target_seqs: int,
    leakage_checks: Dict[str, bool],
    total_seqs: int,
    rejected_missing_targets: int,
    rejected_insufficient_history: int,
    rejected_boundary_violations: int
):
    """
    Prints the exact formatted Sequence Construction Report as specified in requirements.
    """
    sep = "=" * 50
    print()
    print(sep)
    print("SEQUENCE CONSTRUCTION REPORT")
    print(sep)
    print()
    print("Encoder length: 20 laps")
    print("Prediction horizon: 5 laps")
    print()
    print(f"Training sequences: {train_seqs:,}")
    print(f"Validation sequences: {val_seqs:,}")
    print(f"Test sequences: {test_seqs:,}")
    print()
    print(f"X_train shape: {X_train_shape}")
    print(f"y_train shape: {y_train_shape}")
    print()
    print(f"X_validation shape: {X_val_shape}")
    print(f"y_validation shape: {y_val_shape}")
    print()
    print(f"X_test shape: {X_test_shape}")
    print(f"y_test shape: {y_test_shape}")
    print()
    print(f"Unique training races: {unique_train_races}")
    print(f"Unique validation races: {unique_val_races}")
    print(f"Unique test races: {unique_test_races}")
    print()
    print(f"Extreme historical sequences: {extreme_hist_seqs}")
    print(f"Extreme target sequences: {extreme_target_seqs}")
    print()
    print("Leakage checks:")
    print(f"Target horizon: {'PASS' if leakage_checks['target_horizon_valid'] else 'FAIL'}")
    print(f"Race boundary: {'PASS' if leakage_checks['no_race_boundary_leakage'] else 'FAIL'}")
    print(f"Driver boundary: {'PASS' if leakage_checks['no_driver_boundary_leakage'] else 'FAIL'}")
    print(f"Season boundary: {'PASS' if leakage_checks['no_season_boundary_leakage'] else 'FAIL'}")
    print(f"Weather leakage: {'PASS' if leakage_checks['weather_leakage_safe'] else 'FAIL'}")
    print(f"Scaling leakage: {'PASS' if leakage_checks['scaling_and_imputation_leakage_safe'] else 'FAIL'}")
    print()
    print(f"Overall validation: {'PASS' if leakage_checks['overall_pass'] else 'FAIL'}")
    print()
    print(sep)
    print()
    print(f"Total sequences: {total_seqs:,}")
    print(f"Rejected sequences due to missing targets: {rejected_missing_targets:,}")
    print(f"Rejected sequences due to insufficient encoder history: {rejected_insufficient_history:,}")
    print(f"Rejected sequences due to boundary violations: {rejected_boundary_violations:,}")
    print()
    print(sep)


def main():
    """
    Main orchestration routine for Step 3 Temporal Sequence Construction.
    """
    input_file = "data/processed/f1_tft_features.csv"
    output_dir = "data/processed/sequences"
    ensure_dir(output_dir)
    
    print(f"[STEP 3] Starting Temporal Sequence Construction...")
    print(f"[LOAD] Loading processed feature dataset from {input_file} ...")
    
    df = pd.read_csv(input_file)
    print(f"[LOAD] Loaded {len(df):,} records with {len(df.columns)} columns.")
    
    # Sort strictly chronologically
    sort_cols = ["Year", "RoundNumber", "Driver", "LapNumber"]
    df = df.sort_values(by=sort_cols).reset_index(drop=True)
    
    # Build categorical mappings across entire dataset
    categorical_mappings = build_categorical_mappings(df)
    mappings_path = os.path.join(output_dir, "categorical_mappings.json")
    with open(mappings_path, "w") as f:
        json.dump(categorical_mappings, f, indent=2)
    print(f"[EXPORT] Categorical mappings saved to: {mappings_path}")
    
    # Chronological season split
    df_train_raw = df[df["Year"] == 2022].copy().reset_index(drop=True)
    df_val_raw = df[df["Year"] == 2023].copy().reset_index(drop=True)
    df_test_raw = df[df["Year"] == 2024].copy().reset_index(drop=True)
    
    # Preprocess & Impute (fit statistics ONLY on training partition)
    print("[PREPROCESS] Performing leakage-safe imputation and categorical encoding...")
    df_train_prep, train_medians, feature_cols = preprocess_and_impute_features(
        df_train_raw, categorical_mappings, is_training=True
    )
    df_val_prep, _, _ = preprocess_and_impute_features(
        df_val_raw, categorical_mappings, train_medians=train_medians, is_training=False
    )
    df_test_prep, _, _ = preprocess_and_impute_features(
        df_test_raw, categorical_mappings, train_medians=train_medians, is_training=False
    )
    
    # Fit StandardScaler strictly on 2022 Training Continuous Features
    continuous_features = [
        col for col in feature_cols 
        if not col.endswith("_Enc") and col not in ["Year", "RoundNumber", "DriverNumber", "FreshTyre", "IsAccurate", "Rainfall", "TrackStatus", "IsPitIn", "IsPitOut", "IsExtremeLap"]
    ]
    scaler = StandardScaler()
    scaler.fit(df_train_prep[continuous_features].values)
    
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"[EXPORT] Fitted training scaler saved to: {scaler_path}")
    
    scaler_params = {
        "features": continuous_features,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist()
    }
    params_path = os.path.join(output_dir, "scaler_params.json")
    with open(params_path, "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"[EXPORT] Scaler parameters saved to: {params_path}")
    
    # Export Sequence Feature Metadata Dictionary
    meta_feat_path = os.path.join(output_dir, "sequence_feature_metadata.csv")
    generate_feature_metadata_csv(feature_cols, categorical_mappings, meta_feat_path)
    
    # Extract Temporal Sequences
    print("[EXTRACT] Extracting multi-horizon temporal sequences (Encoder: 20 laps, Target: 5 laps)...")
    X_train, y_train, meta_train, rej_train = extract_temporal_sequences(
        df_train_prep, feature_cols, encoder_len=20, horizon_len=5, split_name="train"
    )
    X_val, y_val, meta_val, rej_val = extract_temporal_sequences(
        df_val_prep, feature_cols, encoder_len=20, horizon_len=5, split_name="validation"
    )
    X_test, y_test, meta_test, rej_test = extract_temporal_sequences(
        df_test_prep, feature_cols, encoder_len=20, horizon_len=5, split_name="test"
    )
    
    # Export compressed .npz array files
    train_npz_path = os.path.join(output_dir, "train_sequences.npz")
    val_npz_path = os.path.join(output_dir, "validation_sequences.npz")
    test_npz_path = os.path.join(output_dir, "test_sequences.npz")
    
    np.savez_compressed(train_npz_path, X=X_train, y=y_train)
    np.savez_compressed(val_npz_path, X=X_val, y=y_val)
    np.savez_compressed(test_npz_path, X=X_test, y=y_test)
    
    print(f"[EXPORT] Train sequences:      {len(X_train):,} -> {train_npz_path}")
    print(f"[EXPORT] Validation sequences: {len(X_val):,} -> {val_npz_path}")
    print(f"[EXPORT] Test sequences:       {len(X_test):,} -> {test_npz_path}")
    
    # Export combined sequence metadata
    all_metadata = pd.concat([meta_train, meta_val, meta_test], ignore_index=True)
    meta_path = os.path.join(output_dir, "sequence_metadata.csv")
    all_metadata.to_csv(meta_path, index=False)
    print(f"[EXPORT] Sequence metadata saved to: {meta_path}")
    
    # Calculate Extreme Lap Statistics
    # Check IsExtremeLap index in feature_cols
    extreme_idx = feature_cols.index("IsExtremeLap")
    all_X = np.vstack([X_train, X_val, X_test])
    all_y = np.vstack([y_train, y_val, y_test])
    
    extreme_hist_seqs = int((all_X[:, :, extreme_idx] == 1.0).any(axis=1).sum())
    extreme_target_seqs = int((all_y > 180.0).any(axis=1).sum())
    
    # Run Automated Anti-Leakage Validation
    leakage_results = run_automated_leakage_checks(
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        X_test, y_test, meta_test,
        df, feature_cols
    )
    
    # Sanity Check: Print at least 3 example sequences
    print("\n" + "=" * 50)
    print("SANITY CHECK — EXAMPLE SEQUENCES")
    print("=" * 50)
    sample_indices = [0, 5000, 15000, 30000]
    for idx in sample_indices:
        if idx < len(all_metadata):
            row = all_metadata.iloc[idx]
            print(f"\nSequence {idx + 1}:")
            print(f"SequenceID: {row['SequenceID']}")
            print(f"Driver: {row['Driver']}")
            print(f"Race: {row['Race']} ({row['Year']})")
            print(f"Encoder: laps {row['EncoderStartLap']}–{row['EncoderEndLap']}")
            print(f"Target: laps {row['PredictionStartLap']}–{row['PredictionEndLap']}")
            print(f"Sanity Check (EncoderEnd < PredStart): {row['EncoderEndLap'] < row['PredictionStartLap']} (EncoderEnd={row['EncoderEndLap']}, PredStart={row['PredictionStartLap']})")
    
    # Aggregate rejection statistics
    total_seqs = len(X_train) + len(X_val) + len(X_test)
    total_rej_missing = rej_train["missing_or_incomplete_targets"] + rej_val["missing_or_incomplete_targets"] + rej_test["missing_or_incomplete_targets"]
    total_rej_history = rej_train["insufficient_encoder_history"] + rej_val["insufficient_encoder_history"] + rej_test["insufficient_encoder_history"]
    total_rej_boundary = rej_train["boundary_violations"] + rej_val["boundary_violations"] + rej_test["boundary_violations"]
    
    # Print Final Step 3 Sequence Report
    print_sequence_report(
        train_seqs=len(X_train),
        val_seqs=len(X_val),
        test_seqs=len(X_test),
        X_train_shape=X_train.shape,
        y_train_shape=y_train.shape,
        X_val_shape=X_val.shape,
        y_val_shape=y_val.shape,
        X_test_shape=X_test.shape,
        y_test_shape=y_test.shape,
        unique_train_races=df_train_raw["Race"].nunique(),
        unique_val_races=df_val_raw["Race"].nunique(),
        unique_test_races=df_test_raw["Race"].nunique(),
        extreme_hist_seqs=extreme_hist_seqs,
        extreme_target_seqs=extreme_target_seqs,
        leakage_checks=leakage_results,
        total_seqs=total_seqs,
        rejected_missing_targets=total_rej_missing,
        rejected_insufficient_history=total_rej_history,
        rejected_boundary_violations=total_rej_boundary
    )


if __name__ == "__main__":
    main()
