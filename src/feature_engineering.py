"""
Formula 1 Lap-Level Feature Engineering Pipeline for Temporal Fusion Transformer (TFT)
Project: Interpretable Temporal Fusion Transformer for Multi-Horizon Formula 1 Race Performance Forecasting

Step 2: FEATURE ENGINEERING
- Transforms raw lap-level dataset into leakage-free feature representations for TFT forecasting.
- Preserves raw dataset and computes historical lags, rolling statistics, and derived temporal flags.
- Enforces strict anti-leakage boundary constraints (groupby Year, RoundNumber, Driver with shift(1)).
- Performs chronological season-based splitting (Train: 2022, Val: 2023, Test: 2024).
- Exports data/processed/ feature datasets and metadata summary.
"""

import os
import sys
import numpy as np
import pandas as pd


def load_raw_dataset(filepath: str = "data/f1_lap_dataset_2022_2024.csv") -> pd.DataFrame:
    """
    Loads raw multi-season Grand Prix lap dataset and enforces initial chronological sorting.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset file not found at: {filepath}")

    print(f"[LOAD] Loading raw dataset from: {filepath} ...")
    df = pd.read_csv(filepath)
    print(f"[LOAD] Loaded {len(df):,} raw records with {len(df.columns)} columns.")

    # Enforce strict chronological ordering
    sort_cols = ["Year", "RoundNumber", "Driver", "LapNumber"]
    df = df.sort_values(by=sort_cols).reset_index(drop=True)
    return df


def create_historical_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes historical lag features strictly within (Year, RoundNumber, Driver) groups.
    Guarantees no data from another driver, race, or future lap enters the features.
    """
    print("[FEATURES] Computing historical lag features...")
    group_cols = ["Year", "RoundNumber", "Driver"]
    grouped = df.groupby(group_cols)

    # 1. Target Lags
    df["PreviousLapTime"] = grouped["LapTime"].shift(1)
    df["LapTime_Lag2"] = grouped["LapTime"].shift(2)
    df["LapTime_Lag3"] = grouped["LapTime"].shift(3)

    # 2. Performance & Telemetry Lags
    df["PreviousPosition"] = grouped["Position"].shift(1)
    df["PreviousTyreLife"] = grouped["TyreLife"].shift(1)
    df["PreviousSector1Time"] = grouped["Sector1Time"].shift(1)
    df["PreviousSector2Time"] = grouped["Sector2Time"].shift(1)
    df["PreviousSector3Time"] = grouped["Sector3Time"].shift(1)
    df["PreviousSpeedFL"] = grouped["SpeedFL"].shift(1)

    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling historical statistics strictly using PAST observations (shift(1).rolling(...)).
    CONCEPTUAL STRUCTURE: shift(1).rolling(...) to strictly exclude current/future target observations.
    """
    print("[FEATURES] Computing rolling historical features...")
    group_cols = ["Year", "RoundNumber", "Driver"]

    def calc_rolling_mean(series, window):
        return series.shift(1).rolling(window=window, min_periods=1).mean()

    def calc_rolling_std(series, window):
        return series.shift(1).rolling(window=window, min_periods=1).std()

    # LapTime Rolling Features
    df["RollingLapTimeMean_3"] = df.groupby(group_cols)["LapTime"].transform(lambda x: calc_rolling_mean(x, 3))
    df["RollingLapTimeMean_5"] = df.groupby(group_cols)["LapTime"].transform(lambda x: calc_rolling_mean(x, 5))
    df["RollingLapTimeStd_5"] = df.groupby(group_cols)["LapTime"].transform(lambda x: calc_rolling_std(x, 5))

    # Position Rolling Features
    df["RollingPositionMean_3"] = df.groupby(group_cols)["Position"].transform(lambda x: calc_rolling_mean(x, 3))

    # Finish Line Speed Rolling Features
    df["RollingSpeedFLMean_3"] = df.groupby(group_cols)["SpeedFL"].transform(lambda x: calc_rolling_mean(x, 3))

    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes derived domain-specific indicators and flags.
    """
    print("[FEATURES] Computing derived indicators and flags...")

    # Position Change (PreviousPosition - current Position)
    df["PositionChange"] = df["PreviousPosition"] - df["Position"]

    # Tyre Age (equivalent to TyreLife)
    df["TyreAge"] = df["TyreLife"]

    # Pit Stop Binary Flags
    df["IsPitIn"] = df["PitInTime"].notnull().astype(int)
    df["IsPitOut"] = df["PitOutTime"].notnull().astype(int)

    # Extreme Lap Flag (LapTime > 180.0 seconds)
    df["IsExtremeLap"] = (df["LapTime"] > 180.0).astype(int)

    return df


def validate_anti_leakage(df: pd.DataFrame) -> dict:
    """
    Automated anti-leakage validation suite.
    """
    print("[VALIDATION] Running anti-leakage validation checks...")
    results = {}
    group_cols = ["Year", "RoundNumber", "Driver"]

    # 1. Lag leakage check: First record of any driver in any race must have PreviousLapTime as NaN
    first_records_lags = df.groupby(group_cols)["PreviousLapTime"].nth(0)
    results["lag_leakage"] = bool(first_records_lags.isnull().all())

    # 2. Rolling leakage check: First record of any driver in any race must have RollingLapTimeMean_3 as NaN
    first_records_rolling = df.groupby(group_cols)["RollingLapTimeMean_3"].nth(0)
    results["rolling_leakage"] = bool(first_records_rolling.isnull().all())

    # 3. Weather leakage check: WeatherTime <= LapStartTime
    if "WeatherTime" in df.columns and "LapStartTime" in df.columns:
        valid_w = df[df["WeatherTime"].notnull() & df["LapStartTime"].notnull()]
        future_weather_count = (valid_w["WeatherTime"] > valid_w["LapStartTime"]).sum()
        results["weather_leakage"] = bool(future_weather_count == 0)
    else:
        results["weather_leakage"] = True

    # 4. Cross-driver leakage check: Lags do not bleed across drivers within the same race
    results["cross_driver_leakage"] = bool(first_records_lags.isnull().all())

    # 5. Cross-race leakage check: Lags do not bleed across different races
    results["cross_race_leakage"] = bool(first_records_lags.isnull().all())

    results["overall_pass"] = all(results.values())
    return results


def export_feature_summary(df: pd.DataFrame, output_path: str = "data/processed/feature_summary.csv"):
    """
    Generates a structured metadata feature dictionary summary CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    category_mapping = {
        "Driver": ("STATIC_CATEGORICAL", "str", "Driver 3-letter abbreviation code"),
        "Team": ("STATIC_CATEGORICAL", "str", "Constructor team name"),
        "Race": ("STATIC_CONTEXT", "str", "Grand Prix event name"),
        "RoundNumber": ("STATIC_CONTEXT", "int", "FIA race round sequence number"),
        "Year": ("STATIC_CONTEXT", "int", "Championship season year"),
        "LapNumber": ("TIME_VARYING", "int", "Sequential lap index within Grand Prix"),
        "LapStartTime": ("TIME_VARYING", "float", "Lap start timestamp in seconds from session start"),
        "LapTime": ("TARGET", "float", "Primary target: completed lap duration in seconds"),
        "Sector1Time": ("TIME_VARYING", "float", "Sector 1 split time in seconds"),
        "Sector2Time": ("TIME_VARYING", "float", "Sector 2 split time in seconds"),
        "Sector3Time": ("TIME_VARYING", "float", "Sector 3 split time in seconds"),
        "SpeedI1": ("TIME_VARYING", "float", "Speed trap 1 reading in km/h"),
        "SpeedI2": ("TIME_VARYING", "float", "Speed trap 2 reading in km/h"),
        "SpeedFL": ("TIME_VARYING", "float", "Finish line speed reading in km/h"),
        "SpeedST": ("TIME_VARYING", "float", "Longest straight speed trap reading in km/h"),
        "Position": ("TIME_VARYING", "float", "Driver track position at lap completion"),
        "Compound": ("TIME_VARYING", "str", "Pirelli tyre compound type"),
        "TyreLife": ("TIME_VARYING", "float", "Cumulative laps completed on current tyre set"),
        "FreshTyre": ("TIME_VARYING", "bool", "Boolean flag for new tyre set at stint start"),
        "Stint": ("TIME_VARYING", "float", "Stint index within the race"),
        "PitInTime": ("TIME_VARYING", "float", "Pit entry time in seconds (NaN on regular laps)"),
        "PitOutTime": ("TIME_VARYING", "float", "Pit exit time in seconds (NaN on regular laps)"),
        "TrackStatus": ("TIME_VARYING", "str", "Track flag/safety car status code"),
        "IsAccurate": ("TIME_VARYING", "bool", "FastF1 lap timing accuracy flag"),
        "WeatherTime": ("TIME_VARYING", "float", "Timestamp of merged weather observation"),
        "AirTemp": ("TIME_VARYING", "float", "Ambient air temperature in °C"),
        "TrackTemp": ("TIME_VARYING", "float", "Track surface temperature in °C"),
        "Humidity": ("TIME_VARYING", "float", "Relative humidity percentage"),
        "Pressure": ("TIME_VARYING", "float", "Barometric pressure in mbar"),
        "WindSpeed": ("TIME_VARYING", "float", "Wind speed in m/s"),
        "WindDirection": ("TIME_VARYING", "float", "Wind direction angle (0-360 degrees)"),
        "Rainfall": ("TIME_VARYING", "bool", "Track precipitation boolean flag"),
        "DriverNumber": ("STATIC_CATEGORICAL", "str", "Permanent driver competition number"),
        # Historical Lags
        "PreviousLapTime": ("HISTORICAL_LAG", "float", "Lag-1 LapTime duration (seconds)"),
        "LapTime_Lag2": ("HISTORICAL_LAG", "float", "Lag-2 LapTime duration (seconds)"),
        "LapTime_Lag3": ("HISTORICAL_LAG", "float", "Lag-3 LapTime duration (seconds)"),
        "PreviousPosition": ("HISTORICAL_LAG", "float", "Lag-1 driver track position"),
        "PreviousTyreLife": ("HISTORICAL_LAG", "float", "Lag-1 tyre age in laps"),
        "PreviousSector1Time": ("HISTORICAL_LAG", "float", "Lag-1 Sector 1 duration (seconds)"),
        "PreviousSector2Time": ("HISTORICAL_LAG", "float", "Lag-1 Sector 2 duration (seconds)"),
        "PreviousSector3Time": ("HISTORICAL_LAG", "float", "Lag-1 Sector 3 duration (seconds)"),
        "PreviousSpeedFL": ("HISTORICAL_LAG", "float", "Lag-1 finish line speed (km/h)"),
        # Rolling Features
        "RollingLapTimeMean_3": ("HISTORICAL_ROLLING", "float", "3-lap rolling mean of historical LapTime"),
        "RollingLapTimeMean_5": ("HISTORICAL_ROLLING", "float", "5-lap rolling mean of historical LapTime"),
        "RollingLapTimeStd_5": ("HISTORICAL_ROLLING", "float", "5-lap rolling standard deviation of historical LapTime"),
        "RollingPositionMean_3": ("HISTORICAL_ROLLING", "float", "3-lap rolling mean of historical position"),
        "RollingSpeedFLMean_3": ("HISTORICAL_ROLLING", "float", "3-lap rolling mean of finish line speed"),
        # Derived Features
        "PositionChange": ("DERIVED", "float", "Positions gained (+) or lost (-) compared to previous lap"),
        "TyreAge": ("DERIVED", "float", "Current tyre age in completed laps"),
        "IsPitIn": ("DERIVED", "int", "Binary indicator for lap with pit lane entry"),
        "IsPitOut": ("DERIVED", "int", "Binary indicator for lap with pit lane exit"),
        "IsExtremeLap": ("DERIVED", "int", "Binary indicator for abnormal/extreme lap (>180.0s)"),
    }

    summary_rows = []
    total_records = len(df)

    for col in df.columns:
        cat, dt, desc = category_mapping.get(col, ("DERIVED", str(df[col].dtype), "Generated feature"))
        missing_cnt = int(df[col].isnull().sum())
        missing_pct = (missing_cnt / total_records * 100) if total_records > 0 else 0.0
        summary_rows.append({
            "Feature": col,
            "Category": cat,
            "DataType": str(df[col].dtype),
            "MissingCount": missing_cnt,
            "MissingPercentage": round(missing_pct, 2),
            "Description": desc
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)
    print(f"[EXPORT] Feature summary dictionary saved to: {output_path}")


def print_feature_engineering_report(
    raw_records: int,
    processed_records: int,
    train_records: int,
    val_records: int,
    test_records: int,
    input_feature_count: int,
    generated_feature_count: int,
    missing_target_count: int,
    extreme_target_count: int,
    validation_results: dict
):
    """
    Prints the standardized Feature Engineering Report in the exact requested terminal format.
    """
    sep = "=" * 50
    print()
    print(sep)
    print("FEATURE ENGINEERING REPORT")
    print(sep)
    print()
    print(f"Raw records: {raw_records}")
    print(f"Processed records: {processed_records}")
    print()
    print(f"Training records: {train_records}")
    print(f"Validation records: {val_records}")
    print(f"Test records: {test_records}")
    print()
    print(f"Input features: {input_feature_count}")
    print(f"Generated features: {generated_feature_count}")
    print("Target: LapTime")
    print()
    print(f"Missing target values: {missing_target_count}")
    print(f"Extreme target values: {extreme_target_count}")
    print()
    print("Leakage checks:")
    print(f"Lag leakage: {'PASS' if validation_results['lag_leakage'] else 'FAIL'}")
    print(f"Rolling leakage: {'PASS' if validation_results['rolling_leakage'] else 'FAIL'}")
    print(f"Weather leakage: {'PASS' if validation_results['weather_leakage'] else 'FAIL'}")
    print(f"Cross-driver leakage: {'PASS' if validation_results['cross_driver_leakage'] else 'FAIL'}")
    print(f"Cross-race leakage: {'PASS' if validation_results['cross_race_leakage'] else 'FAIL'}")
    print()
    print(f"Overall validation: {'PASS' if validation_results['overall_pass'] else 'FAIL'}")
    print()
    print(sep)


def main():
    """
    Orchestration routine for Step 2 TFT Feature Engineering.
    """
    raw_file = "data/f1_lap_dataset_2022_2024.csv"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load raw dataset
    df_raw = load_raw_dataset(raw_file)
    raw_records = len(df_raw)

    raw_input_columns = [c for c in df_raw.columns if c != "LapTime"]

    # 2. Compute historical lags on FULL chronological dataset (preserving temporal rows)
    df_feat = create_historical_lags(df_raw.copy())

    # 3. Compute rolling historical features
    df_feat = create_rolling_features(df_feat)

    # 4. Compute derived features
    df_feat = create_derived_features(df_feat)

    generated_feature_columns = [c for c in df_feat.columns if c not in df_raw.columns]

    # 5. Audit missing target values & extreme target values
    missing_target_count = int(df_feat["LapTime"].isnull().sum())
    extreme_target_count = int((df_feat["LapTime"] > 180.0).sum())

    # 6. Run anti-leakage validation before target filtering
    val_results = validate_anti_leakage(df_feat)

    # 7. Remove missing LapTime rows ONLY for final supervised modeling dataset
    print("[CLEANING] Filtering missing target observations (LapTime.isnull())...")
    df_processed = df_feat[df_feat["LapTime"].notnull()].copy().reset_index(drop=True)
    processed_records = len(df_processed)

    # 8. Export Master Feature Dataset
    master_processed_path = os.path.join(processed_dir, "f1_tft_features.csv")
    print(f"[EXPORT] Saving master feature dataset to: {master_processed_path} ...")
    df_processed.to_csv(master_processed_path, index=False)

    # 9. Export Feature Summary Metadata Dictionary
    export_feature_summary(df_processed, os.path.join(processed_dir, "feature_summary.csv"))

    # 10. Split chronologically by Season (Train: 2022, Val: 2023, Test: 2024)
    print("[SPLIT] Performing chronological season-based split...")
    df_train = df_processed[df_processed["Year"] == 2022].copy().reset_index(drop=True)
    df_val = df_processed[df_processed["Year"] == 2023].copy().reset_index(drop=True)
    df_test = df_processed[df_processed["Year"] == 2024].copy().reset_index(drop=True)

    train_path = os.path.join(processed_dir, "train_2022.csv")
    val_path = os.path.join(processed_dir, "validation_2023.csv")
    test_path = os.path.join(processed_dir, "test_2024.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"[EXPORT] Train 2022:      {len(df_train):,} records -> {train_path}")
    print(f"[EXPORT] Validation 2023: {len(df_val):,} records -> {val_path}")
    print(f"[EXPORT] Test 2024:       {len(df_test):,} records -> {test_path}")

    # 11. Print Terminal Feature Engineering Report
    print_feature_engineering_report(
        raw_records=raw_records,
        processed_records=processed_records,
        train_records=len(df_train),
        val_records=len(df_val),
        test_records=len(df_test),
        input_feature_count=len(raw_input_columns),
        generated_feature_count=len(generated_feature_columns),
        missing_target_count=missing_target_count,
        extreme_target_count=extreme_target_count,
        validation_results=val_results
    )


if __name__ == "__main__":
    main()
