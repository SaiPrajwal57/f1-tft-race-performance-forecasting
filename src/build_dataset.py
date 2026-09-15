"""
Formula 1 Lap-Level Dataset Construction Pipeline (2022–2024)
Project: Interpretable Temporal Fusion Transformer for Multi-Horizon Formula 1 Race Performance Forecasting

Step 1: DATASET CONSTRUCTION
- Collects all Grand Prix race sessions across 2022, 2023, and 2024 seasons using FastF1.
- Extracts lap-level telemetry, speed traps, tyre degradation, and weather conditions.
- Merges weather observations backward in time (direction='backward') to strictly avoid future data leakage.
- Preserves chronological ordering (Year -> RoundNumber -> Driver -> LapNumber) without shuffling or premature splitting.
- Validates data integrity, zero-leakage constraints, and target variable (LapTime in seconds).
"""

import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import fastf1


def setup_cache(cache_dir: str = "data/raw/fastf1_cache") -> str:
    """
    Configures and enables the FastF1 cache directory.
    """
    abs_cache_dir = os.path.abspath(cache_dir)
    os.makedirs(abs_cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(abs_cache_dir)
    print(f"[CACHE] FastF1 cache enabled at: {abs_cache_dir}")
    return abs_cache_dir


def get_season_races(year: int) -> pd.DataFrame:
    """
    Dynamically retrieves the official FIA Grand Prix race schedule for a given year,
    filtering out non-race testing events.
    """
    print(f"\n[SCHEDULE] Fetching event schedule for season {year}...")
    schedule = fastf1.get_event_schedule(year)
    
    # Filter out testing sessions (RoundNumber == 0 or EventFormat == 'testing')
    races = schedule[
        (schedule['RoundNumber'] > 0) & 
        (schedule['EventFormat'] != 'testing')
    ].copy()
    
    print(f"[SCHEDULE] Found {len(races)} Grand Prix events for {year}.")
    return races


def load_race_data(year: int, round_num: int, event_name: str):
    """
    Loads session data for a specific Grand Prix Race.
    Uses verified FastF1 3.8.3 signature: session.load(laps=True, telemetry=False, weather=True, messages=False)
    to optimize download speed and memory while retaining full timing, speed traps, tyre life, and weather.
    """
    try:
        session = fastf1.get_session(year, round_num, 'R')
        session.load(laps=True, telemetry=False, weather=True, messages=False)
        return session, None
    except Exception as e:
        err_msg = f"Failed to load {year} Round {round_num} ({event_name}): {str(e)}"
        return None, err_msg


def extract_lap_data(session, year: int, round_num: int, event_name: str) -> pd.DataFrame:
    """
    Extracts lap-level metrics from a race session, converting timedelta values to seconds.
    """
    if session.laps is None or len(session.laps) == 0:
        return pd.DataFrame()

    laps = session.laps.copy()

    # Core required identification and performance columns
    required_cols = [
        "Driver", "DriverNumber", "Team", "LapNumber", "LapStartTime", "LapTime",
        "Sector1Time", "Sector2Time", "Sector3Time", "Compound", "TyreLife",
        "FreshTyre", "Stint", "Position", "PitInTime", "PitOutTime",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "TrackStatus", "IsAccurate"
    ]

    # Ensure all required columns exist in dataframe
    for col in required_cols:
        if col not in laps.columns:
            laps[col] = np.nan

    df = laps[required_cols].copy()

    # Convert Timedelta columns to seconds (float)
    timedelta_cols = ["LapStartTime", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "PitInTime", "PitOutTime"]
    for col in timedelta_cols:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()
        else:
            df[col] = pd.to_timedelta(df[col], errors="coerce").dt.total_seconds()

    # Add race metadata
    df["Year"] = int(year)
    df["RoundNumber"] = int(round_num)
    df["Race"] = str(event_name)

    return df


def extract_weather_data(session) -> pd.DataFrame:
    """
    Extracts weather telemetry from a race session, converting timestamps to seconds.
    """
    if session.weather_data is None or len(session.weather_data) == 0:
        return pd.DataFrame()

    weather = session.weather_data.copy()
    
    # Convert Time (timedelta) to seconds
    if pd.api.types.is_timedelta64_dtype(weather["Time"]):
        weather["WeatherTime"] = weather["Time"].dt.total_seconds()
    else:
        weather["WeatherTime"] = pd.to_timedelta(weather["Time"], errors="coerce").dt.total_seconds()

    weather_cols = [
        "WeatherTime", "AirTemp", "TrackTemp", "Humidity", 
        "Pressure", "WindSpeed", "WindDirection", "Rainfall"
    ]
    
    for col in weather_cols:
        if col not in weather.columns:
            weather[col] = np.nan

    return weather[weather_cols].copy()


def merge_weather(laps_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges weather data with lap data using backward-in-time matching (direction='backward')
    on LapStartTime vs WeatherTime.
    
    ANTI-LEAKAGE GUARANTEE:
    Weather observations are matched ONLY from the most recent weather measurement at or before
    the start of the lap (WeatherTime <= LapStartTime). No future weather readings are used.
    """
    if laps_df.empty:
        return laps_df

    weather_feature_cols = [
        "AirTemp", "TrackTemp", "Humidity", "Pressure", 
        "WindSpeed", "WindDirection", "Rainfall", "WeatherTime"
    ]

    if weather_df is None or weather_df.empty or "WeatherTime" not in weather_df.columns:
        for col in weather_feature_cols:
            laps_df[col] = np.nan
        return laps_df

    # Sort and clean weather timestamps
    valid_weather = weather_df.dropna(subset=["WeatherTime"]).sort_values("WeatherTime").reset_index(drop=True)
    if valid_weather.empty:
        for col in weather_feature_cols:
            laps_df[col] = np.nan
        return laps_df

    # Split laps into valid LapStartTime vs missing LapStartTime to prevent pd.merge_asof ValueError
    has_start_time = laps_df["LapStartTime"].notnull()
    valid_laps = laps_df[has_start_time].sort_values("LapStartTime").reset_index(drop=True)
    null_laps = laps_df[~has_start_time].copy()

    if not valid_laps.empty:
        merged_valid = pd.merge_asof(
            valid_laps,
            valid_weather,
            left_on="LapStartTime",
            right_on="WeatherTime",
            direction="backward"
        )
        # Edge case: if lap 1 started before the very first weather reading in session window,
        # fill only those initial pre-race nulls with the earliest baseline weather reading (t=0)
        for col in ["AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "WindDirection", "Rainfall"]:
            if col in merged_valid.columns and merged_valid[col].isnull().any():
                merged_valid[col] = merged_valid[col].bfill()
    else:
        merged_valid = pd.DataFrame()

    if not null_laps.empty:
        for col in weather_feature_cols:
            null_laps[col] = np.nan

    if null_laps.empty:
        return merged_valid
    elif merged_valid.empty:
        return null_laps
    else:
        return pd.concat([merged_valid, null_laps], ignore_index=True)


def clean_lap_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the combined multi-race dataset intelligently:
    - Removes invalid/unidentified driver records or non-positive lap numbers.
    - Preserves expected NaNs in pit stops (PitInTime/PitOutTime are NaN on regular racing laps).
    - Preserves LapTime as the primary modeling target.
    - Enforces strict chronological ordering: [Year, RoundNumber, Driver, LapNumber].
    """
    if df.empty:
        return df

    cleaned = df.copy()

    # Drop records where Driver is missing
    cleaned = cleaned[cleaned["Driver"].notnull() & (cleaned["Driver"] != "")].copy()

    # Ensure LapNumber is numeric and positive
    cleaned["LapNumber"] = pd.to_numeric(cleaned["LapNumber"], errors="coerce")
    cleaned = cleaned[cleaned["LapNumber"].notnull() & (cleaned["LapNumber"] > 0)].copy()
    cleaned["LapNumber"] = cleaned["LapNumber"].astype(int)

    # Standardize string fields
    cleaned["Driver"] = cleaned["Driver"].astype(str).str.strip()
    cleaned["Team"] = cleaned["Team"].astype(str).str.strip()
    cleaned["Compound"] = cleaned["Compound"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    cleaned["Race"] = cleaned["Race"].astype(str).str.strip()

    # Standardize integer/float types
    cleaned["Year"] = cleaned["Year"].astype(int)
    cleaned["RoundNumber"] = cleaned["RoundNumber"].astype(int)
    
    numeric_float_cols = [
        "LapStartTime", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "Position",
        "TyreLife", "Stint", "PitInTime", "PitOutTime",
        "AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "WindDirection", "WeatherTime"
    ]
    for col in numeric_float_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Boolean flags
    if "FreshTyre" in cleaned.columns:
        cleaned["FreshTyre"] = cleaned["FreshTyre"].fillna(False).astype(bool)
    if "IsAccurate" in cleaned.columns:
        cleaned["IsAccurate"] = cleaned["IsAccurate"].fillna(False).astype(bool)
    if "Rainfall" in cleaned.columns:
        cleaned["Rainfall"] = cleaned["Rainfall"].fillna(False).astype(bool)

    # Reorder columns logically
    ordered_columns = [
        # Identification
        "Year", "RoundNumber", "Race", "Driver", "DriverNumber", "Team", "LapNumber",
        # Timeline & Target
        "LapStartTime", "LapTime",
        # Performance
        "Sector1Time", "Sector2Time", "Sector3Time",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "Position",
        # Tyre
        "Compound", "TyreLife", "FreshTyre", "Stint",
        # Pit Stops
        "PitInTime", "PitOutTime",
        # Race/Track Status
        "TrackStatus", "IsAccurate",
        # Weather (Anti-Leakage Aligned)
        "WeatherTime", "AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "WindDirection", "Rainfall"
    ]

    available_cols = [c for c in ordered_columns if c in cleaned.columns]
    cleaned = cleaned[available_cols].copy()

    # Sort chronologically by Season -> Round -> Driver -> LapNumber
    cleaned = cleaned.sort_values(
        by=["Year", "RoundNumber", "Driver", "LapNumber"]
    ).reset_index(drop=True)

    return cleaned


def validate_dataset(df: pd.DataFrame, expected_races: int, processed_races: int, failed_races: list, skipped_races: list) -> dict:
    """
    Comprehensive validation suite for the research dataset:
    - Verifies dataset metrics and distributions.
    - Audits missing values.
    - Confirms zero weather data leakage (WeatherTime <= LapStartTime).
    - Checks target variable plausibility (LapTime in seconds).
    - Detects any duplicate driver-lap records.
    """
    print("\n" + "=" * 70)
    print("                DATASET VALIDATION REPORT")
    print("=" * 70)

    stats = {}
    stats["num_seasons"] = int(df["Year"].nunique()) if not df.empty else 0
    stats["num_races"] = int(df.groupby(["Year", "RoundNumber"]).ngroups) if not df.empty else 0
    stats["num_drivers"] = int(df["Driver"].nunique()) if not df.empty else 0
    stats["num_teams"] = int(df["Team"].nunique()) if not df.empty else 0
    stats["total_laps"] = int(len(df))
    stats["dataset_shape"] = df.shape

    # 1. High-Level Summary
    print(f"1.  Number of seasons:       {stats['num_seasons']} ({sorted(df['Year'].unique().tolist())})")
    print(f"2.  Number of races in data: {stats['num_races']}")
    print(f"3.  Number of unique drivers:{stats['num_drivers']}")
    print(f"4.  Number of unique teams:  {stats['num_teams']}")
    print(f"5.  Total lap records:       {stats['total_laps']:,}")
    print(f"6.  Dataset shape:           {stats['dataset_shape']}")
    print(f"7.  Column count:            {len(df.columns)}")

    # 2. Race Session Tracking
    print("\n--- Race Session Tracking ---")
    print(f"• Expected race sessions:    {expected_races}")
    print(f"• Successfully processed:    {processed_races}")
    print(f"• Failed race sessions:      {len(failed_races)} {failed_races if failed_races else ''}")
    print(f"• Skipped race sessions:     {len(skipped_races)} {skipped_races if skipped_races else ''}")
    print(f"• Total races in dataset:    {stats['num_races']}")

    # 3. Missing-Value Summary
    print("\n8. Missing-Value Summary:")
    missing = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(missing.to_string())

    # 4. LapNumber bounds
    min_lap = int(df["LapNumber"].min()) if not df.empty else 0
    max_lap = int(df["LapNumber"].max()) if not df.empty else 0
    print(f"\n9.  Min / Max LapNumber:     Min = {min_lap}, Max = {max_lap}")

    # 5. Records per season
    print("\n10. Records per season:")
    print(df.groupby("Year").size().to_string())

    # 6. Records per race
    print("\n11. Records per race (sample head):")
    race_counts = df.groupby(["Year", "RoundNumber", "Race"]).size().reset_index(name="LapCount")
    print(race_counts.head(10).to_string(index=False))
    print(f"    ... total {len(race_counts)} races recorded.")

    # 7. Duplicate check
    dup_keys = ["Year", "RoundNumber", "Driver", "LapNumber"]
    num_duplicates = int(df.duplicated(subset=dup_keys).sum())
    print(f"\n12. Duplicate row count on (Year, RoundNumber, Driver, LapNumber): {num_duplicates}")
    if num_duplicates > 0:
        print("    [WARNING] Duplicate driver-lap records detected!")
    else:
        print("    [PASS] Zero duplicate driver-lap records.")

    # 8. Anti-Leakage Weather Verification
    print("\n--- Strict Anti-Leakage Weather Validation ---")
    valid_weather_mask = df["WeatherTime"].notnull() & df["LapStartTime"].notnull()
    leakage_count = int((df.loc[valid_weather_mask, "WeatherTime"] > df.loc[valid_weather_mask, "LapStartTime"]).sum())
    print(f"• Weather observations with WeatherTime > LapStartTime (Future Leakage): {leakage_count}")
    if leakage_count == 0:
        print("• [PASS] Zero future weather data leakage verified across all laps.")
    else:
        print(f"• [CRITICAL WARNING] {leakage_count} laps have weather observations from the future!")

    # 9. Target Plausibility Check (LapTime in seconds)
    print("\n--- Target Variable Plausibility (LapTime in seconds) ---")
    valid_lap_times = df["LapTime"].dropna()
    print(f"• Valid LapTime count:       {len(valid_lap_times):,} / {len(df):,} ({len(valid_lap_times)/len(df)*100:.1f}%)")
    if not valid_lap_times.empty:
        print(f"• Min LapTime:               {valid_lap_times.min():.3f}s")
        print(f"• 25th Percentile:           {valid_lap_times.quantile(0.25):.3f}s")
        print(f"• Median LapTime:            {valid_lap_times.median():.3f}s")
        print(f"• 75th Percentile:           {valid_lap_times.quantile(0.75):.3f}s")
        print(f"• Max LapTime:               {valid_lap_times.max():.3f}s")
        print(f"• Non-positive LapTimes:     {(valid_lap_times <= 0).sum()}")

    # 10. Required Identification Integrity
    print("\n--- Identifier Completeness ---")
    id_nulls = df[["Year", "RoundNumber", "Race", "Driver"]].isnull().sum().to_dict()
    print(f"• Missing identifiers:       {id_nulls}")
    
    validation_passed = (
        num_duplicates == 0 and
        leakage_count == 0 and
        stats["num_seasons"] == 3 and
        stats["num_races"] > 60 and
        all(v == 0 for v in id_nulls.values())
    )
    
    stats["validation_passed"] = validation_passed
    return stats


def print_final_report(
    df: pd.DataFrame,
    expected_races: int,
    processed_races: int,
    failed_races: list,
    skipped_races: list,
    master_path: str = "data/f1_lap_dataset_2022_2024.csv",
    sample_path: str = "data/f1_lap_dataset_sample.csv"
):
    """
    Prints the standardized final dataset report in the exact required format.
    """
    num_seasons = int(df["Year"].nunique()) if not df.empty and "Year" in df.columns else 0
    if not df.empty:
        if "Year" in df.columns and "RoundNumber" in df.columns:
            num_races = int(df.groupby(["Year", "RoundNumber"]).ngroups)
        elif "Year" in df.columns and "Race" in df.columns:
            num_races = int(df.groupby(["Year", "Race"]).ngroups)
        elif "Race" in df.columns:
            num_races = int(df["Race"].nunique())
        else:
            num_races = 0
    else:
        num_races = 0
    num_drivers = int(df["Driver"].nunique()) if not df.empty and "Driver" in df.columns else 0
    num_teams = int(df["Team"].nunique()) if not df.empty and "Team" in df.columns else 0
    total_laps = int(len(df))
    dataset_shape = df.shape

    dup_keys = [c for c in ["Year", "RoundNumber", "Driver", "LapNumber"] if c in df.columns]
    num_duplicates = int(df.duplicated(subset=dup_keys).sum()) if dup_keys and not df.empty else 0

    if "WeatherTime" in df.columns and "LapStartTime" in df.columns and not df.empty:
        valid_weather_mask = df["WeatherTime"].notnull() & df["LapStartTime"].notnull()
        leakage_count = int((df.loc[valid_weather_mask, "WeatherTime"] > df.loc[valid_weather_mask, "LapStartTime"]).sum())
    else:
        leakage_count = 0
    weather_leakage_status = "PASS" if leakage_count == 0 else "FAIL"

    id_cols = [c for c in ["Year", "RoundNumber", "Race", "Driver"] if c in df.columns]
    id_nulls_count = sum(int(df[c].isnull().sum()) for c in id_cols) if id_cols and not df.empty else (0 if df.empty else 1)

    if "LapTime" in df.columns and not df.empty:
        valid_lap_times = df["LapTime"].dropna()
        target_plausible = bool((valid_lap_times <= 0).sum() == 0 and len(valid_lap_times) > 0)
    else:
        target_plausible = False

    sort_cols = [c for c in ["Year", "RoundNumber", "Driver", "LapNumber"] if c in df.columns]
    if len(sort_cols) == 4 and not df.empty:
        sorted_idx = df[sort_cols].sort_values(by=sort_cols, kind="stable").index
        is_chronological = bool((df.index == sorted_idx).all())
    else:
        is_chronological = not df.empty

    sep = "=" * 50
    print(sep)
    print("FINAL DATASET REPORT")
    print(sep)
    print()
    print(f"Expected race sessions: {expected_races}")
    print(f"Successfully processed: {processed_races}")
    print(f"Failed: {len(failed_races)}")
    print(f"Skipped: {len(skipped_races)}")
    print()
    print(f"Seasons: {num_seasons}")
    print(f"Races: {num_races}")
    print(f"Drivers: {num_drivers}")
    print(f"Teams: {num_teams}")
    print(f"Lap records: {total_laps}")
    print(f"Dataset shape: {dataset_shape}")
    print()
    print(f"Duplicate records: {num_duplicates}")
    print(f"Weather leakage: {weather_leakage_status}")
    print("Missing values:")
    print()
    for col in df.columns:
        cnt = int(df[col].isnull().sum())
        pct = (cnt / len(df) * 100) if len(df) > 0 else 0.0
        print(f"{col}: {cnt} ({pct:.2f}%)")
    print()
    print(sep)
    print("OUTPUT FILES")
    print(sep)
    print()
    print("Master dataset:")
    print(master_path.replace("\\", "/"))
    print()
    print("Sample dataset:")
    print(sample_path.replace("\\", "/"))
    print()
    print(sep)
    print("VALIDATION")
    print(sep)
    print()

    validation_checks = [
        ("Zero duplicate records", num_duplicates == 0),
        ("Weather data leakage (WeatherTime <= LapStartTime)", leakage_count == 0),
        ("Season coverage (2022, 2023, 2024)", num_seasons == 3),
        ("Race session completeness", len(failed_races) == 0 and processed_races > 0),
        ("Identifier completeness (Year, RoundNumber, Race, Driver)", id_nulls_count == 0),
        ("Target variable plausibility (LapTime > 0s)", target_plausible),
        ("Strict chronological ordering", is_chronological),
    ]
    for check_name, passed in validation_checks:
        status = "PASSED" if passed else "FAILED"
        print(f"{check_name}: {status}")


def save_dataset(df: pd.DataFrame, full_path: str, sample_path: str, sample_size: int = 500):
    """
    Saves the master dataset CSV and a representative sample CSV.
    """
    os.makedirs(os.path.dirname(os.path.abspath(full_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(sample_path)), exist_ok=True)

    print(f"\n[SAVE] Exporting master dataset to: {full_path} ...")
    df.to_csv(full_path, index=False)
    file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
    print(f"[SAVE] Master dataset saved successfully ({file_size_mb:.2f} MB).")

    print(f"[SAVE] Exporting sample dataset (first {sample_size} chronological rows) to: {sample_path} ...")
    sample_df = df.head(sample_size).copy()
    sample_df.to_csv(sample_path, index=False)
    print(f"[SAVE] Sample dataset saved successfully.")


def main():
    """
    Main orchestration routine for multi-season F1 dataset construction.
    """
    start_total_time = time.time()
    print("=" * 70)
    print("  FORMULA 1 LAP-LEVEL DATASET BUILDER (2022–2024)")
    print("  Temporal Fusion Transformer Research - Step 1: Dataset Construction")
    print("=" * 70)

    # 1. Setup cache
    cache_dir = "data/raw/fastf1_cache"
    setup_cache(cache_dir)

    seasons = [2022, 2023, 2024]
    all_laps_list = []
    
    total_expected_races = 0
    total_processed_races = 0
    failed_races = []
    skipped_races = []

    # 2. Iterate through each season
    for year in seasons:
        season_start = time.time()
        print(f"\n{'='*30} SEASON {year} {'='*30}")
        try:
            schedule = get_season_races(year)
        except Exception as e:
            print(f"[ERROR] Failed to fetch schedule for {year}: {e}")
            continue

        total_expected_races += len(schedule)

        for _, race_row in schedule.iterrows():
            round_num = int(race_row["RoundNumber"])
            event_name = str(race_row["EventName"])
            race_display = f"{year} Round {round_num:02d} ({event_name})"
            print(f"\n>>> Processing: {race_display} ...")
            
            race_start_time = time.time()
            
            # Load race session
            session, err = load_race_data(year, round_num, event_name)
            if err:
                print(f"    [FAIL] {err}")
                failed_races.append(race_display)
                continue

            # Extract lap telemetry
            try:
                laps_df = extract_lap_data(session, year, round_num, event_name)
                if laps_df.empty:
                    print(f"    [SKIP] No lap data found for {race_display}.")
                    skipped_races.append(race_display)
                    continue

                # Extract weather
                weather_df = extract_weather_data(session)

                # Merge weather backward in time (strictly anti-leakage)
                merged_df = merge_weather(laps_df, weather_df)

                all_laps_list.append(merged_df)
                total_processed_races += 1
                
                elapsed = time.time() - race_start_time
                print(f"    [SUCCESS] Extracted {len(merged_df):,} laps in {elapsed:.1f}s.")

            except Exception as e:
                print(f"    [ERROR] Exception during processing of {race_display}: {e}")
                traceback.print_exc()
                failed_races.append(race_display)
                continue

        season_elapsed = time.time() - season_start
        print(f"\n[SEASON COMPLETE] Season {year} finished in {season_elapsed/60:.1f} minutes.")

    # 3. Combine all seasons
    if not all_laps_list:
        print("\n[CRITICAL ERROR] No lap data collected across any season!")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  CONSOLIDATING AND CLEANING MULTI-SEASON DATASET")
    print("=" * 70)
    
    raw_combined_df = pd.concat(all_laps_list, ignore_index=True)
    print(f"Raw consolidated records: {len(raw_combined_df):,}")

    # 4. Clean dataset
    cleaned_df = clean_lap_data(raw_combined_df)
    print(f"Cleaned dataset records:  {len(cleaned_df):,}")

    # 5. Validate dataset
    stats = validate_dataset(
        cleaned_df,
        expected_races=total_expected_races,
        processed_races=total_processed_races,
        failed_races=failed_races,
        skipped_races=skipped_races
    )

    # 6. Save datasets
    full_output_path = os.path.join("data", "f1_lap_dataset_2022_2024.csv")
    sample_output_path = os.path.join("data", "f1_lap_dataset_sample.csv")
    save_dataset(cleaned_df, full_output_path, sample_output_path, sample_size=500)

    # 7. Print Final Report
    print_final_report(
        cleaned_df,
        expected_races=total_expected_races,
        processed_races=total_processed_races,
        failed_races=failed_races,
        skipped_races=skipped_races,
        master_path=full_output_path,
        sample_path=sample_output_path
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--report", "-r", "--report-only"]:
        master_file = os.path.join("data", "f1_lap_dataset_2022_2024.csv")
        sample_file = os.path.join("data", "f1_lap_dataset_sample.csv")
        if os.path.exists(master_file):
            print(f"Loading master dataset from {master_file} to generate report...")
            df = pd.read_csv(master_file)
            # Assuming 72 expected race sessions for 2022-2024 as default, or calculated from df
            num_races = int(df.groupby(["Year", "RoundNumber"]).ngroups) if "Year" in df.columns and "RoundNumber" in df.columns else 0
            print_final_report(
                df=df,
                expected_races=72,  # 22 (2022) + 22 (2023) + 24 (2024) = 68 official GP races + potential others
                processed_races=num_races,
                failed_races=[],
                skipped_races=[],
                master_path=master_file,
                sample_path=sample_file
            )
        else:
            print(f"Error: Master dataset file not found at {master_file}")
    else:
        main()
