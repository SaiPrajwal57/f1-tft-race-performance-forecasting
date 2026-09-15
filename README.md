# An Interpretable Temporal Fusion Transformer for Multi-Horizon Formula 1 Race Performance Forecasting

## Step 1: Research-Grade Formula 1 Lap-Level Dataset (2022–2024)

This repository contains the dataset engineering pipeline and validated data files for multi-horizon Formula 1 race performance forecasting using Temporal Fusion Transformers (TFT).

---

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   └── fastf1_cache/               # Local cache for FastF1 API responses
│   ├── processed/                      # Intermediate processing caches
│   ├── f1_lap_dataset_2022_2024.csv    # Master research dataset (all race laps 2022-2024)
│   └── f1_lap_dataset_sample.csv       # Chronological sample dataset (first 500 records)
├── src/
│   └── build_dataset.py                # Modular dataset extraction, cleaning, and validation pipeline
├── notebooks/                          # Exploratory data analysis & EDA notebooks
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation & feature data dictionary
```

---

## Feature Data Dictionary

| Column | Type | Category | Units / Format | Description |
| :--- | :--- | :--- | :--- | :--- |
| `Year` | `int` | Identification | YYYY (2022-2024) | Formula 1 World Championship season year |
| `RoundNumber` | `int` | Identification | 1 .. 24 | Official FIA race round sequence number |
| `Race` | `str` | Identification | Text | Grand Prix event name (e.g. `Bahrain Grand Prix`) |
| `Driver` | `str` | Identification | 3-letter code | Driver abbreviation code (e.g. `VER`, `LEC`, `HAM`) |
| `DriverNumber` | `str` | Identification | Number | Permanent driver competition number |
| `Team` | `str` | Identification | Text | Constructor team name (e.g. `Red Bull Racing`, `Ferrari`) |
| `LapNumber` | `int` | Identification | 1 .. N | Sequential lap number within the Grand Prix race |
| `LapStartTime` | `float` | Timeline | Seconds | Lap start time measured from session start ($t=0$) |
| `LapTime` | `float` | Target | Seconds | **Primary Target Variable**: Completed lap duration in seconds |
| `Sector1Time` | `float` | Performance | Seconds | Sector 1 split time in seconds |
| `Sector2Time` | `float` | Performance | Seconds | Sector 2 split time in seconds |
| `Sector3Time` | `float` | Performance | Seconds | Sector 3 split time in seconds |
| `SpeedI1` | `float` | Performance | km/h | Speed trap reading at intermediate timing line 1 |
| `SpeedI2` | `float` | Performance | km/h | Speed trap reading at intermediate timing line 2 |
| `SpeedFL` | `float` | Performance | km/h | Speed trap reading at the finish line |
| `SpeedST` | `float` | Performance | km/h | Speed trap reading at the longest straight |
| `Position` | `float` | Performance | 1 .. 20 | Driver running track position at lap completion |
| `Compound` | `str` | Tyre | Text | Pirelli tyre compound (`SOFT`, `MEDIUM`, `HARD`, `INTERMEDIATE`, `WET`) |
| `TyreLife` | `float` | Tyre | Laps | Cumulative laps completed on current tyre set |
| `FreshTyre` | `bool` | Tyre | True / False | Boolean indicating whether tyres were new at stint start |
| `Stint` | `float` | Tyre | Integer (1, 2..) | Driver stint index within the race |
| `PitInTime` | `float` | Pit Stop | Seconds | Session time when car entered the pit lane (`NaN` on non-pit laps) |
| `PitOutTime` | `float` | Pit Stop | Seconds | Session time when car exited the pit lane (`NaN` on non-pit laps) |
| `TrackStatus` | `str` | Race / Track | Status codes | Track flags (`1`=Clear, `2`=Yellow, `4`=Safety Car, `5`=Red Flag, `6`=VSC, etc.) |
| `IsAccurate` | `bool` | Race / Track | True / False | FastF1 accuracy flag indicating uninterrupted lap timing |
| `WeatherTime` | `float` | Weather | Seconds | Session timestamp of the merged weather measurement |
| `AirTemp` | `float` | Weather | °C | Ambient air temperature |
| `TrackTemp` | `float` | Weather | °C | Track surface temperature |
| `Humidity` | `float` | Weather | % | Relative humidity |
| `Pressure` | `float` | Weather | mbar | Barometric pressure |
| `WindSpeed` | `float` | Weather | m/s | Wind speed |
| `WindDirection`| `int` | Weather | Degrees (0–360) | Wind direction heading |
| `Rainfall` | `bool` | Weather | True / False | Boolean flag indicating active track precipitation |

---

## Anti-Data-Leakage Guarantees

1. **Chronological Alignment**:
   - The dataset is strictly ordered by `[Year, RoundNumber, Driver, LapNumber]`.
   - Data is never randomly shuffled or split prior to multi-horizon sequence windowing.
2. **Backward Weather Merging**:
   - Weather observations are integrated using `pd.merge_asof(..., direction="backward")` matching on `LapStartTime` vs `WeatherTime`.
   - Guaranteed invariant: $\text{WeatherTime} \le \text{LapStartTime}$.
   - No future weather readings or telemetry are ever accessible to past laps.
3. **Target Integrity**:
   - `LapTime` in seconds is maintained as the target variable for multi-horizon forecasting.

---

## Reproducibility & Execution

To reproduce the dataset extraction from the project root:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dataset construction pipeline
python src/build_dataset.py
```
