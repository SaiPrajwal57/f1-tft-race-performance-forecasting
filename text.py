import fastf1
import pandas as pd

# Load 2024 Monaco GP Race
session = fastf1.get_session(2024, "Monaco", "R")
session.load()

# Get lap data
laps = session.laps.copy()

# Select features for our project
columns = [
    "Driver",
    "DriverNumber",
    "Team",
    "LapNumber",
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "Stint",
    "Position",
    "PitInTime",
    "PitOutTime",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "TrackStatus",
    "IsAccurate"
]

dataset = laps[columns].copy()

# Convert time columns to seconds
time_columns = [
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time"
]

for col in time_columns:
    dataset[col] = dataset[col].dt.total_seconds()

# Create race information
dataset["Year"] = 2024
dataset["Race"] = "Monaco"

# Sort properly
dataset = dataset.sort_values(
    ["Driver", "LapNumber"]
).reset_index(drop=True)

# Save
dataset.to_csv("monaco_2024_dataset.csv", index=False)

print("\nDataset shape:")
print(dataset.shape)

print("\nColumns:")
print(dataset.columns.tolist())

print("\nFirst 10 rows:")
print(dataset.head(10))

print("\nMissing values:")
print(dataset.isnull().sum())