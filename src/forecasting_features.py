import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/daily_revenue.csv"
OUTPUT_PATH = "data/processed/forecasting_features.csv"

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

df["Revenue_Lag_1"] = df["revenue"].shift(1)

df["Revenue_Lag_7"] = df["revenue"].shift(7)

df["Revenue_Lag_14"] = df["revenue"].shift(14)

df["Revenue_Lag_28"] = df["revenue"].shift(28)

df["Revenue_Rolling_7"] = (
    df["revenue"]
    .shift(1)
    .rolling(7)
    .mean()
)

df["Revenue_Rolling_14"] = (
    df["revenue"]
    .shift(1)
    .rolling(14)
    .mean()
)

df["Revenue_Rolling_28"] = (
    df["revenue"]
    .shift(1)
    .rolling(28)
    .mean()
)

df["Day_of_Week"] = df["date"].dt.dayofweek

df["Week_of_Year"] = (
    df["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

df["Month"] = df["date"].dt.month

df["Quarter"] = df["date"].dt.quarter

df["Day_Sin"] = np.sin(
    2 * np.pi * df["Day_of_Week"] / 7
)

df["Day_Cos"] = np.cos(
    2 * np.pi * df["Day_of_Week"] / 7
)

df["Month_Sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_Cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)

df["Spend_Lag_1"] = df["spend"].shift(1)

df["Spend_Lag_7"] = df["spend"].shift(7)

df["Clicks_Lag_1"] = df["clicks"].shift(1)

df["Conversions_Lag_1"] = df["conversions"].shift(1)

df = df.dropna().reset_index(drop=True)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Forecasting features created successfully.")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())