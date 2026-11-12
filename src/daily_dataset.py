import pandas as pd

INPUT_PATH = "data/processed/featured_dataset.csv"
OUTPUT_PATH = "data/processed/daily_revenue.csv"

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

daily = (
    df.groupby("date")
    .agg(
        spend=("spend", "sum"),
        clicks=("clicks", "sum"),
        conversions=("conversions", "sum"),
        impressions=("impressions", "sum"),
        revenue=("revenue", "sum")
    )
    .reset_index()
)

daily = daily.sort_values("date")

daily.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Daily dataset created.")
print("Shape:", daily.shape)
print(daily.head())