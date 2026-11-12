import pandas as pd
import numpy as np

# PATHS
INPUT_PATH = (
    "data/processed/cleaned_revenue.csv"
)

OUTPUT_PATH = (
    "data/processed/channel_performance.csv"
)



# LOAD DATA
df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])



# CAMPAIGN TYPE PERFORMANCE
performance = (
    df.groupby("campaign_type")
    .agg(
        Spend=("spend", "sum"),
        Revenue=("revenue", "sum"),
        Clicks=("clicks", "sum"),
        Conversions=("conversions", "sum"),
        Impressions=("impressions", "sum")
    )
    .reset_index()
)
# ROAS
performance["ROAS"] = np.where(
    performance["Spend"] > 0,
    performance["Revenue"] /
    performance["Spend"],
    0
)
# CTR
performance["CTR"] = np.where(
    performance["Impressions"] > 0,
    performance["Clicks"] /
    performance["Impressions"] * 100,
    0
)

# CONVERSION RATE
performance["Conversion_Rate"] = np.where(
    performance["Clicks"] > 0,
    performance["Conversions"] /
    performance["Clicks"] * 100,
    0
)
# CPA
performance["CPA"] = np.where(
    performance["Conversions"] > 0,
    performance["Spend"] /
    performance["Conversions"],
    0
)

# SORT BY ROAS
performance = performance.sort_values(
    "ROAS",
    ascending=False
).reset_index(drop=True)

# RANK
performance["ROAS_Rank"] = (
    performance.index + 1
)

# SAVE
performance.to_csv(
    OUTPUT_PATH,
    index=False
)


# DISPLAY


print("REVCAST CHANNEL PERFORMANCE")

print(
    performance.to_string(
        index=False
    )
)

print("\nBest performing campaign type:")

best = performance.iloc[0]

print(
    f"{best['campaign_type']} "
    f"→ ROAS {best['ROAS']:.2f}x"
)

print("\nSaved to:")
print(OUTPUT_PATH)