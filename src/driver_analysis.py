import pandas as pd
import joblib

MODEL_PATH = "models/xgboost_revenue_model.pkl"

OUTPUT_PATH = "data/processed/revenue_drivers.csv"

model = joblib.load(
    MODEL_PATH
)

features = [
    "Revenue_Lag_1",
    "Revenue_Lag_7",
    "Revenue_Lag_14",
    "Revenue_Lag_28",

    "Revenue_Rolling_7",
    "Revenue_Rolling_14",
    "Revenue_Rolling_28",

    "Day_of_Week",
    "Week_of_Year",
    "Month",
    "Quarter",

    "Day_Sin",
    "Day_Cos",
    "Month_Sin",
    "Month_Cos",

    "Spend_Lag_1",
    "Spend_Lag_7",
    "Clicks_Lag_1",
    "Conversions_Lag_1"
]

drivers = pd.DataFrame({

    "Feature": features,

    "Importance":
        model.feature_importances_
})

drivers = drivers.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

drivers["Rank"] = (
    drivers.index + 1
)

drivers.to_csv(
    OUTPUT_PATH,
    index=False
)

print("REVCAST REVENUE DRIVER ANALYSIS")

print("\nTop Revenue Drivers:")

print(
    drivers[
        ["Rank", "Feature", "Importance"]
    ]
    .head(10)
    .to_string(index=False)
)


print("\nSaved to:")
print(OUTPUT_PATH)