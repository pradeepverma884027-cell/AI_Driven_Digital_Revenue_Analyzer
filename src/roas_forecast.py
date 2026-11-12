import pandas as pd
import numpy as np

HISTORICAL_PATH = "data/processed/daily_revenue.csv"
FORECAST_PATH = "data/processed/probabilistic_forecast.csv"

OUTPUT_PATH = "data/processed/roas_forecast.csv"

historical = pd.read_csv(HISTORICAL_PATH)
forecast = pd.read_csv(FORECAST_PATH)

historical["date"] = pd.to_datetime(
    historical["date"]
)

forecast["date"] = pd.to_datetime(
    forecast["date"]
)
latest_spend = historical["spend"].iloc[-1]


forecast["Forecast_Spend"] = latest_spend

forecast["ROAS_P10"] = np.where(
    forecast["Forecast_Spend"] > 0,
    forecast["P10"] / forecast["Forecast_Spend"],
    0
)

forecast["ROAS_P50"] = np.where(
    forecast["Forecast_Spend"] > 0,
    forecast["P50"] / forecast["Forecast_Spend"],
    0
)

forecast["ROAS_P90"] = np.where(
    forecast["Forecast_Spend"] > 0,
    forecast["P90"] / forecast["Forecast_Spend"],
    0
)

forecast.to_csv(
    OUTPUT_PATH,
    index=False
)

summary = []

for days in [30, 60, 90]:

    data = forecast.iloc[:days]

    total_spend = data["Forecast_Spend"].sum()

    p10_revenue = data["P10"].sum()
    p50_revenue = data["P50"].sum()
    p90_revenue = data["P90"].sum()

    summary.append({

        "Horizon_Days": days,

        "Forecast_Spend":
            total_spend,

        "ROAS_P10":
            p10_revenue / total_spend
            if total_spend > 0 else 0,

        "ROAS_P50":
            p50_revenue / total_spend
            if total_spend > 0 else 0,

        "ROAS_P90":
            p90_revenue / total_spend
            if total_spend > 0 else 0
    })


summary_df = pd.DataFrame(summary)

print("REVCAST ROAS FORECAST")
print(
    "\nFuture spend assumption:"
)

print(
    f"₹{latest_spend:,.2f} per day"
)

print("\n30 / 60 / 90 Day ROAS:")

print(
    summary_df.to_string(
        index=False
    )
)

print("\nSaved to:")

print(OUTPUT_PATH)