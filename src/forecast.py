import pandas as pd
import numpy as np
import joblib

DATA_PATH = "data/processed/daily_revenue.csv"
MODEL_PATH = "models/xgboost_revenue_model.pkl"

OUTPUT_PATH = "data/processed/future_forecast.csv"

FORECAST_DAYS = 90

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

model = joblib.load(MODEL_PATH)

def create_features(history, future_date):

    revenue = history["revenue"]

    spend = history["spend"]
    clicks = history["clicks"]
    conversions = history["conversions"]

    features = {
        "Revenue_Lag_1": revenue.iloc[-1],
        "Revenue_Lag_7": revenue.iloc[-7],
        "Revenue_Lag_14": revenue.iloc[-14],
        "Revenue_Lag_28": revenue.iloc[-28],

        "Revenue_Rolling_7":
            revenue.iloc[-7:].mean(),

        "Revenue_Rolling_14":
            revenue.iloc[-14:].mean(),

        "Revenue_Rolling_28":
            revenue.iloc[-28:].mean(),

        "Day_of_Week":
            future_date.dayofweek,

        "Week_of_Year":
            future_date.isocalendar().week,

        "Month":
            future_date.month,

        "Quarter":
            future_date.quarter,

        "Day_Sin":
            np.sin(
                2 * np.pi *
                future_date.dayofweek / 7
            ),

        "Day_Cos":
            np.cos(
                2 * np.pi *
                future_date.dayofweek / 7
            ),

        "Month_Sin":
            np.sin(
                2 * np.pi *
                future_date.month / 12
            ),

        "Month_Cos":
            np.cos(
                2 * np.pi *
                future_date.month / 12
            ),

        "Spend_Lag_1":
            spend.iloc[-1],

        "Spend_Lag_7":
            spend.iloc[-7],

        "Clicks_Lag_1":
            clicks.iloc[-1],

        "Conversions_Lag_1":
            conversions.iloc[-1]
    }

    return pd.DataFrame([features])

history = df.copy()

future_results = []

last_date = history["date"].iloc[-1]


for day in range(1, FORECAST_DAYS + 1):

    future_date = (
        last_date +
        pd.Timedelta(days=day)
    )

    X_future = create_features(
        history,
        future_date
    )

    prediction = model.predict(
        X_future
    )[0]

    prediction = max(
        prediction,
        0
    )

    future_results.append({
        "date": future_date,
        "Predicted_Revenue": prediction
    })

    new_row = {
        "date": future_date,
        "revenue": prediction,
        "spend": history["spend"].iloc[-1],
        "clicks": history["clicks"].iloc[-1],
        "conversions": history["conversions"].iloc[-1],
        "impressions": history["impressions"].iloc[-1]
    }

    history = pd.concat(
        [
            history,
            pd.DataFrame([new_row])
        ],
        ignore_index=True
    )

forecast = pd.DataFrame(
    future_results
)

forecast.to_csv(
    OUTPUT_PATH,
    index=False
)

revenue_30 = forecast.iloc[:30][
    "Predicted_Revenue"
].sum()

revenue_60 = forecast.iloc[:60][
    "Predicted_Revenue"
].sum()

revenue_90 = forecast.iloc[:90][
    "Predicted_Revenue"
].sum()

print(
    f"\n30-Day Revenue : "
    f"{revenue_30:,.2f}"
)

print(
    f"60-Day Revenue : "
    f"{revenue_60:,.2f}"
)

print(
    f"90-Day Revenue : "
    f"{revenue_90:,.2f}"
)

print("\nFirst 10 Forecast Days:")

print(
    forecast.head(10).to_string(
        index=False
    )
)

print("\nForecast saved to:")

print(OUTPUT_PATH)