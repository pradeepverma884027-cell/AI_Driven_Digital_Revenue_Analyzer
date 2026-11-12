import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DATA_PATH = "data/processed/daily_revenue.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

MODEL_PATH = "models/xgboost_revenue_model.pkl"

OUTPUT_PATH = "data/processed/probabilistic_forecast.csv"

FORECAST_DAYS = 90
N_SIMULATIONS = 5000
RANDOM_STATE = 42


df = pd.read_csv(DATA_PATH)
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

df["date"] = pd.to_datetime(df["date"])
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])

df = df.sort_values("date").reset_index(drop=True)
train = train.sort_values("date").reset_index(drop=True)
test = test.sort_values("date").reset_index(drop=True)

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

xgb_model = joblib.load(MODEL_PATH)

xgb_test_predictions = xgb_model.predict(
    test[features]
)

y_train = train.set_index("date")["revenue"]

ets_model = ExponentialSmoothing(
    y_train,
    trend="add",
    seasonal="add",
    seasonal_periods=7
)

ets_fitted = ets_model.fit(
    optimized=True
)

ets_test_predictions = ets_fitted.forecast(
    steps=len(test)
).values

actual_test = test["revenue"].values

best_mae = float("inf")
best_xgb_weight = 0.5
best_ets_weight = 0.5

for xgb_weight in np.arange(0, 1.1, 0.1):

    ets_weight = 1 - xgb_weight

    prediction = (
        xgb_weight * xgb_test_predictions
        +
        ets_weight * ets_test_predictions
    )

    mae = mean_absolute_error(
        actual_test,
        prediction
    )

    if mae < best_mae:

        best_mae = mae
        best_xgb_weight = xgb_weight
        best_ets_weight = ets_weight

test_ensemble = (
    best_xgb_weight * xgb_test_predictions
    +
    best_ets_weight * ets_test_predictions
)

residuals = (
    actual_test - test_ensemble
)

history = df.copy()

future_xgb = []

last_date = history["date"].iloc[-1]


def create_features(history, future_date):

    revenue = history["revenue"]
    spend = history["spend"]
    clicks = history["clicks"]
    conversions = history["conversions"]

    return pd.DataFrame([{

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

        "Spend_Lag_1": spend.iloc[-1],
        "Spend_Lag_7": spend.iloc[-7],
        "Clicks_Lag_1": clicks.iloc[-1],
        "Conversions_Lag_1": conversions.iloc[-1]

    }])


for day in range(1, FORECAST_DAYS + 1):

    future_date = (
        last_date +
        pd.Timedelta(days=day)
    )

    X_future = create_features(
        history,
        future_date
    )

    prediction = xgb_model.predict(
        X_future
    )[0]

    prediction = max(prediction, 0)

    future_xgb.append(prediction)

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


future_xgb = np.array(future_xgb)

full_series = df.set_index("date")["revenue"]

future_ets_model = ExponentialSmoothing(
    full_series,
    trend="add",
    seasonal="add",
    seasonal_periods=7
)

future_ets_fitted = future_ets_model.fit(
    optimized=True
)

future_ets = future_ets_fitted.forecast(
    FORECAST_DAYS
).values

future_ensemble = (
    best_xgb_weight * future_xgb
    +
    best_ets_weight * future_ets
)

future_ensemble = np.maximum(
    future_ensemble,
    0
)

rng = np.random.default_rng(
    RANDOM_STATE
)

simulations = np.zeros(
    (N_SIMULATIONS, FORECAST_DAYS)
)


for i in range(N_SIMULATIONS):

    sampled_errors = rng.choice(
        residuals,
        size=FORECAST_DAYS,
        replace=True
    )

    simulations[i] = np.maximum(
        future_ensemble + sampled_errors,
        0
    )

p10_daily = np.percentile(
    simulations,
    10,
    axis=0
)

p50_daily = np.percentile(
    simulations,
    50,
    axis=0
)

p90_daily = np.percentile(
    simulations,
    90,
    axis=0
)

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=FORECAST_DAYS,
    freq="D"
)

daily_output = pd.DataFrame({

    "date": future_dates,

    "Forecast":
        future_ensemble,

    "P10":
        p10_daily,

    "P50":
        p50_daily,

    "P90":
        p90_daily
})


daily_output.to_csv(
    OUTPUT_PATH,
    index=False
)

summary = []

for days in [30, 60, 90]:

    summary.append({

        "Horizon_Days": days,

        "P10":
            simulations[:, :days].sum(axis=1).mean(),

        "P50":
            np.percentile(
                simulations[:, :days].sum(axis=1),
                50
            ),

        "P90":
            np.percentile(
                simulations[:, :days].sum(axis=1),
                90
            )
    })


summary_df = pd.DataFrame(summary)

print(
    f"\nXGBoost Weight : "
    f"{best_xgb_weight:.1f}"
)

print(
    f"ETS Weight     : "
    f"{best_ets_weight:.1f}"
)

print(
    f"Simulations    : "
    f"{N_SIMULATIONS}"
)

print("\n30 / 60 / 90 DAY FORECAST")

print(
    summary_df.to_string(
        index=False
    )
)

print("\nSaved daily forecast:")
print(OUTPUT_PATH)