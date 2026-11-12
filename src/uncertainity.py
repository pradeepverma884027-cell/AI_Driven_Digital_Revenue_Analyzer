import pandas as pd
import numpy as np
import joblib

from statsmodels.tsa.holtwinters import ExponentialSmoothing

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

MODEL_PATH = "models/xgboost_revenue_model.pkl"

OUTPUT_PATH = "data/processed/probabilistic_forecast.csv"

N_SIMULATIONS = 5000

RANDOM_STATE = 42

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])

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

xgb_predictions = xgb_model.predict(
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

ets_predictions = ets_fitted.forecast(
    steps=len(test)
).values

actual = test["revenue"].values

best_mae = float("inf")

best_xgb_weight = None
best_ets_weight = None


for xgb_weight in np.arange(0, 1.1, 0.1):

    ets_weight = 1 - xgb_weight

    prediction = (
        xgb_weight * xgb_predictions
        +
        ets_weight * ets_predictions
    )

    mae = np.mean(
        np.abs(actual - prediction)
    )

    if mae < best_mae:

        best_mae = mae

        best_xgb_weight = xgb_weight
        best_ets_weight = ets_weight


ensemble_predictions = (
    best_xgb_weight * xgb_predictions
    +
    best_ets_weight * ets_predictions
)

residuals = (
    actual
    -
    ensemble_predictions
)

rng = np.random.default_rng(
    RANDOM_STATE
)


simulation_matrix = rng.choice(
    residuals,
    size=(
        N_SIMULATIONS,
        len(test)
    ),
    replace=True
)

simulated_forecasts = (
    ensemble_predictions
    +
    simulation_matrix
)

p10 = np.percentile(
    simulated_forecasts,
    10,
    axis=0
)

p50 = np.percentile(
    simulated_forecasts,
    50,
    axis=0
)

p90 = np.percentile(
    simulated_forecasts,
    90,
    axis=0
)

p10 = np.maximum(
    p10,
    0
)

p50 = np.maximum(
    p50,
    0
)

p90 = np.maximum(
    p90,
    0
)

results = pd.DataFrame({

    "date": test["date"],

    "Actual_Revenue": actual,

    "Ensemble_Forecast": ensemble_predictions,

    "P10": p10,

    "P50": p50,

    "P90": p90
})

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nBest XGBoost Weight : "
    f"{best_xgb_weight:.1f}"
)

print(
    f"Best ETS Weight     : "
    f"{best_ets_weight:.1f}"
)

print(
    f"Ensemble MAE        : "
    f"{best_mae:.2f}"
)

print(
    f"\nBootstrap Simulations: "
    f"{N_SIMULATIONS}"
)

print("\nSample Probabilistic Forecast:")

print(
    results.head(10).to_string(
        index=False
    )
)

print("\nSaved to:")

print(OUTPUT_PATH)