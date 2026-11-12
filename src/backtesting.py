import pandas as pd
import numpy as np
import joblib

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error


TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

MODEL_PATH = "models/xgboost_revenue_model.pkl"

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

baseline_predictions = test["Revenue_Lag_1"].values

baseline_mae = mean_absolute_error(
    actual,
    baseline_predictions
)

weights = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
]

results = []


for xgb_weight in weights:

    ets_weight = 1 - xgb_weight

    prediction = (
        xgb_weight * xgb_predictions
        +
        ets_weight * ets_predictions
    )

    mae = mean_absolute_error(
        actual,
        prediction
    )

    results.append({
        "XGBoost_Weight": xgb_weight,
        "ETS_Weight": ets_weight,
        "MAE": mae
    })


results_df = pd.DataFrame(results)

best_result = results_df.loc[
    results_df["MAE"].idxmin()
]

print("\nBaseline MAE:")
print(f"{baseline_mae:.2f}")


print("\nWeight Testing:")
print(results_df.to_string(index=False))

print(
    f"XGBoost Weight : "
    f"{best_result['XGBoost_Weight']:.1f}"
)

print(
    f"ETS Weight     : "
    f"{best_result['ETS_Weight']:.1f}"
)

print(
    f"Best MAE       : "
    f"{best_result['MAE']:.2f}"
)

improvement = (
    (baseline_mae - best_result["MAE"])
    / baseline_mae
) * 100


print(
    f"\nImprovement over baseline: "
    f"{improvement:.2f}%"
)