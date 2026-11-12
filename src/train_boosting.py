import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

MODEL_PATH = "models/xgboost_revenue_model.pkl"


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])

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

X_train = train[features]
y_train = train["revenue"]

X_test = test[features]
y_test = test["revenue"]

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop Feature Drivers:")
print(importance.head(10))

results = pd.DataFrame({
    "date": test["date"],
    "Actual_Revenue": y_test,
    "Predicted_Revenue": predictions
})

print("\nSample Predictions:")
print(results.head(10))

joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel saved successfully:")
print(MODEL_PATH)