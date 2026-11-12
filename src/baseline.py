import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])


test["Baseline_Prediction"] = test["Revenue_Lag_1"]


y_true = test["revenue"]

y_pred = test["Baseline_Prediction"]


mae = mean_absolute_error(y_true, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_true, y_pred)
)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

results = test[
    ["date", "revenue", "Baseline_Prediction"]
].copy()

print("\nSample Predictions:")
print(results.head(10))