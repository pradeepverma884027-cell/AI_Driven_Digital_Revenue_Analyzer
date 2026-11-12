import pandas as pd
import numpy as np

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error



TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"



train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])

train = train.sort_values("date")
test = test.sort_values("date")

y_train = train.set_index("date")["revenue"]

y_test = test.set_index("date")["revenue"]

model = ExponentialSmoothing(
    y_train,
    trend="add",
    seasonal="add",
    seasonal_periods=7
)
fitted_model = model.fit(
    optimized=True
)

forecast = fitted_model.forecast(
    steps=len(y_test)
)

mae = mean_absolute_error(
    y_test,
    forecast
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        forecast
    )
)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

results = pd.DataFrame({
    "date": y_test.index,
    "Actual_Revenue": y_test.values,
    "TimeSeries_Prediction": forecast.values
})

print("\nSample Predictions:")
print(results.head(10))