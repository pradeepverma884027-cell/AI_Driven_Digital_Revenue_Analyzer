import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error


#paths
DATA_PATH = (
    "data/processed/forecasting_features.csv"
)

MODEL_PATH = (
    "models/tuned_xgboost_revenue_model.pkl"
)
# LOAD DATA

df = pd.read_csv(
    DATA_PATH
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)
# FEATURES

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


X = df[features]

y = df["revenue"]



# TIME SERIES SPLIT

tscv = TimeSeriesSplit(
    n_splits=4
)


# PARAMETER COMBINATIONS
parameter_sets = [

    {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3
    },

    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4
    },

    {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 5
    },

    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 4
    },

    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 6
    },

    {
        "n_estimators": 600,
        "learning_rate": 0.02,
        "max_depth": 5
    }
]


# TUNING

results = []


for params in parameter_sets:

    fold_errors = []


    for train_index, test_index in tscv.split(X):

        X_train = X.iloc[
            train_index
        ]

        X_test = X.iloc[
            test_index
        ]

        y_train = y.iloc[
            train_index
        ]

        y_test = y.iloc[
            test_index
        ]


        model = XGBRegressor(

            n_estimators=
                params["n_estimators"],

            learning_rate=
                params["learning_rate"],

            max_depth=
                params["max_depth"],

            subsample=0.8,

            colsample_bytree=0.8,

            objective="reg:squarederror",

            random_state=42

        )


        model.fit(
            X_train,
            y_train
        )


        predictions = model.predict(
            X_test
        )


        mae = mean_absolute_error(
            y_test,
            predictions
        )


        fold_errors.append(
            mae
        )


    average_mae = np.mean(
        fold_errors
    )


    results.append({

        **params,

        "Average_MAE":
            average_mae

    })

# RESULTS

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Average_MAE"
).reset_index(drop=True)


# BEST PARAMETERS

best = results_df.iloc[0]


print("REVCAST XGBOOST HYPERPARAMETER TUNING")

print("\nAll Results:")

print(
    results_df.to_string(
        index=False
    )
)


print("\nBest Parameters:")

print(
    best.to_string()
)


# TRAIN FINAL MODEL
best_model = XGBRegressor(

    n_estimators=
        int(best["n_estimators"]),

    learning_rate=
        float(best["learning_rate"]),

    max_depth=
        int(best["max_depth"]),

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42

)


best_model.fit(
    X,
    y
)

# SAVE

joblib.dump(
    best_model,
    MODEL_PATH
)


print("\nTuned model saved to:")

print(MODEL_PATH)