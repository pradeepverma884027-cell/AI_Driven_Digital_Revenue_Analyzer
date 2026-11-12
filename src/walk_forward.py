import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# PATH

DATA_PATH = (
    "data/processed/forecasting_features.csv"
)
# LOAD DATA
df = pd.read_csv(DATA_PATH)

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


TARGET = "revenue"

# WALK-FORWARD SETTINGS

initial_train_size = int(
    len(df) * 0.50
)

test_size = 30

results = []

# WALK-FORWARD LOOP

fold = 1

train_end = initial_train_size


while train_end + test_size <= len(df):

    train = df.iloc[
        :train_end
    ]

    test = df.iloc[
        train_end:
        train_end + test_size
    ]


    # TRAIN
 
    X_train = train[features]

    y_train = train[TARGET]

    # TEST

    X_test = test[features]

    y_test = test[TARGET]

    # MODEL
    model = XGBRegressor(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=6,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=42

    )



    # FIT
    model.fit(
        X_train,
        y_train
    )

    # PREDICT

    predictions = model.predict(
        X_test
    )



    # METRICS

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

    # STORE

    results.append({

        "Fold":
            fold,

        "Train_Start":
            train["date"].min(),

        "Train_End":
            train["date"].max(),

        "Test_Start":
            test["date"].min(),

        "Test_End":
            test["date"].max(),

        "MAE":
            mae,

        "RMSE":
            rmse

    })


    print(
        f"Fold {fold} | "
        f"Train: {train['date'].min().date()} "
        f"→ {train['date'].max().date()} | "
        f"Test: {test['date'].min().date()} "
        f"→ {test['date'].max().date()} | "
        f"MAE: {mae:.2f}"
    )

    # MOVE WINDOW FORWARD
    train_end += test_size

    fold += 1


# RESULTS DATAFRAME

results_df = pd.DataFrame(
    results
)

# OVERALL PERFORMANCE

average_mae = results_df[
    "MAE"
].mean()

average_rmse = results_df[
    "RMSE"
].mean()

# DISPLAY
print("\n")

print("REVCAST WALK-FORWARD VALIDATION")

print("\nFold Results:")

print(
    results_df[
        [
            "Fold",
            "Test_Start",
            "Test_End",
            "MAE",
            "RMSE"
        ]
    ].to_string(
        index=False
    )
)


print("\nOverall Performance:")

print(
    f"Average MAE  : "
    f"{average_mae:.2f}"
)

print(
    f"Average RMSE : "
    f"{average_rmse:.2f}"
)
