import pandas as pd
import numpy as np
import joblib


#paths
DATA_PATH = "data/processed/daily_revenue.csv"

MODEL_PATH = "models/tuned_xgboost_revenue_model.pkl"

OUTPUT_PATH = "data/processed/scenario_forecast.csv"

# SETTINGS
FORECAST_DAYS = 90


SCENARIOS = {
    "Conservative": 0.90,
    "Baseline": 1.00,
    "Growth": 1.10,
    "Aggressive": 1.20
}

# LOAD DATA
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)
# LOAD MODEL

model = joblib.load(
    MODEL_PATH
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

# FEATURE CREATION
def create_features(history, future_date):

    revenue = history["revenue"]

    spend = history["spend"]

    clicks = history["clicks"]

    conversions = history["conversions"]


    return pd.DataFrame([{

        "Revenue_Lag_1":
            revenue.iloc[-1],

        "Revenue_Lag_7":
            revenue.iloc[-7],

        "Revenue_Lag_14":
            revenue.iloc[-14],

        "Revenue_Lag_28":
            revenue.iloc[-28],


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

    }])
# RUN SCENARIO
def run_scenario(
    scenario_name,
    spend_multiplier
):

    history = df.copy()

    predictions = []

    last_date = history["date"].iloc[-1]


    for day in range(
        1,
        FORECAST_DAYS + 1
    ):

        future_date = (
            last_date
            +
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


        future_spend = (
            history["spend"].iloc[-1]
            *
            spend_multiplier
        )


        predictions.append({

            "date":
                future_date,

            "Scenario":
                scenario_name,

            "Spend_Multiplier":
                spend_multiplier,

            "Forecast_Revenue":
                prediction,

            "Forecast_Spend":
                future_spend

        })


        # Add predicted revenue and
        # scenario spend to history

        new_row = {

            "date":
                future_date,

            "revenue":
                prediction,

            "spend":
                future_spend,

            "clicks":
                history["clicks"].iloc[-1],

            "conversions":
                history["conversions"].iloc[-1],

            "impressions":
                history["impressions"].iloc[-1]

        }


        history = pd.concat(

            [
                history,

                pd.DataFrame(
                    [new_row]
                )
            ],

            ignore_index=True

        )


    return pd.DataFrame(
        predictions
    )

# RUN ALL SCENARIOS
all_scenarios = []


for scenario_name, multiplier in SCENARIOS.items():

    result = run_scenario(
        scenario_name,
        multiplier
    )

    all_scenarios.append(
        result
    )


scenario_forecast = pd.concat(
    all_scenarios,
    ignore_index=True
)
# ROAS

scenario_forecast["Forecast_ROAS"] = np.where(

    scenario_forecast["Forecast_Spend"] > 0,

    scenario_forecast["Forecast_Revenue"]
    /
    scenario_forecast["Forecast_Spend"],

    0
)
# SAVE
scenario_forecast.to_csv(
    OUTPUT_PATH,
    index=False
)

# SUMMARY

summary = []


for scenario in SCENARIOS:

    scenario_data = (
        scenario_forecast[
            scenario_forecast["Scenario"]
            == scenario
        ]
    )


    for days in [30, 60, 90]:

        data = scenario_data.iloc[
            :days
        ]


        total_revenue = (
            data["Forecast_Revenue"]
            .sum()
        )


        total_spend = (
            data["Forecast_Spend"]
            .sum()
        )


        roas = (
            total_revenue
            /
            total_spend
            if total_spend > 0
            else 0
        )


        summary.append({

            "Scenario":
                scenario,

            "Horizon_Days":
                days,

            "Forecast_Revenue":
                total_revenue,

            "Forecast_Spend":
                total_spend,

            "Forecast_ROAS":
                roas

        })


summary_df = pd.DataFrame(
    summary
)

# DISPLAY
print("REVCAST MARKETING SCENARIO ANALYSIS")

print("\n")

print(
    summary_df.to_string(
        index=False
    )
)


print("\nSaved to:")

print(OUTPUT_PATH)