from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

app = FastAPI(
    title="REVCAST API",
    description="AI-Driven Probabilistic Revenue Forecasting API",
    version="1.0.0"
)

FORECAST_PATH = "data/processed/probabilistic_forecast.csv"
ROAS_PATH = "data/processed/roas_forecast.csv"
DRIVERS_PATH = "data/processed/revenue_drivers.csv"
SCENARIO_PATH = "data/processed/scenario_forecast.csv"

app.mount(
    "/static",
    StaticFiles(directory="app"),
    name="static"
)


@app.get("/")
def dashboard():
    return FileResponse("app/dashboard.html")


@app.get("/health")
def health():
    return {
        "project": "REVCAST",
        "status": "running",
        "message": "Revenue forecasting API is active"
    }


@app.get("/forecast")
def forecast():

    df = pd.read_csv(FORECAST_PATH)

    return df.to_dict(orient="records")


@app.get("/forecast/summary")
def forecast_summary():

    df = pd.read_csv(FORECAST_PATH)

    summary = []

    for days in [30, 60, 90]:

        data = df.iloc[:days]

        summary.append({
            "horizon_days": days,
            "p10_revenue": float(
                data["P10"].sum()
            ),
            "p50_revenue": float(
                data["P50"].sum()
            ),
            "p90_revenue": float(
                data["P90"].sum()
            )
        })

    return {
        "forecast": summary
    }


@app.get("/roas")
def roas():

    df = pd.read_csv(ROAS_PATH)

    return df.to_dict(orient="records")


@app.get("/drivers")
def drivers():

    df = pd.read_csv(DRIVERS_PATH)

    return df.head(10).to_dict(
        orient="records"
    )


@app.get("/insights")
def insights():

    forecast = pd.read_csv(
        FORECAST_PATH
    )

    drivers = pd.read_csv(
        DRIVERS_PATH
    )

    revenue_30 = (
        forecast.iloc[:30]["P50"].sum()
    )

    revenue_60 = (
        forecast.iloc[:60]["P50"].sum()
    )

    revenue_90 = (
        forecast.iloc[:90]["P50"].sum()
    )

    top_drivers = drivers.head(3)[
        ["Rank", "Feature", "Importance"]
    ].to_dict(
        orient="records"
    )

    return {

        "revenue_forecast": {

            "30_day": float(
                revenue_30
            ),

            "60_day": float(
                revenue_60
            ),

            "90_day": float(
                revenue_90
            )
        },

        "top_drivers": top_drivers,

        "assumptions": [

            "Future marketing activity uses latest known values.",

            "Forecast uncertainty is estimated using historical residuals.",

            "P50 represents the median forecast."
        ]
    }


@app.get("/scenarios")
def scenarios():

    df = pd.read_csv(
        SCENARIO_PATH
    )

    print(
        "Scenario CSV columns:",
        list(df.columns)
    )

    columns = {
        column.lower().strip(): column
        for column in df.columns
    }

    scenario_column = None

    for name in [
        "scenario",
        "scenario_name",
        "name"
    ]:

        if name in columns:
            scenario_column = columns[name]
            break

    if scenario_column is None:

        raise ValueError(
            "Scenario column not found. "
            f"Available columns: {list(df.columns)}"
        )


    revenue_column = None

    for name in [
        "forecast_revenue",
        "revenue",
        "forecastrevenue"
    ]:

        if name in columns:
            revenue_column = columns[name]
            break

    if revenue_column is None:

        raise ValueError(
            "Revenue column not found. "
            f"Available columns: {list(df.columns)}"
        )


    spend_column = None

    for name in [
        "forecast_spend",
        "spend",
        "forecastspend"
    ]:

        if name in columns:
            spend_column = columns[name]
            break


    roas_column = None

    for name in [
        "forecast_roas",
        "roas",
        "forecastroas"
    ]:

        if name in columns:
            roas_column = columns[name]
            break


    horizon_column = None

    for name in [
        "horizon_days",
        "horizon",
        "days",
        "forecast_days"
    ]:

        if name in columns:
            horizon_column = columns[name]
            break


    result = []

    grouped = df.groupby(
        scenario_column,
        sort=False
    )


    for scenario_name, group in grouped:

        group = group.reset_index(
            drop=True
        )

        if horizon_column is not None:

            horizons = pd.to_numeric(
                group[horizon_column],
                errors="coerce"
            ).tolist()

        else:

            horizons = []

            for index in range(
                len(group)
            ):

                if index == 0:
                    horizons.append(30)

                elif index == 1:
                    horizons.append(60)

                elif index == 2:
                    horizons.append(90)

                else:
                    horizons.append(
                        (index + 1) * 30
                    )


        for index, row in group.iterrows():

            revenue = pd.to_numeric(
                row[revenue_column],
                errors="coerce"
            )

            if spend_column is not None:

                spend = pd.to_numeric(
                    row[spend_column],
                    errors="coerce"
                )

            else:

                spend = 0


            if roas_column is not None:

                roas = pd.to_numeric(
                    row[roas_column],
                    errors="coerce"
                )

            else:

                if (
                    pd.notna(revenue)
                    and
                    pd.notna(spend)
                    and
                    float(spend) > 0
                ):

                    roas = (
                        float(revenue)
                        /
                        float(spend)
                    )

                else:

                    roas = 0


            result.append({

                "Scenario":
                    str(scenario_name),

                "Horizon_Days":
                    int(horizons[index]),

                "Forecast_Revenue":
                    float(
                        revenue
                    )
                    if pd.notna(revenue)
                    else 0,

                "Forecast_Spend":
                    float(
                        spend
                    )
                    if pd.notna(spend)
                    else 0,

                "Forecast_ROAS":
                    float(
                        roas
                    )
                    if pd.notna(roas)
                    else 0
            })


    return result
