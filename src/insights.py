import pandas as pd

FORECAST_PATH = (
    "data/processed/probabilistic_forecast.csv"
)

DRIVERS_PATH = (
    "data/processed/revenue_drivers.csv"
)

forecast = pd.read_csv(
    FORECAST_PATH
)

drivers = pd.read_csv(
    DRIVERS_PATH
)

revenue_30 = forecast.iloc[:30]["P50"].sum()

revenue_60 = forecast.iloc[:60]["P50"].sum()

revenue_90 = forecast.iloc[:90]["P50"].sum()


p10_90 = forecast.iloc[:90]["P10"].sum()

p90_90 = forecast.iloc[:90]["P90"].sum()

top_drivers = drivers.head(3)

print("\n")

print("          REVCAST INSIGHT CARD")

print("\n REVENUE FORECAST")

print(
    f"30-Day P50 Revenue : "
    f"₹{revenue_30:,.2f}"
)

print(
    f"60-Day P50 Revenue : "
    f"₹{revenue_60:,.2f}"
)

print(
    f"90-Day P50 Revenue : "
    f"₹{revenue_90:,.2f}"
)


print("\n 90-DAY UNCERTAINTY")

print(
    f"P10 : ₹{p10_90:,.2f}"
)

print(
    f"P50 : ₹{revenue_90:,.2f}"
)

print(
    f"P90 : ₹{p90_90:,.2f}"
)


print("\n TOP 3 MODEL DRIVERS")

for _, row in top_drivers.iterrows():

    print(
        f"{int(row['Rank'])}. "
        f"{row['Feature']} "
        f"({row['Importance']:.3f})"
    )


print("\n ASSUMPTIONS")

print(
    "• Future marketing activity is based "
    "on the latest known values."
)

print(
    "• Forecast uncertainty is estimated "
    "using historical model residuals."
)

print(
    "• P50 represents the median forecast."
)


print("\n RECOMMENDED ACTION")

print(
    "Use the P50 forecast as the central "
    "planning scenario and compare the "
    "P10/P90 range before making budget "
    "decisions."
)