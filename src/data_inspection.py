import pandas as pd

DATA_PATH=pd.read_csv("data/raw/ai_revenue_forecasting_ads_dataset.csv")

df= pd.DataFrame(DATA_PATH)

print(df.head(10))

print(df.columns)

print(df.isnull().sum())

print(df.duplicated().sum())



print(df["campaign"].unique())
print(df["campaign"].nunique())

print(df["campaign_type"].unique())
print(df["campaign_type"].nunique())

print(df.describe())

print(df.describe(include=["object"]))


print(
    df[
        [
            "spend",
            "conversions",
            "revenue",
            "clicks",
            "impressions"
        ]
    ].describe()
)