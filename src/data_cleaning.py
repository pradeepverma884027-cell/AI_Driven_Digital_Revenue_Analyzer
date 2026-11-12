import pandas as pd

RAW_DATA_PATH = "data/raw/ai_revenue_forecasting_ads_dataset.csv"
df = pd.read_csv(RAW_DATA_PATH)
cleaned_df = df.copy()

duplicate_count = cleaned_df.duplicated().sum()
print("Duplicates found:", duplicate_count)

cleaned_df = cleaned_df.drop_duplicates()

print("Duplicates after removal:",cleaned_df.duplicated().sum())

cleaned_df["date"] = pd.to_datetime(cleaned_df["date"],errors="coerce")
print("Invalid / missing dates:",cleaned_df["date"].isnull().sum())
print(cleaned_df["date"])

cleaned_df = cleaned_df.dropna(subset=["date"])

print("Missing date after cleaning",cleaned_df["date"].isnull().sum())

cleaned_df = cleaned_df.dropna(subset=["revenue"])
print("Missing revenue after cleaning",cleaned_df["revenue"].isnull().sum())

missing_campaign = cleaned_df["campaign"].isnull().sum()

print("Missing campaign values:", missing_campaign)

cleaned_df["campaign"] = cleaned_df["campaign"].fillna("Unknown")

print("Missing campaign values after cleaning:",cleaned_df["campaign"].isnull().sum())

missing_campaign_type = cleaned_df["campaign_type"].isnull().sum()

print("Missing campaign types:", missing_campaign_type)

cleaned_df["campaign_type"] = cleaned_df["campaign_type"].fillna("Unknown")

print("Missing campaign types after cleaning:",cleaned_df["campaign_type"].isnull().sum())

spend_median = cleaned_df["spend"].median()

print("Median spend:", spend_median)
print("Missing spend:", cleaned_df["spend"].isnull().sum())

cleaned_df["spend"] = cleaned_df["spend"].fillna(spend_median)

print("Missing spend after cleaning:",cleaned_df["spend"].isnull().sum())

conversions_median = cleaned_df["conversions"].median()
print("Missing conversions:",cleaned_df["conversions"].isnull().sum())

cleaned_df["conversions"] = cleaned_df["conversions"].fillna(conversions_median)

print("Missing conversions after cleaning:",cleaned_df["conversions"].isnull().sum())

impressions_median = cleaned_df["impressions"].median()

print("Missing impressions:",cleaned_df["impressions"].isnull().sum())

cleaned_df["impressions"] = cleaned_df["impressions"].fillna(impressions_median)

print("Missing impressions after cleaning:",cleaned_df["impressions"].isnull().sum())

clicks_median = cleaned_df["clicks"].median()
print("Median clicks:", clicks_median)
print("Missing clicks:",cleaned_df["clicks"].isnull().sum())

cleaned_df["clicks"] = cleaned_df["clicks"].fillna(clicks_median)

print("Missing clicks after cleaning:",cleaned_df["clicks"].isnull().sum())
print("checking all missing values are handled")
print(cleaned_df.isnull().sum())

numeric_columns = ["spend","conversions","revenue","clicks","impressions"]

for column in numeric_columns:
    negative_count = (cleaned_df[column] < 0).sum()

    print(f"{column}: {negative_count} negative values")

    invalid_clicks = (cleaned_df["clicks"] > cleaned_df["impressions"]).sum()

print("Rows where clicks > impressions:",invalid_clicks)

invalid_conversions = (cleaned_df["conversions"] > cleaned_df["clicks"]).sum()

print("Rows where conversions > clicks:",invalid_conversions)

zero_revenue = (cleaned_df["revenue"] == 0).sum()
print("Rows with zero revenue:", zero_revenue)

zero_spend = (cleaned_df["spend"] == 0).sum()
print("Rows with zero spend:", zero_spend)

invalid_conversion_rows = cleaned_df[cleaned_df["conversions"] > cleaned_df["clicks"]]
print(invalid_conversion_rows[["date","campaign","campaign_type","clicks","conversions","revenue"]])

categorical_columns = ["campaign","campaign_type"]

for column in categorical_columns:
    cleaned_df[column] = cleaned_df[column].astype("category")

count_columns = ["conversions","clicks","impressions"]
for column in count_columns:
    cleaned_df[column] = cleaned_df[column].round().astype(int)

print(cleaned_df.dtypes)

OUTPUT_PATH = "data/processed/cleaned_revenue.csv"

cleaned_df.to_csv(OUTPUT_PATH,index=False)

print("Cleaned dataset saved successfully.")
