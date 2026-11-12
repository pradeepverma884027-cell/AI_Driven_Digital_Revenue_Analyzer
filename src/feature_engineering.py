import pandas as pd
import numpy as np

DATASET_PATH="data/processed/cleaned_revenue.csv"

cleaned_df=pd.read_csv(DATASET_PATH)
cleaned_df["date"] = pd.to_datetime(cleaned_df["date"])

#CTR - click through rate
cleaned_df["CTR"] = np.where(cleaned_df["impressions"] > 0,(cleaned_df["clicks"] / cleaned_df["impressions"]) * 100,0)

#conversion rate
cleaned_df["Conversion_Rate"] = np.where(cleaned_df["clicks"] > 0,(cleaned_df["conversions"] /cleaned_df["clicks"]) * 100,0)

#cost per click
cleaned_df["CPC"] = np.where(cleaned_df["clicks"] > 0,cleaned_df["spend"] / cleaned_df["clicks"],0)

#cost per acquistion
cleaned_df["CPA"] = np.where(cleaned_df["conversions"] > 0,cleaned_df["spend"] / cleaned_df["conversions"],0)

#return on ad spend
cleaned_df["ROAS"] = np.where(cleaned_df["spend"] > 0,cleaned_df["revenue"] / cleaned_df["spend"],0)

#revenue per conversion
cleaned_df["Revenue_Per_Conversion"] = np.where(cleaned_df["conversions"] > 0,cleaned_df["revenue"] /cleaned_df["conversions"],0)

#handling  date features
cleaned_df["Year"] = cleaned_df["date"].dt.year
cleaned_df["Month"] = cleaned_df["date"].dt.month
cleaned_df["Day_of_Week"] = cleaned_df["date"].dt.dayofweek

cleaned_df["Quarter"] = cleaned_df["date"].dt.quarter

print(cleaned_df.head(10))

OUTPUT_PATH="data/processed/featured_dataset.csv"
cleaned_df.to_csv(OUTPUT_PATH,index=False)

print("Feature engineering completed successfully.")

print(cleaned_df.columns)

