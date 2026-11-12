import pandas as pd

INPUT_PATH="data/processed/featured_dataset.csv"
df= pd.read_csv(INPUT_PATH)
print(df.columns)
feature_columns=[
    "date","campaign","campaign_type","spend","conversions","clicks","impressions","CTR","Conversion_Rate","CPC","CPA","ROAS","Revenue_Per_Conversion","Year","Month","Day_of_Week","Quarter"
]

x=df[feature_columns]
y=df["revenue"]

print(y.head(10))
print(x.shape)
print(y.shape)

