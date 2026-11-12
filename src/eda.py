import pandas as pd

DATASET_PATH="data/processed/featured_dataset.csv"
df= pd.read_csv(DATASET_PATH)
df["date"] = pd.to_datetime(df["date"])
print(df.columns.tolist())
total_spend = df["spend"].sum()
total_revenue = df["revenue"].sum()
total_clicks = df["clicks"].sum()
total_conversions = df["conversions"].sum()
total_impressions = df["impressions"].sum()

overall_roas = total_revenue / total_spend
print("Total Impressions:", total_impressions)
print("Total Clicks:", total_clicks)
print("Total Conversions:", total_conversions)
print("Total Spend: ₹", round(total_spend, 2))
print("Total Revenue: ₹", round(total_revenue, 2))
print("Overall ROAS:", round(overall_roas, 2))

campaign_performance = (df.groupby("campaign").agg(Spend=("spend", "sum"),Revenue=("revenue", "sum"),Clicks=("clicks", "sum"),Conversions=("conversions", "sum")).reset_index())
campaign_performance["ROAS"] =(campaign_performance["Revenue"] /campaign_performance["Spend"])
print(campaign_performance.sort_values("Revenue", ascending=False))

best_revenue_campaign = campaign_performance.loc[campaign_performance["Revenue"].idxmax()]
print(best_revenue_campaign)

best_roas_campaign = campaign_performance.loc[campaign_performance["ROAS"].idxmax()]
print(best_roas_campaign)



campaign_type_performance = (df.groupby("campaign_type").agg(Spend=("spend", "sum"),Revenue=("revenue", "sum"),Clicks=("clicks", "sum"),Conversions=("conversions", "sum"),Impressions=("impressions", "sum")).reset_index())

campaign_type_performance["ROAS"] = (campaign_type_performance["Revenue"] /campaign_type_performance["Spend"])
campaign_type_performance["CTR"] = (campaign_type_performance["Clicks"] /campaign_type_performance["Impressions"]) * 100

campaign_type_performance["Conversion_Rate"] = (campaign_type_performance["Conversions"]/campaign_type_performance["Clicks"]) * 100


print(campaign_type_performance.sort_values("Revenue", ascending=False))

best_type_revenue = campaign_type_performance.loc[campaign_type_performance["Revenue"].idxmax()]
print(best_type_revenue)

best_type_roas = campaign_type_performance.loc[campaign_type_performance["ROAS"].idxmax()]
print(best_type_roas)

spend_revenue_correlation = df[["spend", "revenue"]].corr()
print(spend_revenue_correlation)

monthly_performance = (
    df.groupby("Month").agg(
        Spend=("spend", "sum"),
        Revenue=("revenue", "sum"),
        Conversions=("conversions", "sum"),
        Clicks=("clicks", "sum")).reset_index())

monthly_performance["ROAS"] = (monthly_performance["Revenue"]/monthly_performance["Spend"])
print(monthly_performance)

best_month = monthly_performance.loc[monthly_performance["Revenue"].idxmax()]
print(best_month)

