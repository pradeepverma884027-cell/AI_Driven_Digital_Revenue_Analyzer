import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = "data/processed/featured_dataset.csv"
df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

print("Dataset loaded successfully.")
campaign_revenue = (
    df.groupby("campaign")["revenue"]
    .sum()
    .sort_values(ascending=False))
plt.figure(figsize=(10, 6))
campaign_revenue.plot(kind="bar")
plt.title("Revenue by Campaign")
plt.xlabel("Campaign")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/revenue_by_campaign.png")
plt.show()
plt.close()


plt.figure(figsize=(8, 6))
plt.scatter(df["spend"],df["revenue"])
plt.title("Ad Spend vs Revenue")
plt.xlabel("Ad Spend")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("outputs/spend_vs_revenue.png")
plt.show()
plt.close()


monthly_revenue = (df.groupby("Month")["revenue"].sum())
plt.figure(figsize=(10, 6))
monthly_revenue.plot(kind="line", marker="o")
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig("outputs/monthly_revenue.png")
plt.show()
plt.close()

type_revenue = (df.groupby("campaign_type")["revenue"].sum().sort_values(ascending=False))
plt.figure(figsize=(10, 6))
type_revenue.plot(kind="bar")
plt.title("Revenue by Campaign Type")
plt.xlabel("Campaign Type")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/revenue_by_campaign_type.png")
plt.show()
plt.close()


campaign_roas = (df.groupby("campaign").agg(Revenue=("revenue", "sum"),Spend=("spend", "sum")))
campaign_roas["ROAS"] = (campaign_roas["Revenue"] /campaign_roas["Spend"])
campaign_roas = campaign_roas["ROAS"].sort_values(ascending=False)
plt.figure(figsize=(10, 6))
campaign_roas.plot(kind="bar")
plt.title("ROAS by Campaign")
plt.xlabel("Campaign")
plt.ylabel("ROAS")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/roas_by_campaign.png")
plt.show()
plt.close()


numeric_columns = [
    "spend",
    "conversions",
    "revenue",
    "clicks",
    "impressions",
    "CTR",
    "Conversion_Rate",
    "CPC",
    "CPA","ROAS","Revenue_Per_Conversion"]

correlation = df[numeric_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation,annot=True,fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.show()
plt.close()






