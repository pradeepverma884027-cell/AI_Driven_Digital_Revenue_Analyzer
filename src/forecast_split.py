import pandas as pd

INPUT_PATH = "data/processed/forecasting_features.csv"

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


split_index = int(len(df) * 0.80)

train = df.iloc[:split_index].copy()

test = df.iloc[split_index:].copy()


train.to_csv(TRAIN_PATH, index=False)

test.to_csv(TEST_PATH, index=False)

print("Time-based split completed.")

print("\nTotal rows:", len(df))

print("Training rows:", len(train))

print("Testing rows:", len(test))

print("\nTraining period:")
print(train["date"].min(), "→", train["date"].max())

print("\nTesting period:")
print(test["date"].min(), "→", test["date"].max())