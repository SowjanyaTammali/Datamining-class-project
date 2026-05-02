import pandas as pd

# Load county monthly wildfire data
df = pd.read_csv("california_county_fire_monthly_2015_2020.csv")

# Sort properly
df = df.sort_values(by=["NAME", "year", "month"])

# Create next-month fire_count column
df["next_month_fire"] = df.groupby("NAME")["fire_count"].shift(-1)

# Drop rows where next month does not exist
df = df.dropna(subset=["next_month_fire"])

# ---- NEW STRONGER LABEL ----
# Using 75th percentile threshold (high wildfire month)

threshold = df["fire_count"].quantile(0.75)
print("Using threshold:", threshold)

df["label_next_month"] = df["next_month_fire"].apply(
    lambda x: 1 if x >= threshold else 0
)

# Save final dataset
df.to_csv("california_county_fire_prediction_dataset.csv", index=False)

# Check balance
print("\nLabel Distribution:")
print(df["label_next_month"].value_counts())

print("\nDataset ready for modeling.")