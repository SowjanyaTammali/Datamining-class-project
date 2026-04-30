import pandas as pd
import numpy as np

# Load labeled dataset
df = pd.read_csv("california_county_fire_prediction_dataset.csv")

# Sort correctly
df = df.sort_values(by=["NAME", "year", "month"])

# ---- LAG FEATURES ----
df["fire_last_1"] = df.groupby("NAME")["fire_count"].shift(1)
df["fire_last_2"] = df.groupby("NAME")["fire_count"].shift(2)
df["frp_last_1"] = df.groupby("NAME")["avg_frp"].shift(1)

# ---- SEASONAL (CYCLICAL) FEATURES ----
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# ---- ROLLING AVERAGE (3 months) ----
df["fire_rolling_3"] = (
    df.groupby("NAME")["fire_count"]
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)

# Drop rows with missing values from lag/rolling
df = df.dropna()

# Save improved dataset
df.to_csv("california_county_fire_final_ml_dataset.csv", index=False)

print(df.head())
print("\nFinal dataset with seasonality ready for ML.")