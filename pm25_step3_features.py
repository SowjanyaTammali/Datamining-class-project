"""
Channel II - Step 3: Feature Engineering for PM2.5 prediction
Input:  california_pm25_county_monthly.csv
Output: california_pm25_final_ml_dataset.csv

Features mirror Channel I's temporal structure (lags, rolling, seasonal)
so both channels are comparable and mergeable.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR   = "data/processed"
OUTPUT_DIR = "data/processed"

df = pd.read_csv(f"{DATA_DIR}/california_pm25_county_monthly.csv")
df = df.sort_values(["NAME", "year", "month"]).reset_index(drop=True)

# ── TEMPORAL FEATURES ────────────────────────────────────────────────────────
grp = df.groupby("NAME")

# Lag features (within each county)
df["pm25_lag1"]     = grp["avg_pm25"].shift(1)   # prev month
df["pm25_lag2"]     = grp["avg_pm25"].shift(2)   # 2 months ago
df["pm25_rolling3"] = grp["avg_pm25"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
df["days_above_lag1"] = grp["days_above_35"].shift(1)

# Seasonal encoding (same as Channel I)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# ── TARGET: next-month avg PM2.5 (regression) ────────────────────────────────
df["next_month_pm25"] = grp["avg_pm25"].shift(-1)

# ── CLASSIFICATION TARGET: will next month exceed EPA standard? ──────────────
# EPA 24-hr PM2.5 standard = 35.4 µg/m³
# We flag months where avg exceeds a meaningful threshold
threshold = df["avg_pm25"].quantile(0.75)
print(f"PM2.5 75th percentile threshold: {threshold:.2f} µg/m³")

df["label_next_month_pm25"] = (df["next_month_pm25"] >= threshold).astype(int)

print(f"Label distribution:\n{df['label_next_month_pm25'].value_counts()}")

# Drop rows missing lag features or target
df = df.dropna(subset=["pm25_lag1", "next_month_pm25"])

out_path = f"{OUTPUT_DIR}/california_pm25_final_ml_dataset.csv"
df.to_csv(out_path, index=False)
print(f"\nFinal dataset shape: {df.shape}")
print(f"Saved → {out_path}")
