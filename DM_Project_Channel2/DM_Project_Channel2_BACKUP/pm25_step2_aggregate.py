"""
Channel II - Step 2: Aggregate daily PM2.5 to county-month
Input:  california_pm25_daily_2015_2020.csv
Output: california_pm25_county_monthly.csv

Aggregation mirrors Channel I's county-month granularity so both
channels can be merged on (NAME, year, month).
"""

import pandas as pd
import os

DATA_DIR   = "data/processed"
OUTPUT_DIR = "data/processed"

df = pd.read_csv(f"{DATA_DIR}/california_pm25_daily_2015_2020.csv",
                 parse_dates=["date"])

# ── COUNTY-MONTH AGGREGATION ─────────────────────────────────────────────────
monthly = (
    df.groupby(["NAME", "year", "month"])
    .agg(
        avg_pm25        = ("pm25",  "mean"),
        max_pm25        = ("pm25",  "max"),
        std_pm25        = ("pm25",  "std"),
        days_above_35   = ("pm25",  lambda x: (x > 35.4).sum()),  # EPA 24hr standard
        obs_count       = ("pm25",  "count"),
    )
    .reset_index()
)

monthly["std_pm25"] = monthly["std_pm25"].fillna(0)

print(f"County-months: {len(monthly):,}")
print(f"Counties: {monthly['NAME'].nunique()}")
print(f"Years: {sorted(monthly['year'].unique())}")
print(monthly.head())

out_path = f"{OUTPUT_DIR}/california_pm25_county_monthly.csv"
monthly.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
