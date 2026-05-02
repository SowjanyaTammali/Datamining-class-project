"""
Channel II - Step 1: Load EPA PM2.5 data and filter California
Input:  daily_88101_2015.zip ... daily_88101_2020.zip
Output: california_pm25_daily_2015_2020.csv
"""

import pandas as pd
import zipfile
import os
import glob

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR   = "data/raw"          # folder where you placed the 6 zip files
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLS_NEEDED = [
    "State Name", "County Name", "Latitude", "Longitude",
    "Date Local", "Arithmetic Mean", "AQI"
]

# ── LOAD & CONCAT ────────────────────────────────────────────────────────────
frames = []
for year in range(2015, 2021):
    # support both naming conventions
    patterns = [
        f"{DATA_DIR}/daily_88101_{year}.zip",
        f"{DATA_DIR}/*daily_88101_{year}.zip",
    ]
    matched = []
    for p in patterns:
        matched.extend(glob.glob(p))

    if not matched:
        print(f"WARNING: No file found for {year}")
        continue

    zip_path = matched[0]
    print(f"Loading {zip_path} ...")
    with zipfile.ZipFile(zip_path) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(z.open(csv_name), usecols=COLS_NEEDED, low_memory=False)

    df["year"] = year
    frames.append(df)
    print(f"  {year}: {len(df):,} rows loaded")

df_all = pd.concat(frames, ignore_index=True)
print(f"\nTotal rows (all states): {len(df_all):,}")

# ── FILTER CALIFORNIA ────────────────────────────────────────────────────────
df_ca = df_all[df_all["State Name"] == "California"].copy()
df_ca = df_ca.rename(columns={
    "Date Local":      "date",
    "Arithmetic Mean": "pm25",
    "County Name":     "NAME"
})
df_ca["date"] = pd.to_datetime(df_ca["date"], format='mixed')
df_ca["month"] = df_ca["date"].dt.month

# Drop rows with missing PM2.5
df_ca = df_ca.dropna(subset=["pm25"])
print(f"California rows: {len(df_ca):,}")
print(df_ca.head(3))

out_path = f"{OUTPUT_DIR}/california_pm25_daily_2015_2020.csv"
df_ca.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
