import pandas as pd
import numpy as np

# Load original ML dataset with true labels
df = pd.read_csv("california_county_fire_final_ml_dataset.csv")

print("Original columns:", df.columns.tolist())

# Load county lookup table
county = pd.read_csv("kumo_county_table.csv")

print("County table columns:", county.columns.tolist())

# If county_id is missing, merge using county name
if "county_id" not in df.columns:
    df = df.merge(county, on="NAME", how="left")

# Check if merge worked
if df["county_id"].isna().any():
    print("Some counties did not match:")
    print(df[df["county_id"].isna()]["NAME"].unique())
    raise ValueError("County ID merge failed.")

# Create real timestamp column
df["event_date"] = pd.to_datetime(
    df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
)

# Add unique row id
df = df.reset_index(drop=True)
df["row_id"] = df.index

# Save actual full file with true labels
df.to_csv("kumo_fire_event_rfm_actual.csv", index=False)

# Create hidden version for Kumo upload
hidden = df.copy()
hidden.loc[hidden["year"] == 2020, "label_next_month"] = np.nan

hidden.to_csv("kumo_fire_event_rfm_hidden.csv", index=False)

# Save 2020 true labels separately for evaluation
actual_2020 = df[df["year"] == 2020][
    ["row_id", "county_id", "NAME", "year", "month", "label_next_month"]
]

actual_2020.to_csv("kumo_actual_2020_labels.csv", index=False)

print("\nSaved:")
print("1. kumo_fire_event_rfm_actual.csv")
print("2. kumo_fire_event_rfm_hidden.csv")
print("3. kumo_actual_2020_labels.csv")

print("\n2020 label distribution:")
print(actual_2020["label_next_month"].value_counts())