import pandas as pd

df = pd.read_csv("california_county_fire_final_ml_dataset.csv")

# -------- COUNTY TABLE --------
county_df = df[["NAME"]].drop_duplicates().reset_index(drop=True)
county_df["county_id"] = county_df.index

county_df = county_df[["county_id", "NAME"]]
county_df.to_csv("kumo_county_table.csv", index=False)

# -------- FIRE EVENT TABLE --------
df = df.merge(county_df, left_on="NAME", right_on="NAME")

fire_df = df[[
    "county_id",
    "year",
    "month",
    "fire_count",
    "avg_frp",
    "fire_last_1",
    "fire_last_2",
    "frp_last_1",
    "month_sin",
    "month_cos",
    "fire_rolling_3",
    "label_next_month"
]]

fire_df.to_csv("kumo_fire_event_table.csv", index=False)

print("Relational tables ready for KumoRFM.")