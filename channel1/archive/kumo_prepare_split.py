import pandas as pd

df = pd.read_csv("kumo_fire_event_table.csv")

# Time-based split
train_df = df[df["year"] <= 2019].copy()
test_df = df[df["year"] == 2020].copy()

# Set test labels to None (VERY IMPORTANT)
test_df["label_next_month"] = None

# Combine back together
final_df = pd.concat([train_df, test_df]).sort_values(
    ["county_id", "year", "month"]
)

final_df.to_csv("kumo_fire_event_full_graph.csv", index=False)

print("Train rows:", len(train_df))
print("Test rows:", len(test_df))
print("Full graph ready for Kumo.")