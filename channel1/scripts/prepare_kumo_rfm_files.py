import pandas as pd

df = pd.read_csv("kumo_fire_event_full_graph.csv")

# Create real timestamp column from year/month
df["event_date"] = pd.to_datetime(
    df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
)

# Save updated file
df.to_csv("kumo_fire_event_rfm.csv", index=False)

print(df.head())
print("Saved: kumo_fire_event_rfm.csv")