import pandas as pd

# Load all 6 years
files = [
    "viirs-snpp_2015_United_States.csv",
    "viirs-snpp_2016_United_States.csv",
    "viirs-snpp_2017_United_States.csv",
    "viirs-snpp_2018_United_States.csv",
    "viirs-snpp_2019_United_States.csv",
    "viirs-snpp_2020_United_States.csv"
]

# Combine all years
df_list = [pd.read_csv(file) for file in files]
df = pd.concat(df_list, ignore_index=True)

print("Total rows:", len(df))
print(df.head())

# Filter California bounding box
ca_df = df[
    (df['latitude'] >= 32) &
    (df['latitude'] <= 42) &
    (df['longitude'] >= -125) &
    (df['longitude'] <= -114)
]

print("California rows:", len(ca_df))

# Convert date column to datetime
ca_df['acq_date'] = pd.to_datetime(ca_df['acq_date'])

# Extract year and month
ca_df['year'] = ca_df['acq_date'].dt.year
ca_df['month'] = ca_df['acq_date'].dt.month

# Aggregate monthly fire counts
monthly_fire = ca_df.groupby(['year', 'month']).agg(
    fire_count=('latitude', 'count'),
    avg_frp=('frp', 'mean')
).reset_index()

print(monthly_fire.head())
print("Total monthly rows:", len(monthly_fire))

# Save cleaned California fire points
ca_df.to_csv("california_fire_points_2015_2020.csv", index=False)

# Save monthly aggregated data
monthly_fire.to_csv("california_fire_monthly_2015_2020.csv", index=False)

print("Files saved successfully.")