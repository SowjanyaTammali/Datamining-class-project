import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load cleaned California fire points
df = pd.read_csv("california_fire_points_2015_2020.csv")

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
gdf_fire = gpd.GeoDataFrame(df, geometry=geometry)

# Load US counties shapefile
counties = gpd.read_file("tl_2020_us_county/tl_2020_us_county.shp")

# Filter California counties only (STATEFP = '06')
ca_counties = counties[counties['STATEFP'] == '06']

# Ensure coordinate systems match
gdf_fire = gdf_fire.set_crs(ca_counties.crs)

# Spatial join: assign each fire point to a county
fire_with_county = gpd.sjoin(
    gdf_fire,
    ca_counties,
    how="inner",
    predicate="within"
)

print("Assigned counties successfully.")
print(fire_with_county[['NAME', 'acq_date']].head())

# Extract year and month
fire_with_county['acq_date'] = pd.to_datetime(fire_with_county['acq_date'])
fire_with_county['year'] = fire_with_county['acq_date'].dt.year
fire_with_county['month'] = fire_with_county['acq_date'].dt.month

# Aggregate monthly by county
county_monthly = fire_with_county.groupby(
    ['NAME', 'year', 'month']
).agg(
    fire_count=('latitude', 'count'),
    avg_frp=('frp', 'mean')
).reset_index()

county_monthly.to_csv(
    "california_county_fire_monthly_2015_2020.csv",
    index=False
)

print("County-level monthly dataset saved.")