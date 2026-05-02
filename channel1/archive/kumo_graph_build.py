import geopandas as gpd

# Load county shapefile
gdf = gpd.read_file("tl_2020_us_county/tl_2020_us_county.shp")

# Filter California (STATEFP = 06)
ca = gdf[gdf["STATEFP"] == "06"].reset_index(drop=True)

ca = ca[["NAME", "geometry"]]

# Find neighbors (touching polygons)
edges = []

for i, county in ca.iterrows():
    for j, other in ca.iterrows():
        if i != j:
            if county["geometry"].touches(other["geometry"]):
                edges.append((county["NAME"], other["NAME"]))

print("Number of edges:", len(edges))