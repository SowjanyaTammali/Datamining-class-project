import os
from kumoai import (
    init,
    FileUploadConnector,
    SourceTable,
    Edge,
    PredictiveQuery,
)

# --------------------------------------------------
# 1️⃣ Initialize SDK (THIS IS CRITICAL)
# --------------------------------------------------

init(
    init(
    init(
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1YTAzNGJiM2MwMTgyMDhhNzdmMWU1YzJlN2UzZTIwYiIsImp0aSI6ImU0MjU3YTE5LTI1NjMtNDNhNi05NTU2LWRkZDViYmI2ZjE3MCIsImlhdCI6MTc3NzU3NzAyMCwiZXhwIjoxNzgyNzYxMDIwfQ.BUYhKgvd7oOd4cKw2EZf0WAl6uYpPdjDsTNWKf3aJfI",
    url="https://kumorfm.ai"
)
)
)

print("Kumo SDK initialized successfully.")

# --------------------------------------------------
# 2️⃣ Create CSV Connectors
# --------------------------------------------------

county_connector = FileUploadConnector(file_type="csv")
fire_connector = FileUploadConnector(file_type="csv")

# --------------------------------------------------
# 3️⃣ Upload Tables
# --------------------------------------------------

county_connector.upload(
    name="county_table",
    path="kumo_county_table.csv"
)

fire_connector.upload(
    name="fire_event_table",
    path="kumo_fire_event_full_graph.csv"
)

print("Files uploaded successfully.")

# --------------------------------------------------
# 4️⃣ Register Source Tables
# --------------------------------------------------

county_table = SourceTable(
    name="County",
    connector=county_connector
)

fire_table = SourceTable(
    name="FireEvent",
    connector=fire_connector
)

print("Source tables registered.")

# --------------------------------------------------
# 5️⃣ Create Relationship
# --------------------------------------------------

edge = Edge(
    from_table="County",
    to_table="FireEvent",
    from_key="county_id",
    to_key="county_id"
)

print("Graph relationship created.")

# --------------------------------------------------
# 6️⃣ Predictive Query
# --------------------------------------------------

query = PredictiveQuery(
    target_table="FireEvent",
    target_column="label_next_month",
)

print("Running prediction...")

result = query.run()

print(result.head())