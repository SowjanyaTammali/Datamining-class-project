import os
from kumoai import KumoClient

client = KumoClient(
    api_key=os.getenv("KUMO_API_KEY"),
    url="https://api.kumo.ai"
)

print("Connected to Kumo successfully.")