import os
from kumoai import KumoClient, Session

client = KumoClient(
    api_key=os.getenv("KUMO_API_KEY"),
    url="https://api.kumo.ai"
)

print("Connected.")

session = Session()
print("Session created:", session)