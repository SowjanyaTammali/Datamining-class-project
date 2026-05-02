import os
from kumo import Client

api_key = os.getenv("KUMO_API_KEY")

client = Client(api_key=api_key)

print("Kumo client initialized successfully.")