import requests
import pandas as pd
import os

SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
VM_NAME = os.getenv("AZURE_VM_NAME")
TOKEN = os.getenv("AZURE_TOKEN")

url = f"""
https://management.azure.com/subscriptions/
{SUBSCRIPTION_ID}/resourceGroups/{RESOURCE_GROUP}
/providers/Microsoft.Compute/virtualMachines/{VM_NAME}
/providers/microsoft.insights/metrics
"""

params = {
    "metricnames": "Percentage CPU,Network In Total,Disk Read Bytes",
    "timespan": "P30D",
    "interval": "PT1M",
    "api-version": "2018-01-01"
}

headers = {
    "Authorization": f"Bearer {TOKEN}"
}

response = requests.get(
    url,
    params=params,
    headers=headers
)

print(response.json())