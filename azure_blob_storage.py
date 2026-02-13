import os
import cv2
import json
from azure.storage.blob import BlobServiceClient

from dotenv import load_dotenv
load_dotenv()

AZURE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER")

blob_client = None

if AZURE_CONN_STR and AZURE_CONTAINER:
    blob_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)

def save_image_to_blob(image_rgb, session_id, metadata, prefix="images"):
    if blob_client is None:
        # Storage disabled (DEV / testing)
        return None
    img_bytes = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))[1].tobytes()

    blob_name = f"{prefix}/{session_id}_{metadata}.jpg"
    blob = blob_client.get_blob_client(container=AZURE_CONTAINER, blob=blob_name)

    blob.upload_blob(img_bytes, overwrite=True)

    return blob.url


def save_log_to_blob(log_dict, session_id, metadata, prefix="logs"):
    if blob_client is None:
        # Storage disabled (DEV / testing)
        return None
    blob_name = f"{prefix}/{session_id}_{metadata}.json"
    print(f"Saving log to Azure Blob Storage: {log_dict}")
    blob = blob_client.get_blob_client(container=AZURE_CONTAINER, blob=blob_name)

    blob.upload_blob(json.dumps(log_dict, indent=2), overwrite=True)

    return blob.url
