from azure.storage.blob import BlobServiceClient
import json
import cv2
import os

blob_client = BlobServiceClient.from_connection_string(
    os.environ["AZURE_STORAGE_CONNECTION_STRING"]
)
container = os.environ["AZURE_CONTAINER"]

def save_image_to_blob(image_rgb, session_id, timestamp, prefix="images"):
    img_bytes = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))[1].tobytes()

    blob_name = f"{prefix}/{session_id}_{timestamp}.jpg"
    blob = blob_client.get_blob_client(container=container, blob=blob_name)

    blob.upload_blob(img_bytes, overwrite=True)

    return blob.url


def save_log_to_blob(log_dict, session_id, timestamp, prefix="logs"):
    blob_name = f"{prefix}/{session_id}_{timestamp}.json"
    blob = blob_client.get_blob_client(container=container, blob=blob_name)

    blob.upload_blob(json.dumps(log_dict, indent=2), overwrite=True)

    return blob.url
