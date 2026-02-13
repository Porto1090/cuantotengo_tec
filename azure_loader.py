import os
import json
import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

AZURE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER")

blob_service = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
container_client = blob_service.get_container_client(AZURE_CONTAINER)

def list_all_blobs(prefix=None):
    blobs = []

    for blob in container_client.list_blobs(name_starts_with=prefix):
        blobs.append(blob.name)

    return blobs
   
def list_blobs_by_session(session_id, prefix=None):
    blobs = []

    for blob in container_client.list_blobs(name_starts_with=prefix):
        if f"{session_id}_" in blob.name:
            blobs.append(blob.name)

    return blobs

def load_logs_by_session(session_id):
    logs = []

    blob_names = list_blobs_by_session(session_id, prefix="logs/")

    for name in blob_names:
        blob = container_client.get_blob_client(name)
        raw = blob.download_blob().readall()
        logs.append(json.loads(raw))

    return logs

def load_images_by_session(session_id):
    images = []

    blob_names = list_blobs_by_session(session_id, prefix="images/")

    for name in blob_names:
        blob = container_client.get_blob_client(name)
        img_bytes = blob.download_blob().readall()

        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append({
            "blob_name": name,
            "image": img
        })

    return images

def save_images_to_route(session_id, output_dir="./saved_images"):
    images = load_images_by_session(session_id)
    session_dir = os.path.join(output_dir, session_id)
    
    os.makedirs(session_dir, exist_ok=True)
    
    for img_data in images:
        filename = os.path.basename(img_data["blob_name"])
        filepath = os.path.join(session_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(img_data["image"], cv2.COLOR_RGB2BGR))
    
    return session_dir

def load_full_session(session_id):
    images_route = save_images_to_route(session_id)
    
    return {
        "session_id": session_id,
        "logs": load_logs_by_session(session_id),
        "images_saved_to": images_route
    }
    
# print(load_full_session("0F5319"))
# print(list_all_blobs())

