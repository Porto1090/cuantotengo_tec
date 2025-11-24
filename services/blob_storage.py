from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from datetime import datetime, timedelta
import numpy as np
import cv2

from keys import blob_keys

blob_service_client = BlobServiceClient.from_connection_string(blob_keys["connection_string"])
container_client = blob_service_client.get_container_client(blob_keys["container_name"])

# TODO: Eventually, this will probably have to be an API that receives the Whatsapp image in some format
def upload_local_image(local_path, blob_path):
    blob_client = container_client.get_blob_client(blob_path)
    with open(local_path, "rb") as img:
        blob_client.upload_blob(img)


def upload_image_to_blob(image_np: np.ndarray, blob_path: str):
    """Encodes a NumPy image array as PNG and uploads it to Azure Blob Storage.

    Args:
        image_np (np.ndarray): Image data as a NumPy array.
        blob_path (str): Path within the Azure Blob container where the image will be saved.

    Raises:
        ValueError: If the image cannot be encoded.
    """
    success, encoded_image = cv2.imencode(".png", image_np)
    if not success:
        raise ValueError("Failed to encode image for uploading.")

    image_bytes = encoded_image.tobytes()
    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(image_bytes, overwrite=True)


def generate_blob_url(blob_path: str, hours_valid: float = 0.1) -> str:
    """Generates a temporary SAS URL for accessing a blob in Azure Blob Storage.

    Args:
        blob_path (str): Path to the blob within the container.
        hours_valid (float, optional): Duration (in hours) for which the URL is valid. Default is 0.1 hours.

    Returns:
        str: A temporary URL with read access to the blob.
    """
    sas_token = generate_blob_sas(
        account_name=container_client.account_name,
        container_name=container_client.container_name,
        blob_name=blob_path,
        account_key=container_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=hours_valid)
    )
    blob_client = container_client.get_blob_client(blob_path)
    return f"{blob_client.url}?{sas_token}"


# TODO: There must be a better way to organize these funcitons
def upload_file(bytes, blob_path):
    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(bytes, overwrite=True)


def load_image(blob_path):
    """Loads an image from Azure Blob Storage into a NumPy array without saving it locally."""

    blob_client = container_client.get_blob_client(blob_path)
    blob_data = blob_client.download_blob().readall()
    image_array = np.asarray(bytearray(blob_data), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # Convert to OpenCV image
    return image