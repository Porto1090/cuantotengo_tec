import os
import time
import io
import zipfile
import requests
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

API_KEY = os.getenv("QUALTRICS_API_KEY")
DATA_CENTER = os.getenv("QUALTRICS_DATA_CENTER")
SURVEY_ID = os.getenv("QUALTRICS_SURVEY_ID")

if not all([API_KEY, DATA_CENTER, SURVEY_ID]):
    raise RuntimeError("Faltan variables en el .env")

BASE_URL = f"https://{DATA_CENTER}.qualtrics.com/API/v3"

HEADERS = {
    "X-API-TOKEN": API_KEY,
    "Content-Type": "application/json"
}

# Iniciar exportación
def start_export():
    url = f"{BASE_URL}/surveys/{SURVEY_ID}/export-responses"

    payload = {
        "format": "csv",
        "useLabels": True
    }

    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()

    return response.json()["result"]["progressId"]

# Esperar a que termine
def wait_for_export(progress_id):
    url = f"{BASE_URL}/surveys/{SURVEY_ID}/export-responses/{progress_id}"

    while True:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        result = response.json()["result"]

        status = result["status"]

        if status == "complete":
            return result["fileId"]
        elif status == "failed":
            raise RuntimeError("La exportación falló")

        time.sleep(2)

# Descargar y extraer ZIP
def download_and_extract(file_id, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    url = f"{BASE_URL}/surveys/{SURVEY_ID}/export-responses/{file_id}/file"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(output_dir)

# Main
if __name__ == "__main__":
    print("Iniciando exportación de Qualtrics...")

    progress_id = start_export()
    file_id = wait_for_export(progress_id)
    download_and_extract(file_id)

    print("Proceso finalizado")
