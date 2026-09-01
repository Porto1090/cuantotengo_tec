"""
CuantoTengo — Backend API (FastAPI) - Memory Optimized
------------------------------------------------------
Optimizado para mantenerse bajo el límite de 512MB RAM en Render.
"""

import os
import re
import cv2
import gc
import string
import secrets
import numpy as np
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import azure_blob_storage as azure_bs
from inference.config import MX_CLASS_NAMES_DICT, LAB_CLASS_NAMES_DICT

from dotenv import load_dotenv
load_dotenv()

BRAND_DETECTION_VERSION = os.getenv("BRAND_DETECTION_VERSION", "v1.0")

BRAND_MAP = {
    **MX_CLASS_NAMES_DICT,
    **LAB_CLASS_NAMES_DICT,
}

SESSION_REGEX = re.compile(r"^[0-9]{1,3}$")
UTC_MINUS_6 = timezone(timedelta(hours=-6))

app = FastAPI(title="CuantoTengo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

images_dir = os.path.join(os.path.dirname(__file__), "images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir, exist_ok=True)

app.mount("/static/images", StaticFiles(directory=images_dir), name="brand-images")

_pipeline = None


def get_pipeline():
    """Lazy import and loading to minimize startup memory overhead."""
    global _pipeline
    if _pipeline is None:
        import torch
        torch.set_num_threads(1)  # Limita hilos de CPU para reducir pico de RAM
        from inference.main_inference import InferencePipeline
        _pipeline = InferencePipeline()
    return _pipeline


# === HELPERS ===

def generate_short_hex() -> str:
    letters = [secrets.choice(string.ascii_uppercase) for _ in range(2)]
    numbers = [secrets.choice(string.digits) for _ in range(4)]
    return "".join(letters + numbers)


def preprocess_image(image_bytes: bytes, max_size: int = 640, jpeg_quality: int = 65):
    """Reducido a 640px max para evitar picos de RAM en OpenCV/NumPy."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    del np_arr  # Liberar buffer original inmediatamente
    
    if img is None:
        raise ValueError("Error: No se logró decodificar la imagen.")

    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, jpg = cv2.imencode(".jpg", img, encode_param)
    del img
    
    decoded_img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    del jpg
    return decoded_img


def build_brand_rows(brand_totals: dict) -> list[dict]:
    rows = []
    for key, value in brand_totals.items():
        pretty_name = (BRAND_MAP.get(key, key) or key).replace("_", " ").title()
        image_slug = key.replace("_", " ").replace(" ", "_")
        rows.append({
            "key": key,
            "marca": pretty_name,
            "total": value,
            "image_url": f"/static/images/{image_slug}.jpg",
        })
    rows.sort(key=lambda r: r["marca"])
    return rows


# === SCHEMAS ===

class SessionRequest(BaseModel):
    session_id: str


class SessionResponse(BaseModel):
    ok: bool
    session_id: str | None = None
    enter_time: str | None = None
    error: str | None = None


class ProcessResponse(BaseModel):
    ok: bool
    rows: list[dict] | None = None
    image_url: str | None = None
    processing_time: float | None = None
    error: str | None = None


# === ENDPOINTS ===

@app.get("/api/health")
def health():
    return {"status": "ok", "version": BRAND_DETECTION_VERSION}


@app.post("/api/session/enter", response_model=SessionResponse)
def enter_session(payload: SessionRequest):
    sid = (payload.session_id or "").strip().upper()

    if not sid:
        return SessionResponse(ok=False, error="ID requerido")

    if not SESSION_REGEX.match(sid):
        return SessionResponse(ok=False, error="ID inválido (1-999)")

    enter_time = datetime.now(UTC_MINUS_6)
    return SessionResponse(ok=True, session_id=sid, enter_time=enter_time.isoformat())


@app.post("/api/session/test", response_model=SessionResponse)
def enter_test_session():
    enter_time = datetime.now(UTC_MINUS_6)
    return SessionResponse(ok=True, session_id="000", enter_time=enter_time.isoformat())


@app.post("/api/process", response_model=ProcessResponse)
async def process_image(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    enter_time: str | None = Form(None),
):
    import torch

    elapsed_time = None
    if enter_time:
        try:
            enter_dt = datetime.fromisoformat(enter_time)
            elapsed_time = (datetime.now(UTC_MINUS_6) - enter_dt).total_seconds()
        except ValueError:
            elapsed_time = None

    try:
        image_bytes = await file.read()
        img_bgr = preprocess_image(image_bytes)
        del image_bytes
    except Exception:
        return ProcessResponse(ok=False, error="No se pudo decodificar la imagen. Vuelve a intentarlo.")

    try:
        pipeline = get_pipeline()
        
        # Ejecutar inferencia desactivando cálculo de gradientes
        with torch.no_grad():
            (
                brand_totals, annotated, cap_data, front_bottles,
                bottle_brand_mapping, lane_totals, processing_time,
            ) = pipeline.run(img_bgr)

    except Exception as e:
        gc.collect()
        return ProcessResponse(ok=False, error=f"Fallo en inferencia: {str(e)}")

    if not brand_totals:
        gc.collect()
        return ProcessResponse(ok=False, error="No se detectaron productos en la imagen. Vuelve a intentarlo.")

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    del annotated
    del img_bgr

    real_timestamp = datetime.now(UTC_MINUS_6).replace(microsecond=0).isoformat()[:-6]

    # === GUARDADO EN AZURE BLOB STORAGE ===
    try:
        azure_bs.save_image_to_blob(img_rgb, session_id, real_timestamp + "_original", BRAND_DETECTION_VERSION + "_images")
        azure_bs.save_image_to_blob(annotated_rgb, session_id, real_timestamp + "_bounding_boxes", BRAND_DETECTION_VERSION + "_images")

        log_dict = {
            "session_id": session_id,
            "version": BRAND_DETECTION_VERSION,
            "timestamp": real_timestamp,
            "brand_totals": brand_totals,
            "processing_time": f"{processing_time:.2f}",
            "elapsed_time_in_session": f"{elapsed_time:.2f}" if elapsed_time is not None else None,
        }
        azure_bs.save_log_to_blob(log_dict, session_id, real_timestamp, BRAND_DETECTION_VERSION + "_logs")
        image_url = azure_bs.get_blob_url(session_id, real_timestamp + "_bounding_boxes", BRAND_DETECTION_VERSION + "_images")
    except Exception:
        image_url = None

    del img_rgb
    del annotated_rgb
    
    rows = build_brand_rows(brand_totals)
    
    # Forzar recolección de basura post-inferencia
    gc.collect()

    return ProcessResponse(ok=True, rows=rows, image_url=image_url, processing_time=processing_time)