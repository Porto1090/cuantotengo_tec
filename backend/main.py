"""
CuantoTengo — Backend API (FastAPI)
CODED BY: @porto1090 - Eduardo Porto
"""

import os
import re
import cv2
import gc
import torch
import numpy as np
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference.config import MX_CLASS_NAMES_DICT, LAB_CLASS_NAMES_DICT

from dotenv import load_dotenv
load_dotenv()

BRAND_DETECTION_VERSION = os.getenv("BRAND_DETECTION_VERSION", "v1.0")

BRAND_MAP = {
    **MX_CLASS_NAMES_DICT,
    **LAB_CLASS_NAMES_DICT,
    "unknown": "No identificado",
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
        torch.set_num_threads(1)
        from inference.main_inference import InferencePipeline
        _pipeline = InferencePipeline()
    return _pipeline


# === HELPERS ===
def preprocess_image(image_bytes: bytes, max_size: int = 640, jpeg_quality: int = 65):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    del np_arr
    
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
    original_image_url: str | None = None
    annotated_image_url: str | None = None
    processing_time: float | None = None
    error: str | None = None


# === ENDPOINTS ===

@app.get("/api/health")
def health():
    return {"status": "ok", "version": BRAND_DETECTION_VERSION}


@app.post("/api/session/enter", response_model=SessionResponse)
def enter_session(payload: SessionRequest):
    sid = (payload.session_id).strip().upper()

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

def save_session_images(session_id: str, img_bgr, annotated_img) -> tuple[str | None, str | None]:
    try:
        timestamp = datetime.now(UTC_MINUS_6).strftime("%Y%m%d_%H%M%S")
        orig_filename = f"original/{session_id}_{timestamp}_original.jpg"
        anno_filename = f"annotated/{session_id}_{timestamp}_annotated.jpg"
        
        orig_path = os.path.join(images_dir, orig_filename)
        anno_path = os.path.join(images_dir, anno_filename)
        
        cv2.imwrite(orig_path, img_bgr)
        cv2.imwrite(anno_path, annotated_img)
        
        return f"/static/images/{orig_filename}", f"/static/images/{anno_filename}"
    except Exception as e:
        print(f"[WARNING] No se pudieron guardar las imágenes localmente: {e}")
        return None, None

@app.post("/api/process", response_model=ProcessResponse)
async def process_image(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):    
    try:
        image_bytes = await file.read()
        img_bgr = preprocess_image(image_bytes)
    except Exception:
        return ProcessResponse(ok=False, error="No se pudo decodificar la imagen. Vuelve a intentarlo.")
    finally:
        del image_bytes

    try:
        pipeline = get_pipeline()
        with torch.no_grad():
            ( brand_totals, annotated, _, _, _, _, processing_time ) = pipeline.run(img_bgr)
    except Exception as e:
        gc.collect()
        return ProcessResponse(ok=False, error=f"Fallo en inferencia: {str(e)}")
    
    if not brand_totals:
        gc.collect()
        return ProcessResponse(ok=False, error="No se detectaron productos en la imagen. Vuelve a intentarlo.")

    original_image_url, annotated_image_url = save_session_images(session_id, img_bgr, annotated)

    del annotated, img_bgr
    gc.collect()
    
    rows = build_brand_rows(brand_totals)
    print("-> FIN. ¡Proceso completado con éxito!")
    
    # Forzar recolección de basura post-inferencia
    gc.collect()
    return ProcessResponse(ok=True, rows=rows, annotated_image_url=annotated_image_url, original_image_url=original_image_url, processing_time=processing_time)