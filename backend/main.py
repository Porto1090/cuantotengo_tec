"""
CuantoTengo — Backend API (FastAPI)
CODED BY: @porto1090 - Eduardo Porto
"""

import os
import re
import uuid
import logging
import threading
from datetime import datetime, timezone, timedelta
 
import cv2
import numpy as np
 
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
 
from inference.config import MX_CLASS_NAMES_DICT, LAB_CLASS_NAMES_DICT
from inference.scoring_client import ScoringError

from dotenv import load_dotenv
load_dotenv()


# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("cuantotengo")


# === ENVIRONMENT VARIABLES ===
BRAND_DETECTION_VERSION = os.getenv("BRAND_DETECTION_VERSION", "v1.0")

BRAND_MAP = {
    **MX_CLASS_NAMES_DICT,
    **LAB_CLASS_NAMES_DICT,
    "unknown": "No identificado",
}

SESSION_REGEX = re.compile(r"^[0-9]{1,3}$")
UTC_MINUS_6 = timezone(timedelta(hours=-6))


# === SERVER ===
from contextlib import asynccontextmanager

_pipeline = None
_pipeline_lock = threading.Lock()

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from inference.main_inference import InferencePipeline
                _pipeline = InferencePipeline()
    return _pipeline

@asynccontextmanager
async def lifespan(app):
    get_pipeline()
    yield

app = FastAPI(title="CuantoTengo API", lifespan=lifespan)

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
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR)

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

def save_session_images(session_id: str, request_id: str, img_bgr, annotated_img) -> tuple[str | None, str | None]:
    """Guarda original y anotada. El request_id evita sobrescrituras dentro del mismo segundo."""
    try:
        timestamp = datetime.now(UTC_MINUS_6).strftime("%Y%m%d_%H%M%S")
        base_name = f"{session_id}_{timestamp}_{request_id}"
        orig_filename = f"original/{base_name}_original.jpg"
        anno_filename = f"annotated/{base_name}_annotated.jpg"
 
        ok_orig = cv2.imwrite(os.path.join(images_dir, orig_filename), img_bgr)
        ok_anno = cv2.imwrite(os.path.join(images_dir, anno_filename), annotated_img)
        if not (ok_orig and ok_anno):
            log.warning("[%s] cv2.imwrite returned False (orig=%s anno=%s)", request_id, ok_orig, ok_anno)
            return None, None
 
        return f"/static/images/{orig_filename}", f"/static/images/{anno_filename}"
    except Exception as e:
        log.warning("[%s] No se pudieron guardar las imágenes: %s", request_id, e)
        return None, None


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


@app.post("/api/process", response_model=ProcessResponse)
async def process_image(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):    
    request_id = uuid.uuid4().hex[:8]
    log.info("[%s] /api/process session=%s file=%s", request_id, session_id, file.filename)
    
    sid = (session_id or "").strip()
    if not (SESSION_REGEX.match(sid) or sid == "000"):
        return ProcessResponse(ok=False, error="ID de sesión inválido")
    
    # 1. Leer y preprocesar la imagen
    try:
        image_bytes = await file.read()
        img_bgr = await run_in_threadpool(preprocess_image, image_bytes)
    except Exception:
        log.warning("[%s] No se pudo decodificar la imagen", request_id)
        return ProcessResponse(ok=False, error="No se pudo decodificar la imagen. Vuelve a intentarlo.")
 
    # 2. Pipeline en un hilo aparte: hace llamadas HTTP síncronas (scorer y GPT)
    #    y así no bloquea el event loop de FastAPI.
    try:
        pipeline = get_pipeline()
        brand_totals, annotated, _, _, _, _, processing_time = await run_in_threadpool(
            pipeline.run, img_bgr, request_id
        )
    except ScoringError as e:
        log.error("[%s] Scoring error: %s", request_id, e)
        return ProcessResponse(ok=False, error="El servicio de detección no respondió. Vuelve a intentarlo.")
    except Exception as e:
        log.exception("[%s] Fallo en inferencia", request_id)
        return ProcessResponse(ok=False, error=f"Fallo en inferencia: {str(e)}")
 
    if not brand_totals:
        log.info("[%s] Sin productos detectados", request_id)
        return ProcessResponse(ok=False, error="No se detectaron productos en la imagen. Vuelve a intentarlo.")
 
    # 3. Guardar imágenes y armar la respuesta
    original_image_url, annotated_image_url = await run_in_threadpool(
        save_session_images, session_id, request_id, img_bgr, annotated
    )
    rows = build_brand_rows(brand_totals)
 
    log.info("[%s] OK en %.2fs | %s", request_id, processing_time, brand_totals)
    return ProcessResponse(
        ok=True,
        rows=rows,
        annotated_image_url=annotated_image_url,
        original_image_url=original_image_url,
        processing_time=processing_time,
    )