"""
CuantoTengo — Scoring de YOLO para Azure ML Managed Online Endpoint.

Contiene únicamente la inferencia de los dos modelos YOLO (tapas y frentes).
Toda la lógica de negocio (columnas, marcas, conteo, anotación) vive en la API.

Contrato
--------
Entrada (JSON):
    {"image_b64": "<jpeg en base64>", "request_id": "opcional"}
Salida (JSON):
    {"ok": true, "request_id": "...", "caps": [...], "fronts": [...],
     "image_size": [w, h], "model": {...}, "inference_ms": {...}}
"""
import os
import json
import time
import base64
import logging

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("cuantotengo.scoring")
log.setLevel(logging.INFO)
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _handler.formatter.converter = time.gmtime
    log.addHandler(_handler)
    log.propagate = False

# --- Configuración (se define en deployment.yml) ---
CAP_MODEL_FILE = os.getenv("CAP_MODEL_FILE", "bottle_can_cap_yolo_weights.pt")
FRONT_MODEL_FILE = os.getenv("FRONT_MODEL_FILE", "bottlefront_weights.pt")
CAP_CONF = float(os.getenv("CAP_DETECTION_CONFIDENCE", "0.25"))
FRONT_CONF = float(os.getenv("FRONT_DETECTION_CONFIDENCE", "0.5"))
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "2"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))

MODELS = {}
MODEL_INFO = {}


def _model_dir():
    """
    Azure ML monta el modelo registrado en AZUREML_MODEL_DIR.
    Al registrar la carpeta 'yolo', los .pt quedan en AZUREML_MODEL_DIR/yolo.
    En local se usa ../model como respaldo.
    """
    base = os.getenv("AZUREML_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "model"))
    sub = os.path.join(base, "yolo")
    return sub if os.path.isdir(sub) else base


def _registered_model_info():
    """AZUREML_MODEL_DIR -> .../azureml-models/<nombre>/<versión>."""
    path = os.getenv("AZUREML_MODEL_DIR", "")
    parts = [p for p in path.split(os.sep) if p]
    if len(parts) >= 2 and "azureml-models" in parts:
        return {"name": parts[-2], "version": parts[-1]}
    return {"name": "local", "version": "dev"}


def init():
    """Se ejecuta una sola vez al arrancar cada worker: carga y calienta los modelos."""
    t0 = time.perf_counter()
    torch.set_num_threads(TORCH_THREADS)

    model_dir = _model_dir()
    log.info("Loading YOLO models from %s (threads=%d)", model_dir, TORCH_THREADS)

    MODELS["cap"] = YOLO(os.path.join(model_dir, CAP_MODEL_FILE))
    MODELS["front"] = YOLO(os.path.join(model_dir, FRONT_MODEL_FILE))

    # Calentamiento: la primera inferencia siempre es más lenta
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    with torch.inference_mode():
        MODELS["cap"].predict(dummy, conf=CAP_CONF, verbose=False)
        MODELS["front"].predict(dummy, conf=FRONT_CONF, verbose=False)

    MODEL_INFO.update(_registered_model_info())
    MODEL_INFO["files"] = {"cap": CAP_MODEL_FILE, "front": FRONT_MODEL_FILE}
    log.info("Models ready in %.2fs | %s", time.perf_counter() - t0, MODEL_INFO)


def _to_boxes(results, with_class):
    boxes_out = []
    for result in results:
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int) if with_class else None
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            box = {
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "confidence": float(confs[i]),
            }
            if with_class:
                box["class"] = int(classes[i])
            boxes_out.append(box)
    return boxes_out


def _error(request_id, message):
    log.warning("[%s] %s", request_id, message)
    return {"ok": False, "request_id": request_id, "error": message}


def run(raw_data):
    t_start = time.perf_counter()

    try:
        payload = json.loads(raw_data)
    except (TypeError, json.JSONDecodeError):
        return _error(None, "Request body must be JSON")

    request_id = payload.get("request_id")
    image_b64 = payload.get("image_b64")
    if not image_b64:
        return _error(request_id, "Missing field: image_b64")

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return _error(request_id, "image_b64 is not valid base64")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return _error(request_id, f"Image too large: {len(image_bytes)} bytes")

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return _error(request_id, "Could not decode image")

    h, w = image.shape[:2]

    with torch.inference_mode():
        t0 = time.perf_counter()
        caps = _to_boxes(MODELS["cap"].predict(image, conf=CAP_CONF, verbose=False), with_class=True)
        t1 = time.perf_counter()
        fronts = _to_boxes(MODELS["front"].predict(image, conf=FRONT_CONF, verbose=False), with_class=False)
        t2 = time.perf_counter()

    # Mismo orden que detect_front_bottles: de izquierda a derecha
    fronts.sort(key=lambda b: b["x1"])

    timings = {
        "cap": round((t1 - t0) * 1000, 1),
        "front": round((t2 - t1) * 1000, 1),
        "total": round((time.perf_counter() - t_start) * 1000, 1),
    }
    log.info("[%s] %dx%d caps=%d fronts=%d ms=%s", request_id, w, h, len(caps), len(fronts), timings)

    return {
        "ok": True,
        "request_id": request_id,
        "caps": caps,
        "fronts": fronts,
        "image_size": [w, h],
        "model": MODEL_INFO,
        "inference_ms": timings,
    }
