import os
import json
import time
import uuid
import base64
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
from openai import OpenAI
from inference.config import SKU_CATALOG

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger("cuantotengo.gpt")


# --- Configuración por variables de entorno ---
OPENAI_BRAND_MODEL = os.getenv("OPENAI_BRAND_MODEL", "gpt-5.4-mini")
BRAND_GPT_MODE = os.getenv("BRAND_GPT_MODE", "batch")             # batch | parallel
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT")     # none | low | medium | high (vacío = default del modelo)
OPENAI_IMAGE_DETAIL = os.getenv("OPENAI_IMAGE_DETAIL", "auto")     # auto | low | high
OPENAI_TIMEOUT_S = float(os.getenv("OPENAI_TIMEOUT_S", "20"))

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=OPENAI_TIMEOUT_S,
    max_retries=1,
)

UNKNOWN = "unknown"

def build_sku_list():
    skus = []
    for brand, flavors in SKU_CATALOG.items():
        for flavor in flavors:
            skus.append(f"{brand} {flavor}")
    return skus

SKU_LIST = build_sku_list()
SKU_OPTIONS = SKU_LIST + [UNKNOWN]
SKU_LIST_TEXT = "\n".join(f"- {s}" for s in SKU_LIST)

log.info("GPT brand detection ready | model=%s mode=%s skus=%d effort=%s detail=%s",
         OPENAI_BRAND_MODEL, BRAND_GPT_MODE, len(SKU_LIST),
         OPENAI_REASONING_EFFORT or "model default", OPENAI_IMAGE_DETAIL)

def encode_crop(crop):
    _, buffer = cv2.imencode(".jpg", crop)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

def _image_part(crop, detail):
    return {"type": "input_image", "image_url": encode_crop(crop), "detail": detail}

def _new_request_id():
    return uuid.uuid4().hex[:8]

def _truncate(text, limit=1000):
    if text is None:
        return None
    return text if len(text) <= limit else text[:limit] + f"... [+{len(text) - limit} chars]"

def _normalize(sku):
    return sku.lower() if sku in SKU_OPTIONS else UNKNOWN

def _usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    out_details = getattr(usage, "output_tokens_details", None)
    return {
        "in": getattr(usage, "input_tokens", None),
        "out": getattr(usage, "output_tokens", None),
        "reasoning": getattr(out_details, "reasoning_tokens", None) if out_details else None,
    }

def _create(req_id, model, effort, content, schema_name, schema):
    """Llamada común: aplica modelo, esfuerzo de razonamiento y registra latencia y tokens."""
    kwargs = {}
    if effort:
        kwargs["reasoning"] = {"effort": effort}

    t0 = time.perf_counter()
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
        **kwargs,
    )
    elapsed = time.perf_counter() - t0
    usage = _usage(response)
    log.info("[%s] %s effort=%s %.2fs tokens in=%s out=%s reasoning=%s",
             req_id, model, effort or "default", elapsed,
             usage.get("in"), usage.get("out"), usage.get("reasoning"))
    log.debug("[%s] raw output: %s", req_id, _truncate(response.output_text))
    return response, {"model": model, "effort": effort, "seconds": elapsed, **usage}


# ============================================================
# MODO A: una sola llamada para todas las botellas
# ============================================================
BATCH_PROMPT = """You are a product recognition system for beverage shelves.
You will receive several cropped images, each labeled "Product N".
Each crop shows the front-facing product of one shelf column.

For EACH product, choose exactly one SKU from this list:
{sku_list}

Rules:
- If the product is not in the list, or you cannot tell, answer "unknown".
- Do not guess a similar brand: a different brand is "unknown".
- Neighboring products can help with context, but classify each one on its own.
- Return one entry per product id."""

def _batch_schema():
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "sku": {"type": "string", "enum": SKU_OPTIONS},
                    },
                    "required": ["id", "sku"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

def classify_batch(crops, model=None, effort=None, detail=None, req_id=None):
    """
    Versión con metadatos (útil para comparar modelos).
    Devuelve (resultados alineados con crops, meta con latencia y tokens).
    """
    model = model or OPENAI_BRAND_MODEL
    effort = effort if effort is not None else OPENAI_REASONING_EFFORT
    detail = detail or OPENAI_IMAGE_DETAIL
    req_id = req_id or _new_request_id()

    if not crops:
        return [], {"model": model, "effort": effort, "seconds": 0.0}

    content = [{"type": "input_text", "text": BATCH_PROMPT.format(sku_list=SKU_LIST_TEXT)}]
    for i, crop in enumerate(crops):
        content.append({"type": "input_text", "text": f"Product {i}:"})
        content.append(_image_part(crop, detail))

    try:
        response, meta = _create(req_id, model, effort, content, "shelf_brands", _batch_schema())
        items = json.loads(response.output_text)["items"]
    except Exception as e:
        log.warning("[%s] Brand batch request failed (%s): %s", req_id, model, e)
        return [UNKNOWN] * len(crops), {"model": model, "effort": effort, "error": str(e)}

    by_id = {item["id"]: item["sku"] for item in items}
    results = [_normalize(by_id.get(i, UNKNOWN)) for i in range(len(crops))]
    log.info("[%s] brands=%s", req_id, results)
    return results, meta

def llm_detect_brands_batch(crops):
    results, _ = classify_batch(crops)
    return results


# ============================================================
# MODO B: una llamada por botella, todas en paralelo
# ============================================================
SINGLE_PROMPT = """You are a product recognition system.
Identify the beverage SKU in the main image.

Choose exactly one option from this list:
{sku_list}

Rules:
- If the product is not in the list, or you cannot tell, answer "unknown".
- Do not guess a similar brand: a different brand is "unknown".

Main product image:"""

def _single_schema():
    return {
        "type": "object",
        "properties": {"sku": {"type": "string", "enum": SKU_OPTIONS}},
        "required": ["sku"],
        "additionalProperties": False,
    }

def llm_detect_brand(crop, nearby_crops, req_id=None):
    req_id = req_id or _new_request_id()
    content = [
        {"type": "input_text", "text": SINGLE_PROMPT.format(sku_list=SKU_LIST_TEXT)},
        _image_part(crop, OPENAI_IMAGE_DETAIL),
    ]
    if nearby_crops:
        content.append({"type": "input_text", "text": "Nearby products for context:"})
        content.extend(_image_part(c, OPENAI_IMAGE_DETAIL) for c in nearby_crops)

    try:
        response, _ = _create(req_id, OPENAI_BRAND_MODEL, OPENAI_REASONING_EFFORT,
                              content, "single_brand", _single_schema())
        return _normalize(json.loads(response.output_text)["sku"])
    except Exception as e:
        log.warning("[%s] Brand request failed: %s", req_id, e)
        return UNKNOWN

def llm_detect_brands_parallel(crops, nearby_for_index):
    if not crops:
        return []
    group = _new_request_id()
    with ThreadPoolExecutor(max_workers=min(8, len(crops))) as pool:
        futures = [pool.submit(llm_detect_brand, c, nearby_for_index(i), f"{group}-p{i}")
                   for i, c in enumerate(crops)]
        return [f.result() for f in futures]


# ============================================================
# LLAMADA PRINCIPAL
# ============================================================
def llm_detect_brands(crops, nearby_for_index=None):
    if BRAND_GPT_MODE == "parallel":
        return llm_detect_brands_parallel(crops, nearby_for_index or (lambda i: []))
    return llm_detect_brands_batch(crops)