import os
import json
import base64
import logging
from concurrent.futures import ThreadPoolExecutor
import uuid

import cv2
from openai import OpenAI
from inference.config import SKU_CATALOG

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger("cuantotengo.gpt")

OPENAI_BRAND_MODEL = os.getenv("OPENAI_BRAND_MODEL", "gpt-5.2")
BRAND_GPT_MODE = os.getenv("BRAND_GPT_MODE", "batch") # batch | parallel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def encode_crop(crop):
	_, buffer = cv2.imencode(".jpg", crop)
	b64 = base64.b64encode(buffer).decode("utf-8")
	return f"data:image/jpeg;base64,{b64}"

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
 
def _new_request_id():
    return uuid.uuid4().hex[:8]
 
def _truncate(text, limit=1000):
    if text is None:
        return None
    return text if len(text) <= limit else text[:limit] + f"... [+{len(text) - limit} chars]"

def _normalize(sku):
    return sku.lower() if sku in SKU_OPTIONS else UNKNOWN


log.info("GPT brand detection ready | model=%s mode=%s skus=%d", OPENAI_BRAND_MODEL, BRAND_GPT_MODE, len(SKU_LIST))

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
 
 
def llm_detect_brands_batch(crops):
    """Clasifica todos los recortes en una sola petición. Devuelve una lista alineada con crops."""
    if not crops:
        return []
 
    content = [{"type": "input_text", "text": BATCH_PROMPT.format(sku_list=SKU_LIST_TEXT)}]
    for i, crop in enumerate(crops):
        content.append({"type": "input_text", "text": f"Product {i}:"})
        content.append({"type": "input_image", "image_url": encode_crop(crop)})
    
    req_id = _new_request_id()
    try:
        response = client.responses.create(
            model=OPENAI_BRAND_MODEL,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "shelf_brands",
                    "schema": _batch_schema(),
                    "strict": True,
                }
            },
        )
        print(f"[{req_id}] raw output: {_truncate(response.output_text)}")
        items = json.loads(response.output_text)["items"]
    except Exception as e:
        log.warning("Brand batch request failed: %s", e)
        return [UNKNOWN] * len(crops)
 
    by_id = {item["id"]: item["sku"] for item in items}
    return [_normalize(by_id.get(i, UNKNOWN)) for i in range(len(crops))]
 
 
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
 
 
def llm_detect_brand(crop, nearby_crops):
    print(f"Using OPENAI_BRAND_MODEL: {OPENAI_BRAND_MODEL}")
    content = [
        {"type": "input_text", "text": SINGLE_PROMPT.format(sku_list=SKU_LIST_TEXT)},
        {"type": "input_image", "image_url": encode_crop(crop)},
    ]
    if nearby_crops:
        content.append({"type": "input_text", "text": "Nearby products for context:"})
        content.extend({"type": "input_image", "image_url": encode_crop(c)} for c in nearby_crops)
 
    req_id = _new_request_id()
    try:
        response = client.responses.create(
            model=OPENAI_BRAND_MODEL,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "single_brand",
                    "schema": _single_schema(),
                    "strict": True,
                }
            },
        )
        print(f"[{req_id}] raw output: {_truncate(response.output_text)}")
        return _normalize(json.loads(response.output_text)["sku"])
    except Exception as e:
        log.warning("Brand request failed: %s", e)
        return UNKNOWN
 
 
def llm_detect_brands_parallel(crops, nearby_for_index):
    """nearby_for_index(i) -> lista de recortes vecinos del producto i."""
    if not crops:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(crops))) as pool:
        futures = [pool.submit(llm_detect_brand, c, nearby_for_index(i)) for i, c in enumerate(crops)]
        return [f.result() for f in futures]
 
 
# ============================================================
# LLAMADA PRINCIPAL: elige batch o paralelo según la variable de entorno
# ============================================================
def llm_detect_brands(crops, nearby_for_index=None):
    print(f"Using BRAND_GPT_MODE: {BRAND_GPT_MODE}")
    if BRAND_GPT_MODE == "parallel":
        return llm_detect_brands_parallel(crops, nearby_for_index or (lambda i: []))
    return llm_detect_brands_batch(crops)