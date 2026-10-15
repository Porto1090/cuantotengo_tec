"""
Cliente del endpoint de scoring (Azure ML) que ejecuta los modelos YOLO.

Variables de entorno:
    SCORING_URI          https://<endpoint>.<region>.inference.ml.azure.com/score
    SCORING_KEY          llave primaria del endpoint
    SCORING_DEPLOYMENT   opcional: fuerza un despliegue concreto (p. ej. "blue")
    SCORING_TIMEOUT_S    opcional, por defecto 30
"""
import os
import json
import time
import uuid
import base64
import logging

import cv2
import httpx

log = logging.getLogger("cuantotengo.scoring")


class ScoringError(RuntimeError):
    pass


class ScoringClient:
    def __init__(self):
        self.url = os.environ["SCORING_URI"]
        self.key = os.environ["SCORING_KEY"]
        self.deployment = os.getenv("SCORING_DEPLOYMENT")
        timeout = float(os.getenv("SCORING_TIMEOUT_S", "30"))
        # Cliente reutilizable: mantiene la conexión abierta entre peticiones
        self.http = httpx.Client(timeout=timeout)
        log.info("Scoring client ready | url=%s deployment=%s timeout=%.0fs",
                 self.url, self.deployment or "default", timeout)

    def detect(self, image_bgr, request_id=None):
        """
        Envía la imagen al endpoint y devuelve (caps, fronts, meta).
        caps y fronts tienen el mismo formato que detect_caps y detect_front_bottles.
        """
        request_id = request_id or uuid.uuid4().hex[:8]

        ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise ScoringError("Could not encode image")
        payload = {"image_b64": base64.b64encode(buf).decode("utf-8"), "request_id": request_id}

        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        if self.deployment:
            headers["azureml-model-deployment"] = self.deployment

        t0 = time.perf_counter()
        try:
            response = self.http.post(self.url, json=payload, headers=headers)
        except httpx.TimeoutException as e:
            log.error("[%s] scoring timeout after %.2fs", request_id, time.perf_counter() - t0)
            raise ScoringError("Scoring endpoint timeout") from e
        except httpx.HTTPError as e:
            log.error("[%s] scoring connection error: %s", request_id, e)
            raise ScoringError("Scoring endpoint unreachable") from e
        elapsed = time.perf_counter() - t0

        if response.status_code != 200:
            log.error("[%s] scoring HTTP %d in %.2fs: %s",
                      request_id, response.status_code, elapsed, response.text[:500])
            raise ScoringError(f"Scoring endpoint returned HTTP {response.status_code}")

        data = response.json()
        if isinstance(data, str):  # algunas versiones devuelven el JSON como string
            data = json.loads(data)

        if not data.get("ok"):
            log.error("[%s] scoring error: %s", request_id, data.get("error"))
            raise ScoringError(data.get("error", "Unknown scoring error"))

        log.info("[%s] scoring %.2fs (server %s ms) caps=%d fronts=%d model=%s:%s",
                 request_id, elapsed, data.get("inference_ms", {}).get("total"),
                 len(data["caps"]), len(data["fronts"]),
                 data.get("model", {}).get("name"), data.get("model", {}).get("version"))

        meta = {
            "request_id": request_id,
            "network_s": elapsed,
            "inference_ms": data.get("inference_ms"),
            "model": data.get("model"),
        }
        return data["caps"], data["fronts"], meta
