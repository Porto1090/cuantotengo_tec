"""
Prepara una petición de prueba con el mismo preprocesamiento que la API
(lado máximo 640 px, JPEG calidad 65) y opcionalmente la envía.

Generar el archivo para `az ml online-endpoint invoke`:
    python tools/make_request.py ../backend/images/pepsi_original.jpg -o sample-request.json

Enviar al servidor local (azmlinfsrv):
    python tools/make_request.py foto.jpg --url http://127.0.0.1:5001/score

Enviar al endpoint en Azure:
    python tools/make_request.py foto.jpg --url "$SCORING_URI" --key "$SCORING_KEY"
"""
import argparse
import base64
import json
import time

import cv2
import httpx


def preprocess(path, max_size=640, quality=65):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"No se pudo leer {path}")
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf).decode("utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("-o", "--output", help="Guardar el JSON de la petición")
    p.add_argument("--url")
    p.add_argument("--key")
    args = p.parse_args()

    payload = {"image_b64": preprocess(args.image), "request_id": "manual-test"}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(payload, f)
        print(f"Petición guardada en {args.output}")

    if args.url:
        headers = {"Content-Type": "application/json"}
        if args.key:
            headers["Authorization"] = f"Bearer {args.key}"
        t0 = time.perf_counter()
        r = httpx.post(args.url, json=payload, headers=headers, timeout=60)
        elapsed = time.perf_counter() - t0
        print(f"HTTP {r.status_code} en {elapsed:.2f}s")
        data = r.json()
        if isinstance(data, str):
            data = json.loads(data)
        summary = {k: v for k, v in data.items() if k not in ("caps", "fronts")}
        summary["caps"] = len(data.get("caps", []))
        summary["fronts"] = len(data.get("fronts", []))
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
