"""
Compara modelos de OpenAI para la clasificación de marca con los MISMOS recortes.

Requiere el scorer corriendo (SCORING_URI en .env) para obtener los frentes.
Uso, desde backend/:
    python compare_brand_models.py foto.jpg \
        --models gpt-5.2 gpt-5.4-mini gpt-5.4-nano --runs 3
"""
import argparse
import logging
import statistics

import cv2
from dotenv import load_dotenv
load_dotenv()

from inference.scoring_client import ScoringClient

try:
    from inference.detection.gpt_detection import classify_batch
except ImportError:
    from inference.gpt_detection import classify_batch

logging.basicConfig(level=logging.WARNING)


def preprocess(path, max_size=640, quality=65):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"No se pudo leer {path}")
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--models", nargs="+", default=["gpt-5.2", "gpt-5.4-mini", "gpt-5.4-nano"])
    p.add_argument("--effort", default="none", help="none | low | medium | high | '' para el default")
    p.add_argument("--detail", default="auto", help="auto | low | high")
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()

    image = preprocess(args.image)
    _, fronts, _ = ScoringClient().detect(image, request_id="compare")
    crops = [image[int(f["y1"]):int(f["y2"]), int(f["x1"]):int(f["x2"])] for f in fronts]
    print(f"{len(crops)} frentes detectados\n")

    effort = args.effort or None
    report = []
    for model in args.models:
        times, tokens_in, tokens_out, outputs, errors = [], [], [], [], 0
        for r in range(args.runs):
            results, meta = classify_batch(crops, model=model, effort=effort,
                                           detail=args.detail, req_id=f"{model}-{r}")
            if "error" in meta:
                errors += 1
                print(f"  {model} corrida {r}: ERROR {meta['error'][:120]}")
                continue
            times.append(meta["seconds"])
            tokens_in.append(meta.get("in") or 0)
            tokens_out.append(meta.get("out") or 0)
            outputs.append(tuple(results))
        report.append((model, times, tokens_in, tokens_out, outputs, errors))

    print(f"{'modelo':<16}{'mediana':>9}{'min':>8}{'max':>8}{'tok in':>9}{'tok out':>9}{'estable':>9}{'errores':>9}")
    for model, times, tin, tout, outputs, errors in report:
        if not times:
            print(f"{model:<16}{'—':>9}{'—':>8}{'—':>8}{'—':>9}{'—':>9}{'—':>9}{errors:>9}")
            continue
        stable = "sí" if len(set(outputs)) == 1 else "no"
        print(f"{model:<16}{statistics.median(times):>8.2f}s{min(times):>7.2f}s{max(times):>7.2f}s"
              f"{int(statistics.median(tin)):>9}{int(statistics.median(tout)):>9}{stable:>9}{errors:>9}")

    print("\nMarcas por producto (primera corrida de cada modelo):")
    for model, _, _, _, outputs, _ in report:
        if outputs:
            print(f"  {model:<16} {list(outputs[0])}")


if __name__ == "__main__":
    main()