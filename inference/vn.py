import cv2
from ultralytics import YOLO
import torch
import concurrent.futures
import time
import logging

from inference.detection.cap_detection import detect_caps
from inference.detection.front_detection import detect_and_match_fronts, detect_front_bottles
from inference.detection.local_brand_detection import get_brands_from_image, match_brands_to_bottles
from inference.detection.column_detection import (
    match_and_extend_columns,
    compute_intersections,
    cluster_vanishing_points,
    check_misaligned_columns,
    correct_misaligned_columns
)
from inference.final_aggregation import (
    match_front_caps_to_bottles,
    compute_brand_counts,
    match_gpt_output_to_list
)
from inference.image_utils import count_caps_per_column
from inference.config import standard_drinks, CAP_MODEL_PATH, FRONT_BOTTLE_MODEL_PATH

log = logging.getLogger("cuantotengo")

# --------------------------
# GPU detection (auto)
# --------------------------
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # mac silicon
        return "mps"
    return "cpu"

DEVICE = get_device()
print("🔥 Using device:", DEVICE)

# --------------------------
# Load YOLO models ONCE
# --------------------------
cap_model = YOLO(CAP_MODEL_PATH)
front_model = YOLO(FRONT_BOTTLE_MODEL_PATH)

cap_model.to(DEVICE)
front_model.to(DEVICE)

# --------------------------
# Resize for speed
# --------------------------
def resize_for_inference(image, max_size=1280):
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return image
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# --------------------------
# FAST inference
# --------------------------
def run_inference(image, sender_phone=None):

    timings = {}

    # --------------------------
    # STEP 0: Resize BEFORE YOLO
    # --------------------------
    t0 = time.time()
    image = resize_for_inference(image, max_size=1280)
    timings["resize"] = time.time() - t0

    # --------------------------
    # STEP 1: YOLO in parallel (GPU only)
    # --------------------------
    t0 = time.time()

    if DEVICE != "cpu":
        # Run both YOLO in parallel using GPU
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_caps = executor.submit(cap_model.predict, image, conf=0.25, max_det=200)
            future_front = executor.submit(front_model.predict, image, conf=0.25, max_det=200)

            caps_pred = future_caps.result()
            front_pred = future_front.result()

    else:
        # Fallback: CPU runs sequentially (threads don't help)
        caps_pred = cap_model.predict(image, conf=0.25, max_det=200)
        front_pred = front_model.predict(image, conf=0.25, max_det=200)

    timings["YOLO_parallel"] = time.time() - t0

    # --------------------------
    # STEP 2: Use predictions
    # --------------------------
    t0 = time.time()
    cap_data = detect_caps(image, caps_pred)
    front_bottles = detect_front_bottles(image, front_pred)
    timings["postprocess_caps/front"] = time.time() - t0

    # --------------------------
    # STEP 3: Main detection
    # --------------------------
    t0 = time.time()
    front_boxes, all_bottles, front_caps, all_caps, annotated_image = \
        detect_and_match_fronts(image, cap_data, front_pred)
    timings["detect_and_match_fronts"] = time.time() - t0

    # --------------------------
    # STEP 4: Column detection
    # --------------------------
    t0 = time.time()
    column_lines = match_and_extend_columns(image, front_caps, all_caps)
    intersections = compute_intersections(column_lines)
    vanishing_x, vanishing_y, labels, clustering = cluster_vanishing_points(intersections)
    timings["column_detection"] = time.time() - t0

    # --------------------------
    # STEP 5: Correction
    # --------------------------
    t0 = time.time()
    misaligned_columns = check_misaligned_columns(column_lines, vanishing_x, vanishing_y)
    corrected_lines = correct_misaligned_columns(
        column_lines, misaligned_columns, vanishing_x, vanishing_y, front_caps
    )
    timings["column_correction"] = time.time() - t0

    # --------------------------
    # STEP 6: Brands
    # --------------------------
    t0 = time.time()
    brands_list = get_brands_from_image(front_bottles, image)
    timings["brand_detection"] = time.time() - t0

    t0 = time.time()
    clean_brands = []
    for brand in brands_list:
        item, gpt_brand, gpt_flavor = brand.split(" - ")
        clean_brand = match_gpt_output_to_list(gpt_brand, gpt_flavor, standard_drinks)
        clean_brands.append(clean_brand)
    timings["brand_cleaning"] = time.time() - t0

    # --------------------------
    # STEP 7: Final aggregation
    # --------------------------
    t0 = time.time()
    bottle_brand_mapping = match_brands_to_bottles(front_bottles, clean_brands)
    front_cap_to_bottle = match_front_caps_to_bottles(front_bottles, cap_data)
    cap_counts = count_caps_per_column(image, corrected_lines, cap_data)
    brand_totals, lane_totals = compute_brand_counts(
        bottle_brand_mapping, front_cap_to_bottle, cap_counts, standard_drinks
    )
    timings["aggregation"] = time.time() - t0

    # PRINT TIMINGS
    print("\n=== Timing Summary ===")
    for k, v in timings.items():
        print(f"{k:<25} {v:.3f}s")
        
    print(f"\nTotal time: {sum(timings.values()):.3f}s")

    return brand_totals, annotated_image, cap_data, front_bottles, bottle_brand_mapping, lane_totals
