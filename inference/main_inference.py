# TODO: Clean this pipeline up. It is not modular at all as of now, making it very, very hard to implement changes.

import cv2
from ultralytics import YOLO
import concurrent.futures
import torch

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
from services.whatsapp import send_message
from inference.config import standard_drinks
from inference.image_utils import count_caps_per_column
from inference.config import CAP_MODEL_PATH, FRONT_BOTTLE_MODEL_PATH

# test run time start
import time
import logging
log = logging.getLogger("cuantotengo")
# test run time end

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # mac silicon
        return "mps"
    return "cpu"

DEVICE = get_device()
print("USING DEVICE:", DEVICE)


print("Loading YOLO models...")
cap_model = YOLO(CAP_MODEL_PATH)
front_model = YOLO(FRONT_BOTTLE_MODEL_PATH)
cap_model.to(DEVICE)
front_model.to(DEVICE)
print("Models loaded.")


def run_inference(image, sender_phone=None):
    """Runs the full product detection pipeline and returns brand counts and annotated image."""

    # test run time start
    timings = {}
    # test run time end

    # 1. Identify product tops (aka caps)

    # test run time start
    t0 = time.time()
    # test run time end

    #send_message(sender_phone, "Counting products...")
    # cap_model = YOLO(CAP_MODEL_PATH)
    if DEVICE != "cpu":
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_caps = executor.submit(detect_caps, image, cap_model)
            future_front = executor.submit(detect_front_bottles, image, front_model)

            cap_data = future_caps.result()
            front_bottles = future_front.result()

    else:
        cap_data = detect_caps(image, cap_model)
        front_bottles = detect_front_bottles(image, front_model)

    # test run time start
    timings["parallel_caps_and_front_bottles"] = time.time() - t0
    # test run time end

    # 2. Identify lanes (aka columns)

    # test run time start
    t0 = time.time()
    # test run time end

    #send_message(sender_phone, "Finding lanes...")
    # front_model = YOLO(FRONT_BOTTLE_MODEL_PATH)
    front_boxes, all_bottles, front_caps, all_caps, annotated_image = detect_and_match_fronts(
        image, cap_data, front_model
    )

    # test run time start
    timings["detect_and_match_fronts"] = time.time() - t0
    # test run time end

    # test run time start
    t0 = time.time()
    # test run time end

    column_lines = match_and_extend_columns(image, front_caps, all_caps)
    intersections = compute_intersections(column_lines)
    vanishing_x, vanishing_y, labels, clustering = cluster_vanishing_points(intersections)

    # test run time start
    timings["column_detection"] = time.time() - t0
    # test run time end

    # test run time start
    t0 = time.time()
    # test run time end    

    misaligned_columns = check_misaligned_columns(column_lines, vanishing_x, vanishing_y)
    corrected_lines = correct_misaligned_columns(
        column_lines, misaligned_columns, vanishing_x, vanishing_y, front_caps
    )

    # test run time start
    timings["column_correction"] = time.time() - t0
    # test run time end
    
    # 3. Identify products and brands for each column
    #send_message(sender_phone, "Identifying products...")

    # test run time start
    t0 = time.time()
    # test run time end

    brands_list = get_brands_from_image(front_bottles, image)

    # test run time start
    timings["brand_detection"] = time.time() - t0
    # test run time end      

    # test run time start
    t0 = time.time()
    # test run time end

    # gpt output splits
    # clean_brands = []
    # for brand in brands_list:
    #     # Split brand and product
    #     gpt_brand, gpt_flavor = brand.split(" - ", 1) if " - " in brand else (brand, "")
    #     clean_brand = match_gpt_output_to_list(gpt_brand, gpt_flavor, standard_drinks)
    #     clean_brands.append(clean_brand)

    # local classifier spplits
    clean_brands = []
    for brand in brands_list:
        # Correct 3-part split: item - brand - flavor
        item, gpt_brand, gpt_flavor = brand.split(" - ")
        clean_brand = match_gpt_output_to_list(gpt_brand, gpt_flavor, standard_drinks)
        clean_brands.append(clean_brand)



    # test run time start
    timings["brand_cleaning"] = time.time() - t0
    # test run time end        

    # test run time start
    t0 = time.time()
    # test run time end

    bottle_brand_mapping = match_brands_to_bottles(front_bottles, clean_brands)

    front_cap_to_bottle = match_front_caps_to_bottles(front_bottles, cap_data)
    cap_counts = count_caps_per_column(image, corrected_lines, cap_data)

    brand_totals, lane_totals = compute_brand_counts(
        bottle_brand_mapping, front_cap_to_bottle, cap_counts, standard_drinks
    )

    # test run time start
    timings["final_aggregation"] = time.time() - t0

    log.info("⏱ Timing summary (seconds): %s", timings)
    print("\n=== Timing Summary ===")
    for step, t in timings.items():
        print(f"{step:<25} {t:.3f}s")
    # test run time end        
    
    # total time completition
    processing_time = sum(timings.values())
    print(f"\nTotal time: {processing_time:.3f}s")

    return brand_totals, annotated_image, cap_data, front_bottles, bottle_brand_mapping, lane_totals, processing_time