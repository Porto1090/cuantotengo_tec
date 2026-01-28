import time
import logging
import concurrent.futures as cf

import inference.config as cfg
from inference.runtime.device import get_device
from inference.runtime.models import load_models
from inference.detection.cap_detection import (
    detect_caps # 1
)
from inference.detection.front_detection import (
    detect_front_bottles, # 2
    detect_and_match_fronts # 3
)
from inference.detection.column_detection import (
    match_and_extend_columns, # 4
    compute_intersections, # 5 
    cluster_vanishing_points, # 6
    check_misaligned_columns, # 7
    correct_misaligned_columns # 8
)
from inference.detection.local_brand_detection import (
    get_brands_from_image, # 9
    match_brands_to_bottles # 11
)
from inference.final_aggregation import (
    match_gpt_output_to_list, # 10
    match_front_caps_to_bottles, # 12
    compute_brand_counts, # 14
)
from inference.image_utils import (
    count_caps_per_column, # 13
)

log = logging.getLogger("cuantotengo")

DEVICE = get_device()
print("USING DEVICE:", DEVICE)

cap_model, front_model = load_models(
    DEVICE,
    cfg.CAP_MODEL_PATH,
    cfg.FRONT_BOTTLE_MODEL_PATH
)
print("YOLO models loaded.")


def run_inference(image):
    """Runs the full product detection pipeline and returns brand counts and annotated image."""
    timings = {}

    # 1. Identify product tops (aka caps)
    t0 = time.time()

    if DEVICE != "cpu":
        with cf.ThreadPoolExecutor() as executor:
            future_caps = executor.submit(detect_caps, image, cap_model)
            future_front = executor.submit(detect_front_bottles, image, front_model)

            cap_data = future_caps.result()
            front_bottles = future_front.result()
    else:
        cap_data = detect_caps(image, cap_model)
        front_bottles = detect_front_bottles(image, front_model)

    timings["parallel_caps_and_front_bottles"] = time.time() - t0

    # 2. Identify lanes (aka columns)
    t0 = time.time()
    
    front_boxes, all_bottles, front_caps, all_caps, annotated_image = detect_and_match_fronts(
        image, cap_data, front_model
    )

    timings["detect_and_match_fronts"] = time.time() - t0

    # 3. Detect and compute column lines
    t0 = time.time()

    column_lines = match_and_extend_columns(image, front_caps, all_caps)
    intersections = compute_intersections(column_lines)
    vanishing_x, vanishing_y, labels, clustering = cluster_vanishing_points(intersections)

    timings["column_detection"] = time.time() - t0

    # 4. Correct misaligned columns
    t0 = time.time()

    misaligned_columns = check_misaligned_columns(column_lines, vanishing_x, vanishing_y)
    corrected_lines = correct_misaligned_columns(
        column_lines, misaligned_columns, vanishing_x, vanishing_y, front_caps
    )

    timings["column_correction"] = time.time() - t0
    
    # 5. Identify products and brands for each column
    t0 = time.time()

    brands_list = get_brands_from_image(front_bottles, image)

    timings["brand_detection"] = time.time() - t0

    # 6. Clean brand names using known product list
    t0 = time.time()
    
    clean_brands = []
    for brand in brands_list:
        # Correct 3-part split: item - brand - flavor
        item, gpt_brand, gpt_flavor = brand.split(" - ")
        clean_brand = match_gpt_output_to_list(gpt_brand, gpt_flavor, cfg.standard_drinks)
        clean_brands.append(clean_brand)
        
    timings["brand_cleaning"] = time.time() - t0

    # 7. Match brands to bottles and compute final counts
    t0 = time.time()

    bottle_brand_mapping = match_brands_to_bottles(front_bottles, clean_brands)

    front_cap_to_bottle = match_front_caps_to_bottles(front_bottles, cap_data)
    cap_counts = count_caps_per_column(image, corrected_lines, cap_data)

    brand_totals, lane_totals = compute_brand_counts(
        bottle_brand_mapping, front_cap_to_bottle, cap_counts, cfg.standard_drinks
    )
    
    timings["final_aggregation"] = time.time() - t0

    # 8. Print timing summary
    
    log.info("⏱ Timing summary (seconds): %s", timings)
    print("\n=== Timing Summary ===")
    for step, t in timings.items():
        print(f"{step:<25} {t:.3f}s")
    
    # total time completition
    processing_time = sum(timings.values())
    print(f"\nTotal time: {processing_time:.3f}s")

    return brand_totals, annotated_image, cap_data, front_bottles, bottle_brand_mapping, lane_totals, processing_time