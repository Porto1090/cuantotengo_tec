import time
import logging
import concurrent.futures as conf

import inference.config as cfg
from inference.runtime.device import get_device
from inference.runtime.models import load_models

from inference.detection.cap_detection import (
    detect_caps
)
from inference.detection.front_detection import (
    detect_front_bottles,
    detect_and_match_fronts
)
from inference.detection.column_detection import (
    match_and_extend_columns,
    compute_intersections,
    cluster_vanishing_points,
    check_misaligned_columns,
    correct_misaligned_columns,
    annotate_column_counts
)
from inference.detection.local_brand_detection import (
    get_brands_from_image,
    match_brands_to_bottles
)
from inference.final_aggregation import (
    match_front_caps_to_bottles,
    compute_brand_counts,
)
from inference.image_utils import (
    count_caps_per_column
)

log = logging.getLogger("cuantotengo")

class InferencePipeline:
    def __init__(self):
        self.device = get_device()
        log.info("Using device: %s", self.device)
        self.models = load_models(
            self.device,
            cfg.CAP_MODEL_PATH,
            cfg.FRONT_BOTTLE_MODEL_PATH
        )
        log.info("YOLO models loaded.")

    def run(self, image):
        """Runs the full product detection pipeline and returns brand counts and annotated image."""
        timings = {}
        
        cap_data, front_bottles = self._detect_caps_and_fronts(
            image, timings
        )
        annotated_image, corrected_lines = self._detect_and_correct_columns(
            image, cap_data, timings
        )
        brand_totals, lane_totals, bottle_brand_mapping = self._brand_aggregation(
            image, cap_data, front_bottles, corrected_lines, timings
        )
        # annotated_image = self._additional_annotations(
        #     annotated_image, lane_totals
        # )
        
        processing_total = sum(timings.values())
        self._log_timings(timings, processing_total)
        
        return brand_totals, annotated_image, cap_data, front_bottles, bottle_brand_mapping, lane_totals, processing_total
        
    def _detect_caps_and_fronts(self, image, timings):
        t0 = time.time()
        
        if self.device != "cpu":
            with conf.ThreadPoolExecutor() as executor:
                caps_future = executor.submit(
                    detect_caps,
                    image,
                    self.models["cap"]
                )
                fronts_future = executor.submit(
                    detect_front_bottles,
                    image,
                    self.models["front"]
                )

                cap_data = caps_future.result()
                front_bottles = fronts_future.result()
        else:
            cap_data = detect_caps(
                image,
                self.models["cap"]
            )
            front_bottles = detect_front_bottles(
                image,
                self.models["front"]
            )

        timings["caps_and_front_detections"] = time.time() - t0
        return cap_data, front_bottles
    
    def _detect_and_correct_columns(self, image, cap_data, timings):
        t0 = time.time()
        
        _, _, front_caps, all_caps, annotated_image = detect_and_match_fronts(
            image, cap_data, self.models["front"]
        )
        
        column_lines = match_and_extend_columns(image, front_caps, all_caps)
        intersections = compute_intersections(column_lines)
        vx, vy, _, _ = cluster_vanishing_points(intersections)
        
        misaligned_columns = check_misaligned_columns(column_lines, vx, vy)
        corrected_lines = correct_misaligned_columns(
            column_lines, misaligned_columns, vx, vy, front_caps
        )
        
        timings["column_processing"] = time.time() - t0
        return annotated_image, corrected_lines

    def _brand_aggregation(self, image, cap_data, front_bottles, corrected_lines, timings):
        t0 = time.time()

        brands_list = get_brands_from_image(front_bottles, image)
        bottle_brand_mapping = match_brands_to_bottles(front_bottles, brands_list)
        front_cap_to_bottle = match_front_caps_to_bottles(front_bottles, cap_data)
        cap_counts = count_caps_per_column(image, corrected_lines, cap_data)

        brand_totals, lane_totals = compute_brand_counts(
            bottle_brand_mapping,
            front_cap_to_bottle,
            cap_counts
        )

        timings["brand_aggregation"] = time.time() - t0
        return brand_totals, lane_totals, bottle_brand_mapping

    def _additional_annotations(self, annotated_image, lane_totals):
        annotated_image = annotate_column_counts(
            annotated_image,
            lane_totals
        )
        return annotated_image
    
    def _log_timings(self, timings, total):
        log.info("Timing summary (seconds): %s", timings)
        for k, v in timings.items():
            log.info("%s: %.3fs", k, v)
        log.info("Total processing time: %.3fs", total)
