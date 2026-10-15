import time
import logging
#logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from inference.scoring_client import ScoringClient

from inference.detection.front_detection import (
    detect_and_match_fronts
)
from inference.detection.column_detection import (
    draw_caps_and_column_lines,
    match_and_extend_columns,
    compute_intersections,
    cluster_vanishing_points,
    check_misaligned_columns,
    correct_misaligned_columns,
)
from inference.detection.local_brand_detection import (
    get_brands_from_image,
    match_brands_to_bottles
)
from inference.image_utils import (
    count_caps_per_column
)

log = logging.getLogger("cuantotengo")

class InferencePipeline:
    def __init__(self):
        self.scoring = ScoringClient()
        self.last_timings = {}
        self.last_scoring_meta = {}
        
    def run(self, image, request_id=None):
        """Runs the full product detection pipeline and returns brand counts and annotated image."""
        timings = {}
        t_start = time.perf_counter()
        
        cap_data, front_bottles = self._detect_caps_and_fronts(
            image, timings, request_id
        )
        annotated_image, corrected_lines = self._detect_and_correct_columns(
            image, cap_data, front_bottles, timings
        )
        brand_totals, lane_totals, bottle_brand_mapping = self._brand_aggregation(
            image, cap_data, front_bottles, corrected_lines, timings
        )
        
        t0 = time.perf_counter()        
        annotated_image = self._additional_annotations(
            annotated_image, lane_totals, cap_data, corrected_lines
        )
        timings["annotations"] = time.perf_counter() - t0
        
        processing_total = time.perf_counter() - t_start
        timings["total_wall"] = processing_total

        self.last_timings = dict(timings)
        self._log_timings(timings, processing_total)

        
        return brand_totals, annotated_image, cap_data, front_bottles, bottle_brand_mapping, lane_totals, processing_total
        
    def _detect_caps_and_fronts(self, image, timings, request_id=None):
        t0 = time.perf_counter()
        
        cap_data, front_bottles, meta = self.scoring.detect(image, request_id=request_id)
        self.last_scoring_meta = meta

        timings["scoring_endpoint"] = time.perf_counter() - t0
        return cap_data, front_bottles
    
    def _detect_and_correct_columns(self, image, cap_data, front_bottles, timings):
        t0 = time.perf_counter()
        img_height, img_width = image.shape[:2]
        
        _, _, front_caps, all_caps, annotated_image = detect_and_match_fronts(
            image, cap_data, front_bottles
        )
        t1 = time.perf_counter()
        timings["match_fronts"] = t1 - t0
        
        # 1. Obtain column lines from front caps and all caps
        column_lines = match_and_extend_columns(image, front_caps, all_caps)
        if column_lines:
            x = int(round(float(column_lines[0][2])))
            x = max(0, min(x, img_width - 1))
        
        # 2. Calculate vanishing point from column lines to correct perspective
        intersections = compute_intersections(column_lines)
        vx, vy, _, _ = cluster_vanishing_points(intersections)
        
        # Validate vanishing point to avoid extreme perspective corrections
        if vx is not None and vy is not None:
            if vy > img_height * 0.6 or vx < -img_width or vx > img_width * 2:
                log.warning("Punto de fuga fuera de rango visual lógico. Descartando para evitar deformaciones.")
                vx, vy = None, None                
        
        # 3. Detect misaligned columns and correct them based on vanishing point
        misaligned_columns = check_misaligned_columns(column_lines, vx, vy)
        corrected_lines = correct_misaligned_columns(
            column_lines, {}, vx, vy, front_caps
        )
        
        timings["column_geometry"] = time.perf_counter() - t1
        return annotated_image, corrected_lines
    
    def _match_front_caps_to_bottles(self, front_bottles, cap_data):
        front_cap_to_bottle = {}

        for cap in cap_data:
            cx1, cy1, cx2, cy2 = cap["x1"], cap["y1"], cap["x2"], cap["y2"]

            for front in front_bottles:
                if front["x1"] <= (cx1 + cx2) / 2 <= front["x2"] and front["y1"] <= (cy1 + cy2) / 2 <= front["y2"]:
                    front_cap_to_bottle[(cx1, cy1, cx2, cy2)] = (front["x1"], front["y1"], front["x2"], front["y2"])
                    break

        return front_cap_to_bottle
    
    def _compute_brand_counts(self, bottle_brand_mapping, front_cap_to_bottle, cap_counts):
        brand_totals = {}
        lane_totals = {}

        for cap_bbox, bottle_bbox in front_cap_to_bottle.items():

            new_bottle_brand_mapping = {
                tuple(list(k)[:-1]): v for k, v in bottle_brand_mapping.items()
            }

            if bottle_bbox in new_bottle_brand_mapping:
                matched_product = new_bottle_brand_mapping[bottle_bbox]

                # Totals count
                counts = cap_counts.get(cap_bbox, {'bottle': 0, 'can': 0})
                if counts['bottle'] > 0:
                    key = f"{matched_product.replace(' ', '_')}"
                    brand_totals[key] = brand_totals.get(key, 0) + counts['bottle']
                if counts['can'] > 0:
                    key = f"{matched_product.replace(' ', '_')}"
                    brand_totals[key] = brand_totals.get(key, 0) + counts['can']

                # Lanes count
                lane_totals[bottle_bbox] = counts["bottle"] + counts['can']

        return brand_totals, lane_totals

    def _brand_aggregation(self, image, cap_data, front_bottles, corrected_lines, timings):
        t0 = time.perf_counter()

        brands_list = get_brands_from_image(front_bottles, image)
        t1 = time.perf_counter()
        timings["brand_model"] = t1 - t0

        bottle_brand_mapping = match_brands_to_bottles(front_bottles, brands_list)
        front_cap_to_bottle = self._match_front_caps_to_bottles(front_bottles, cap_data)
        
        cap_counts = count_caps_per_column(image, corrected_lines, cap_data)

        brand_totals, lane_totals = self._compute_brand_counts(
            bottle_brand_mapping,
            front_cap_to_bottle,
            cap_counts
        )    

        timings["counting"] = time.perf_counter() - t1
        return brand_totals, lane_totals, bottle_brand_mapping

    def _additional_annotations(self, image, lane_totals, cap_data, corrected_lines):
        # annotated_image = annotate_column_counts(
        #     annotated_image,
        #     lane_totals
        # )
        annotated_image = draw_caps_and_column_lines(image, cap_data, corrected_lines)
        return annotated_image
    
    def _log_timings(self, timings, total):
        log.info("Timing summary (seconds): %s", timings)
        # for k, v in timings.items():
        #     log.info("%s: %.3fs", k, v)
        log.info("Total processing time: %.3fs", total)
