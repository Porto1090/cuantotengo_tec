import cv2
from inference.config import (
    FRONT_DETECTION_CONFIDENCE,
    FRONT_CAP_OVERLAP_THRESHOLD
)

def detect_front_bottles(image, model):
    """
    Detects front-facing bottles using YOLO and sorts them from left to right.
    This is the ONLY place where the front model runs.

    Returns:
        list[dict]: [{"x1", "y1", "x2", "y2", "confidence"}] sorted by x1.
    """
    results = model.predict(image, conf=FRONT_DETECTION_CONFIDENCE)
    front_bottles = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            front_bottles.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confs[i]
            })

    front_bottles.sort(key=lambda box: box["x1"])
    return front_bottles


def detect_and_match_fronts(image, cap_data, front_detections, line_weight=3, min_bottle_width=60):
    """
    Filters precomputed front detections (no model call) and matches caps to them.

    Args:
        image: BGR image.
        cap_data: output of detect_caps.
        front_detections: output of detect_front_bottles.

    Returns:
        front_boxes, all_bottle_boxes, front_caps, all_caps, output_image
    """
    output_image = image.copy()
    img_height, img_width = output_image.shape[:2]
    front_boxes = []
    all_bottle_boxes = []
    
    print(f"[INFO] Received {len(front_detections)} raw front-facing bottles with confidence >= {FRONT_DETECTION_CONFIDENCE}.")

    for det in front_detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        confidence = det["confidence"]
        box_width = x2 - x1
            
        # --- FILTRO DE TRUNCAMIENTO EN BORDES ---
        # Un objeto cortado que no sirve como pivote de columna.
        is_cut_at_edge = (x1 <= 0.5 or x2 >= img_width - 0.5)
        if is_cut_at_edge and box_width < min_bottle_width:
            print(f"[WARNING] Detección frontal no confiable (cortada en borde): [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] (ancho: {box_width:.1f}px). Buscando alternativa...")
            # En lugar de ignorarla por completo sin opción, podemos degradarla o buscar la tapa interior disponible más cercana.
            continue

        front_boxes.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": confidence
        })
        all_bottle_boxes.append((x1, y1, x2, y2))
            
    if not front_boxes and cap_data:
        print("[INFO] Aplicando fallback: Seleccionando la tapa más baja disponible como pivote alternativo.")
        lowest_cap = max(cap_data, key=lambda c: c["y2"])
        # Creamos una caja sintética basada en esta tapa para que sirva de ancla
        lc_x1 = max(0, lowest_cap["x1"] - 20)
        lc_y1 = max(0, lowest_cap["y1"] - 40)
        lc_x2 = min(img_width, lowest_cap["x2"] + 20)
        lc_y2 = min(img_height, lowest_cap["y2"] + 100)
        
        front_boxes.append({
            "x1": lc_x1, "y1": lc_y1, "x2": lc_x2, "y2": lc_y2, "confidence": 1.0
        })
        
    #print(f"[INFO] Valid front-facing bottles after edge filtering: {len(front_boxes)}")

    COLOR_MATCHED = (0, 255, 0)
    COLOR_UNMATCHED = (0, 165, 255)

    front_caps = []
    all_caps = []

    for cap in cap_data:
        x1, y1, x2, y2, cls = cap["x1"], cap["y1"], cap["x2"], cap["y2"], cap["class"]
        cap_cx, cap_cy = (x1 + x2) / 2, (y1 + y2) / 2
        all_caps.append((cap_cx, cap_cy))

        for box in front_boxes:
            fx1, fy1, fx2, fy2 = box["x1"], box["y1"], box["x2"], box["y2"]

            cap_area = (x2 - x1) * (y2 - y1)
            intersection_x1 = max(fx1, x1)
            intersection_y1 = max(fy1, y1)
            intersection_x2 = min(fx2, x2)
            intersection_y2 = min(fy2, y2)

            w = max(0, intersection_x2 - intersection_x1)
            h = max(0, intersection_y2 - intersection_y1)
            intersection_area = w * h

            if cap_area > 0 and (intersection_area / cap_area) >= FRONT_CAP_OVERLAP_THRESHOLD:
                front_caps.append((cap_cx, cap_cy))
                break

    for box in front_boxes:
        bx1, by1, bx2, by2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        cv2.rectangle(output_image, (bx1, by1), (bx2, by2), COLOR_MATCHED, line_weight)

    return front_boxes, all_bottle_boxes, front_caps, all_caps, output_image