import cv2
from inference.config import (
    FRONT_DETECTION_CONFIDENCE,
    FRONT_CAP_OVERLAP_THRESHOLD
)

def detect_front_bottles(image, model):
    """
    Detects front-facing bottles using YOLO and sorts them from left to right.

    Args:
        image (numpy.ndarray): The input image as a NumPy array (BGR format).
        model: YOLO model used to detect front-facing bottles.

    Returns:
        list: A list of bounding boxes [(x1, y1, x2, y2)] sorted from left to right.
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


def detect_and_match_fronts(image, cap_data, model, line_weight=3):
    output_image = image.copy()

    results = model.predict(output_image, conf=FRONT_DETECTION_CONFIDENCE)
    front_boxes = []
    all_bottle_boxes = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            confidence = confs[i]
            box_with_conf = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence
            }
            front_boxes.append(box_with_conf)
            all_bottle_boxes.append((x1, y1, x2, y2))

    class_colors = {
        0: (0, 165, 255),
        1: (0, 165, 255)
    }
    highlight_color = (0, 165, 255)

    front_caps = []
    all_caps = []

    for cap in cap_data:
        x1, y1, x2, y2, cls = cap["x1"], cap["y1"], cap["x2"], cap["y2"], cap["class"]
        cap_cx, cap_cy = (x1 + x2) / 2, (y1 + y2) / 2
        all_caps.append((cap_cx, cap_cy))

        matched = False
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

            if intersection_area / cap_area >= FRONT_CAP_OVERLAP_THRESHOLD:
                matched = True
                front_caps.append((cap_cx, cap_cy))
                break

        color = highlight_color if matched else class_colors.get(cls, (0, 255, 0))
        cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color, line_weight)

    for box in front_boxes:
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), line_weight)

    return front_boxes, all_bottle_boxes, front_caps, all_caps, output_image
