import cv2
import base64
import numpy as np
from tqdm import tqdm
from rapidfuzz import process

def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.3):
    boxes = np.array(boxes)
    scores = np.array(scores)
    N = boxes.shape[0]
    indices = np.arange(N)
    keep = []
    visited = np.zeros(N, dtype=bool)
    pbar = tqdm(total=N, desc="Applying Soft-NMS", unit="box", leave=False)

    while len(indices) > 0:
        max_score_idx = np.argmax(scores[indices])
        current_idx = indices[max_score_idx]

        if visited[current_idx]:
            indices = np.delete(indices, max_score_idx)
            continue

        keep.append(current_idx)
        visited[current_idx] = True

        x1, y1, x2, y2 = boxes[current_idx]
        current_area = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = []
        for idx in indices:
            if visited[idx]:
                overlaps.append(0)
                continue

            xx1 = max(x1, boxes[idx][0])
            yy1 = max(y1, boxes[idx][1])
            xx2 = min(x2, boxes[idx][2])
            yy2 = min(y2, boxes[idx][3])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            intersection = w * h
            area = (boxes[idx][2] - boxes[idx][0] + 1) * (boxes[idx][3] - boxes[idx][1] + 1)
            iou = intersection / (current_area + area - intersection)
            overlaps.append(iou)

        overlaps = np.array(overlaps)
        decay = np.exp(-(overlaps ** 2) / sigma)
        scores[indices] = scores[indices] * decay

        indices = indices[scores[indices] > score_threshold]
        pbar.update(1)

    pbar.close()
    return keep


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def annotate_image(results):
    for result in results:
        image = result.orig_img.copy()
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return image


def normalize_brands(brand_list, similarity_threshold=95):
    normalized_mapping = {}
    unique_brands = []

    for brand in brand_list:
        best_match, score, _ = process.extractOne(brand, unique_brands) if unique_brands else (None, 0, None)
        if best_match and score >= similarity_threshold:
            normalized_mapping[brand] = best_match
        else:
            unique_brands.append(brand)
            normalized_mapping[brand] = brand

    return normalized_mapping


def count_caps_per_column(image, refined_column_lines, cap_data):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cap_counts = {}

    for i, (slope, intercept, f_cx) in enumerate(refined_column_lines):
        front_cap_box = next(
            (cap for cap in cap_data if (cap["x1"] + cap["x2"]) / 2 == f_cx),
            None
        )

        if front_cap_box is None:
            continue

        box_key = (front_cap_box["x1"], front_cap_box["y1"], front_cap_box["x2"], front_cap_box["y2"])
        cap_counts[box_key] = {'bottle': 0, 'can': 0}

        for cap in cap_data:
            x1, y1, x2, y2 = cap["x1"], cap["y1"], cap["x2"], cap["y2"]
            crossings = 0

            if slope is not None:
                y_at_x1 = slope * x1 + intercept
                y_at_x2 = slope * x2 + intercept
                x_at_y1 = (y1 - intercept) / slope if slope != 0 else None
                x_at_y2 = (y2 - intercept) / slope if slope != 0 else None

                if y1 <= y_at_x1 <= y2:
                    crossings += 1
                if y1 <= y_at_x2 <= y2:
                    crossings += 1
                if x_at_y1 is not None and x1 <= x_at_y1 <= x2:
                    crossings += 1
                if x_at_y2 is not None and x1 <= x_at_y2 <= x2:
                    crossings += 1
            else:
                if x1 <= f_cx <= x2:
                    crossings += 2

            if crossings >= 2:
                class_id = cap["class"]
                if class_id == 0:
                    cap_counts[box_key]['bottle'] += 1
                else:
                    cap_counts[box_key]['can'] += 1

    return cap_counts