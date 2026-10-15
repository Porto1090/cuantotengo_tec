from inference.config import (
    CAP_DETECTION_CONFIDENCE,
    SOFT_NMS_IOU_THRESHOLD,
    SOFT_NMS_SIGMA,
    SOFT_NMS_SCORE_THRESHOLD,
)
import numpy as np
from tqdm import tqdm

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

def detect_caps(image, cap_model):
    results = cap_model.predict(image, conf=CAP_DETECTION_CONFIDENCE)
    # cap_boxes = []
    # scores = []
    # class_ids = []

    # for result in results:
    #     boxes = result.boxes.xyxy.cpu().numpy()
    #     confs = result.boxes.conf.cpu().numpy()
    #     cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    #     for i, (x1, y1, x2, y2) in enumerate(boxes):
    #         cap_boxes.append((x1, y1, x2, y2))
    #         scores.append(confs[i])
    #         class_ids.append(cls_ids[i])

    # keep_indices = soft_nms(
    #     cap_boxes,
    #     scores,
    #     iou_threshold=SOFT_NMS_IOU_THRESHOLD,
    #     sigma=SOFT_NMS_SIGMA,
    #     score_threshold=SOFT_NMS_SCORE_THRESHOLD,
    # )
    # refined_caps = [cap_boxes[idx] for idx in keep_indices]
    # refined_classes = [class_ids[idx] for idx in keep_indices]
    # refined_scores = [scores[idx] for idx in keep_indices]

    detected_caps = []
    # for i, (x1, y1, x2, y2) in enumerate(refined_caps):
    #     detected_caps.append({
    #         "x1": x1,
    #         "y1": y1,
    #         "x2": x2,
    #         "y2": y2,
    #         "class": refined_classes[i],
    #         "confidence": refined_scores[i]
    #     })

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            detected_caps.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class": cls_ids[i],
                "confidence": confs[i]
            })

    return detected_caps
