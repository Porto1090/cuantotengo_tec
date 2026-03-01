from inference.image_utils import soft_nms
from inference.config import (
    CAP_DETECTION_CONFIDENCE,
    SOFT_NMS_IOU_THRESHOLD,
    SOFT_NMS_SIGMA,
    SOFT_NMS_SCORE_THRESHOLD,
)

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
