import cv2
import base64


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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