import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from itertools import combinations
from inference.config import (
    DISTANCE_THRESHOLD_TO_FRONT_CAP,
    VANISHING_BOX_SIZE
)

def annotate_column_counts(image, lane_totals):
    annotated = image.copy()
    h, w = annotated.shape[:2]

    for bbox, count in lane_totals.items():
        x1, y1, x2, y2 = map(int, bbox)

        # Clamp to image bounds
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        # Column center
        x_center = (x1 + x2) // 2
        y_top = max(40, y1 - 10)

        text = str(count)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(1.2, w / 1200)
        thickness = 3

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        cv2.rectangle(
            annotated,
            (x_center - tw // 2 - 8, y_top - th - 12),
            (x_center + tw // 2 + 8, y_top + 8),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            annotated,
            text,
            (x_center - tw // 2, y_top),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA
        )

    return annotated

def match_and_extend_columns(image, front_caps, all_caps):
    """
    Matches each front cap to the closest non-front cap using vanishing point constraints,
    extends a straight line through both points to the top boundary of the image,
    and ensures accurate column assignments with additional distance constraints.

    Args:
        image (numpy.ndarray): The input image in BGR format.
        front_caps (list): List of (cx, cy) for front bottle caps.
        all_caps (list): List of (cx, cy) for all detected caps.

    Returns:
        column_lines (list): List of (slope, intercept, front_cap_x) for column line equations.
    """
    img_height, img_width = image.shape[:2]
    if not front_caps:
        print("[WARNING] No front caps detected — cannot compute columns.")
        return []
    x_center = img_width // 2
    middle_front_cap = min(front_caps, key=lambda cap: abs(cap[0] - x_center))

    column_pairs = []
    column_lines = []

    for f_cx, f_cy in front_caps:
        closest_cap = None
        min_distance = float("inf")

        for a_cx, a_cy in all_caps:
            if (a_cx, a_cy) == (f_cx, f_cy):
                continue

            distance = np.hypot(a_cx - f_cx, a_cy - f_cy)
            if distance > DISTANCE_THRESHOLD_TO_FRONT_CAP:
                continue

            if f_cx == middle_front_cap[0] or \
               (f_cx < middle_front_cap[0] and a_cx > f_cx) or \
               (f_cx > middle_front_cap[0] and a_cx < f_cx):
                if distance < min_distance:
                    min_distance = distance
                    closest_cap = (a_cx, a_cy)

        if closest_cap:
            column_pairs.append(((f_cx, f_cy), closest_cap))

    for (f_cx, f_cy), (c_cx, c_cy) in column_pairs:
        if c_cx != f_cx:
            slope = (c_cy - f_cy) / (c_cx - f_cx)
            intercept = f_cy - slope * f_cx
            column_lines.append((slope, intercept, f_cx))
        else:
            column_lines.append((None, None, f_cx))

    return column_lines

def draw_caps_and_column_lines(image, cap_data, column_lines, line_weight=3, distance_tolerance=20):
    output_image = image.copy()
    img_height, img_width = output_image.shape[:2]
    
    # 1. Dibujar las líneas de columna calculadas
    LINE_COLOR = (255, 0, 0)
    
    for slope, intercept, f_cx in column_lines:
        #print(f"[INFO] Drawing column line: slope={slope}, intercept={intercept}, front_cap_x={f_cx}")
        if slope is None:
            cv2.line(output_image, (int(f_cx), 0), (int(f_cx), img_height), LINE_COLOR, 2)
        else:
            x1_line = 0
            y1_line = int(slope * x1_line + intercept)
            x2_line = img_width
            y2_line = int(slope * x2_line + intercept)
            
            cv2.line(output_image, (x1_line, y1_line), (x2_line, y2_line), LINE_COLOR, 2)

    # 2. Evaluar y pintar las cajas de las tapas según su pertenencia a la columna
    COLOR_IN_COLUMN = (0, 255, 0)
    COLOR_OUTSIDE = (0, 0, 255)

    for cap in cap_data:
        x1, y1, x2, y2 = int(cap["x1"]), int(cap["y1"]), int(cap["x2"]), int(cap["y2"])
        cap_cx, cap_cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        is_in_column = False

        for slope, intercept, f_cx in column_lines:
            if slope is None:
                # Línea vertical: distancia pura en X
                if (x1 - distance_tolerance) <= f_cx <= (x2 + distance_tolerance) or abs(cap_cx - f_cx) <= 25:
                    is_in_column = True
                    break
            else:
                A = slope
                B = -1
                C = intercept
                distance = abs(A * cap_cx + B * cap_cy + C) / np.sqrt(A**2 + B**2)
                
                y_at_x1 = slope * x1 + intercept
                y_at_x2 = slope * x2 + intercept
                
                line_crosses_box = (y1 <= y_at_x1 <= y2) or (y1 <= y_at_x2 <= y2) or (min(y_at_x1, y_at_x2) <= y2 and max(y_at_x1, y_at_x2) >= y1)

                if distance <= distance_tolerance or line_crosses_box:
                    is_in_column = True
                    break

        color = COLOR_IN_COLUMN if is_in_column else COLOR_OUTSIDE
        #print(f"[INFO] Drawing cap box at ({x1}, {y1}, {x2}, {y2}) - {'IN COLUMN' if is_in_column else 'OUTSIDE COLUMN'}")
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, line_weight)

    return output_image


def compute_intersections(column_lines):
    """
    Finds intersection points of column lines.

    Args:
        column_lines (list): List of (slope, intercept, front_cap_x) tuples.

    Returns:
        np.ndarray: Array of (x, y) intersection points.
    """
    intersections = []
    for (m1, b1, _), (m2, b2, _) in combinations(column_lines, 2):
        if m1 is None or m2 is None or abs(m1 - m2) < 1e-6:
            continue
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        intersections.append((x, y))
    return np.array(intersections)


def cluster_vanishing_points(intersections):
    """
    Clusters intersection points to identify the vanishing region and visualizes the clustering process.

    Args:
        intersections (np.ndarray): Array of (x, y) intersections.

    Returns:
        tuple: (vanishing_x, vanishing_y, labels, clustering object)
    """
    if len(intersections) == 0:
        return None, None, None, None

    clustering = DBSCAN(eps=40, min_samples=3).fit(intersections)
    labels = clustering.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    if -1 in unique_labels:
        mask = unique_labels != -1
        unique_labels = unique_labels[mask]
        counts = counts[mask]
    if len(unique_labels) == 0:
        return None, None, None, None

    densest_label = unique_labels[np.argmax(counts)]
    cluster_points = intersections[labels == densest_label]

    vanishing_x = np.mean(cluster_points[:, 0])
    vanishing_y = np.mean(cluster_points[:, 1])

    return vanishing_x, vanishing_y, labels, clustering


def check_misaligned_columns(column_lines, vanishing_x, vanishing_y):
    if not column_lines:
        print("[WARNING] No column lines for misalignment check.")
        return {}

    misaligned = {}

    # CASO 1: Si tenemos punto de fuga, usamos la validación original de convergencia
    if vanishing_x is not None and vanishing_y is not None:
        x_min = vanishing_x - VANISHING_BOX_SIZE
        x_max = vanishing_x + VANISHING_BOX_SIZE
        y_min = vanishing_y - VANISHING_BOX_SIZE
        y_max = vanishing_y + VANISHING_BOX_SIZE

        for slope, intercept, f_cx in column_lines:
            if slope is None:
                continue

            y_at_x_min = slope * x_min + intercept
            y_at_x_max = slope * x_max + intercept
            x_at_y_min = (y_min - intercept) / slope if slope != 0 else f_cx
            x_at_y_max = (y_max - intercept) / slope if slope != 0 else f_cx

            crossings = sum([
                y_min <= y_at_x_min <= y_max,
                y_min <= y_at_x_max <= y_max,
                x_min <= x_at_y_min <= x_max,
                x_min <= x_at_y_max <= x_max
            ])

            if crossings < 2:
                misaligned[f_cx] = (slope, intercept)
    
    # CASO 2: Si NO hay punto de fuga, aplicamos inspección geométrica local de pendientes y cruces
    else:
        print("[INFO] Vanishing point not found. Running local geometric anomaly detection.")
        
        valid_slopes = [slope for slope, _, _ in column_lines if slope is not None]
        if not valid_slopes:
            return {}

        median_slope = np.median(valid_slopes)
        reference_sign = np.sign(median_slope) if median_slope != 0 else 1

        for slope, intercept, f_cx in column_lines:
            if slope is None:
                continue
            
            # Criterio 1: Inconsistencia de signo (si apunta al lado opuesto del resto de columnas)
            if reference_sign != 0 and np.sign(slope) != reference_sign:
                print(f"[DETECTED] Anomalía de signo en columna x={f_cx}: pendiente={slope} (esperado signo {reference_sign})")
                misaligned[f_cx] = (slope, intercept)
                continue

            # Criterio 2: Desviación excesiva respecto a la mediana de las pendientes
            if abs(slope - median_slope) > 0.8:  # Umbral más estricto
                print(f"[DETECTED] Desviación excesiva en columna x={f_cx}: pendiente={slope} vs mediana={median_slope}")
                misaligned[f_cx] = (slope, intercept)

    return misaligned


def correct_misaligned_columns(column_lines, misaligned_columns, vanishing_x, vanishing_y, front_caps):
    if not column_lines:
        print("[WARNING] No column lines to correct.")
        return []

    if not misaligned_columns:
        #print("[INFO] No misaligned columns detected — no corrections needed.")
        return column_lines

    print("[INFO] Correcting misaligned/overlapping columns to vertical lines.")
    corrected = []
    
    for slope, intercept, f_cx in column_lines:
        if f_cx not in misaligned_columns:
            corrected.append((slope, intercept, f_cx))
        else:
            # En lugar de recalcular un punto de fuga fallido, la enderezamos a vertical pura
            print(f"[INFO] Forzando columna en x={f_cx} a vertical para evitar sobreposiciones.")
            corrected.append((None, None, f_cx))

    return corrected