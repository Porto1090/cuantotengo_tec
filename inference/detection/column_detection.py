import numpy as np
from sklearn.cluster import DBSCAN
from itertools import combinations
from inference.config import (
    DISTANCE_THRESHOLD_TO_FRONT_CAP,
    VANISHING_BOX_SIZE
)

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
    """
    Identifies misaligned column lines that do not pass through the vanishing region.

    Args:
        column_lines (list): List of (slope, intercept, front_cap_x).
        vanishing_x (float): x-coordinate of vanishing point.
        vanishing_y (float): y-coordinate of vanishing point.

    Returns:
        dict: Mapping of misaligned front_cap_x to their (slope, intercept).
    """
    
    # SAFETY CHECK: no column lines, 
    if not column_lines:
        print("[WARNING] No column lines for misalignment check.")
        return {}

    # SAFETY CHECK: no vanishing point
    if vanishing_x is None or vanishing_y is None:
        print("[WARNING] Vanishing point not found — skipping misalignment detection.")
        return {}
    
    misaligned = {}

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

    return misaligned


def correct_misaligned_columns(column_lines, misaligned_columns, vanishing_x, vanishing_y, front_caps):
    """
    Corrects misaligned columns by re-regressing from vanishing region to front cap.

    Args:
        column_lines (list): Original column line tuples.
        misaligned_columns (dict): Misaligned columns {f_cx: (slope, intercept)}.
        vanishing_x (float): x-coordinate of vanishing point.
        vanishing_y (float): y-coordinate of vanishing point.
        front_caps (list): List of (cx, cy) for front bottle caps.

    Returns:
        list: Corrected list of (slope, intercept, front_cap_x) lines.
    """
    
    # SAFETY:
    if not column_lines:
        print("[WARNING] No column lines to correct.")
        return []

    # SAFETY:
    if not misaligned_columns:
        return column_lines

    # SAFETY:
    if vanishing_x is None or vanishing_y is None:
        print("[WARNING] No vanishing point — skipping misalignment corrections.")
        return column_lines
    
    corrected = []

    for slope, intercept, f_cx in column_lines:
        if f_cx not in misaligned_columns:
            corrected.append((slope, intercept, f_cx))
        else:
            f_cy = next((cy for cx, cy in front_caps if cx == f_cx), None)
            if f_cy is None:
                corrected.append((slope, intercept, f_cx))
                continue
            new_slope = (f_cy - vanishing_y) / (f_cx - vanishing_x)
            new_intercept = f_cy - new_slope * f_cx
            corrected.append((new_slope, new_intercept, f_cx))

    return corrected
