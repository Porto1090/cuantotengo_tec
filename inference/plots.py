import cv2
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
THIS MODULE IS NOT NECESSARILY BEING USED IN THE CURRENT PIPELINE.
"""

def plot_cap_centers_by_column(image, cap_columns, save_path="plots/cap_centers.png"):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.title("Cap Centers by Column")

    for column_idx, ((x1, y1, x2, y2), caps) in enumerate(cap_columns.items()):
        cap_centers = caps
        for (cx, cy) in cap_centers:
            plt.scatter(cx, cy, label=f"Column {column_idx + 1}", s=60)
        plt.scatter((x1 + x2) / 2, (y1 + y2) / 2, color="black", s=120, edgecolor="white")

    plt.axis("off")
    plt.savefig(save_path)
    plt.close()


def plot_corrected_columns(image_path, original_lines, corrected_lines, misaligned_columns, vanishing_x, vanishing_y, front_caps, save_path="plots/corrected_columns.png"):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.title("Corrected Column Lines")

    for slope, intercept, f_cx in original_lines:
        if slope is not None:
            x_vals = [0, image.shape[1]]
            y_vals = [slope * x + intercept for x in x_vals]
            plt.plot(x_vals, y_vals, linestyle='dashed', color='gray', alpha=0.5)

    for slope, intercept, f_cx in corrected_lines:
        if slope is not None:
            x_vals = [0, image.shape[1]]
            y_vals = [slope * x + intercept for x in x_vals]
            plt.plot(x_vals, y_vals, color='blue')

    plt.scatter(vanishing_x, vanishing_y, c='red', s=100, marker='x')

    for cx, cy in front_caps:
        plt.scatter(cx, cy, color="green", s=60)

    plt.axis("off")
    plt.savefig(save_path)
    plt.close()


def plot_vanishing_point(image_path, intersections, labels, vanishing_x, vanishing_y, save_path="plots/vanishing_point.png"):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.title("Vanishing Point and Intersections")

    for (x, y), label in zip(intersections, labels):
        color = 'red' if label == -1 else 'blue'
        plt.scatter(x, y, c=color, s=30)

    plt.scatter(vanishing_x, vanishing_y, c='yellow', s=100, marker='x')
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()


def plot_nearby_caps_with_labels(image, front_caps, all_caps, x_tolerance=20, save_path="plots/nearby_caps.png"):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.title("Nearby Caps with Labels")

    for i, (fx, fy) in enumerate(front_caps):
        plt.scatter(fx, fy, color='green', label=f"Front Cap {i+1}")
        for (cx, cy) in all_caps:
            if abs(cx - fx) <= x_tolerance:
                plt.scatter(cx, cy, color='orange')

    plt.axis("off")
    plt.savefig(save_path)
    plt.close()


def plot_cap_centers_by_column(image, cap_columns):
    """
    Plots the center of each cap's bounding box, using a different color for each identified column.

    Args:
        image (numpy.ndarray): The input image in BGR format (as loaded by OpenCV).
        cap_columns (dict): Dictionary mapping each front cap's bounding box (4-coordinates)
                            to a list of assigned cap centers (2-coordinates).
    """
    # Ensure the image is valid
    if image is None:
        raise ValueError("Error: The provided image array is None.")

    # Convert from BGR to RGB for plotting
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define a color palette for each column
    num_columns = len(cap_columns)
    color_palette = plt.colormaps["tab10"]  # Compatible with Matplotlib 3.7+

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.title("Cap Centers Colored by Column")

    for i, (front_cap_box, assigned_caps) in enumerate(cap_columns.items()):
        # Extract front cap's center from its bounding box
        f_cx = (front_cap_box[0] + front_cap_box[2]) / 2  # (x1 + x2) / 2
        f_cy = (front_cap_box[1] + front_cap_box[3]) / 2  # (y1 + y2) / 2

        # Choose color for this column (wrap with modulo if more than 10 columns)
        color = color_palette(i % 10)

        # Plot the front cap in black (with a white edge)
        plt.scatter(
            f_cx, f_cy,
            color="black",
            s=100,
            edgecolor="white",
            label="Front Cap" if i == 0 else ""  # Add label only once
        )

        # Plot the assigned (back) caps in the same color
        for cap_center in assigned_caps:
            if len(cap_center) == 2:
                c_cx, c_cy = cap_center
            else:
                raise ValueError(f"Unexpected cap format: {cap_center}")

            plt.scatter(
                c_cx, c_cy,
                color=color,
                s=70,
                label=f"Column {i+1}" if i == 0 else ""  # Add label only once
            )
            # Connect the front cap and this assigned cap with a dashed line
            plt.plot([f_cx, c_cx], [f_cy, c_cy], color=color, linestyle="dashed", linewidth=2)

    plt.legend()
    plt.axis("off")
    plt.savefig("my_plot.png")  # Saves the figure to a PNG file
    plt.close()  
    

def plot_corrected_columns(image_path, column_lines, corrected_lines, misaligned_columns, vanishing_x, vanishing_y, front_caps):
    """
    Overlays original and corrected columns with vanishing region, ensuring start at front caps.
    Keeps all lines within the image range and preserves the original image scale.
    
    - Blue (dotted): Original correct lines
    - Red (dashed): Misaligned lines
    - Green (solid): Corrected lines
    - Red box: Vanishing region
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not load the image from {image_path}")
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image.shape[:2]

    # plt.figure(figsize=(img_width / 100, img_height / 100))  # Preserve original scale
    # plt.imshow(image_rgb)
    # plt.title("Vanishing Region & Corrected Columns")

    # Keep lines within the image bounds
    def clip_x(x):
        return max(0, min(img_width - 1, x))

    def clip_y(y):
        return max(0, min(img_height - 1, y))

    # Plot original lines (Correct ones in BLUE, misaligned ones in RED)
    for (slope, intercept, f_cx), (f_cx, f_cy) in zip(column_lines, front_caps):
        y_end = 0  # Top of the image
        x_end = clip_x((y_end - intercept) / slope if slope != 0 else f_cx)
        
        # **If misaligned, plot in RED (dashed), else BLUE (dotted)**
        color = "red" if f_cx in misaligned_columns else "blue"
        linestyle = "dashed" if f_cx in misaligned_columns else "dotted"
        
        # plt.plot([clip_x(f_cx), x_end], [clip_y(f_cy), clip_y(y_end)], color=color, linestyle=linestyle, linewidth=2)

    # Plot corrected lines in GREEN (solid)
    for (slope, intercept, f_cx), (f_cx, f_cy) in zip(corrected_lines, front_caps):
        y_end = 0
        x_end = clip_x((y_end - intercept) / slope if slope != 0 else f_cx)
        # plt.plot([clip_x(f_cx), x_end], [clip_y(f_cy), clip_y(y_end)], color="green", linestyle="solid", linewidth=2)

    # Plot vanishing region as a RED bounding box
    box_size = 80  # Adjustable vanishing region size
    # plt.plot(
    #     [clip_x(vanishing_x - box_size), clip_x(vanishing_x + box_size), 
    #      clip_x(vanishing_x + box_size), clip_x(vanishing_x - box_size), clip_x(vanishing_x - box_size)],
    #     [clip_y(vanishing_y - box_size), clip_y(vanishing_y - box_size), 
    #      clip_y(vanishing_y + box_size), clip_y(vanishing_y + box_size), clip_y(vanishing_y - box_size)],
    #     color="red", linestyle="dotted", linewidth=2, label="Vanishing Region"
    # )

    # plt.legend(["Original Correct", "Misaligned", "Corrected", "Vanishing Region"])
    # plt.axis("off")
    # plt.show()


def plot_vanishing_point(image_path, intersections, labels, vanishing_x, vanishing_y):
    """Plots detected intersections, clustered vanishing points, and final vanishing region."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # plt.figure(figsize=(10, 6))
    # plt.imshow(image_rgb)
    # plt.title("Vanishing Point Identification")

    # Color map for clusters
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    for i, (x, y) in enumerate(intersections):
        color = colors(labels[i] % 10) if labels[i] != -1 else "gray"
        # plt.scatter(x, y, color=color, s=50, label=f"Cluster {labels[i]}" if labels[i] != -1 else "Noise")

    # Highlight vanishing region
    # plt.scatter(vanishing_x, vanishing_y, color="red", s=200, marker="x", label="Vanishing Point")

    # Draw vanishing region bounding box
    # plt.plot(
    #     [vanishing_x - VANISHING_BOX_SIZE, vanishing_x + VANISHING_BOX_SIZE,
    #      vanishing_x + VANISHING_BOX_SIZE, vanishing_x - VANISHING_BOX_SIZE, vanishing_x - VANISHING_BOX_SIZE],
    #     [vanishing_y - VANISHING_BOX_SIZE, vanishing_y - VANISHING_BOX_SIZE,
    #      vanishing_y + VANISHING_BOX_SIZE, vanishing_y + VANISHING_BOX_SIZE, vanishing_y - VANISHING_BOX_SIZE],
    #     color="red", linestyle="dashed", linewidth=2
    # )

    #plt.legend()
    # plt.axis("off")
    # plt.show()


def plot_nearby_caps_with_labels(image, front_caps, all_caps, x_tolerance=20):
    """
    Plots the front caps and highlights the nearby caps within the given x-tolerance.
    Displays the rounded coordinates next to the points on the plot.

    Args:
        image (numpy.ndarray): The image in BGR format.
        front_caps (list): List of (cx, cy) for front bottle caps.
        all_caps (list): List of (cx, cy) for all detected caps.
        x_tolerance (int): Allowed x-coordinate variation for identifying close matches.
    """
    # Validate the image
    if image is None:
        raise ValueError("Error: The provided image array is None.")

    # Convert BGR to RGB for plotting
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find nearby caps while avoiding self-matching
    nearby_caps_dict = {cap: [] for cap in front_caps}
    for f_cx, f_cy in front_caps:
        for a_cx, a_cy in all_caps:
            if (a_cx, a_cy) != (f_cx, f_cy) and abs(a_cx - f_cx) <= x_tolerance:
                nearby_caps_dict[(f_cx, f_cy)].append((a_cx, a_cy))

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.title(f"Nearby Caps within x-tolerance {x_tolerance}")

    # Draw all caps in green
    for a_cx, a_cy in all_caps:
        plt.scatter(a_cx, a_cy, color="green", s=50,
                    label="All Caps" if "All Caps" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(a_cx + 10, a_cy, f"({round(a_cx)}, {round(a_cy)})",
                 fontsize=8, color="green")

    # Draw front caps in orange
    for f_cx, f_cy in front_caps:
        plt.scatter(f_cx, f_cy, color="orange", s=100,
                    label="Front Cap" if "Front Cap" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(f_cx + 10, f_cy, f"({round(f_cx)}, {round(f_cy)})",
                 fontsize=8, color="orange")

    plt.legend()
    plt.axis("off")
    plt.savefig("my_plot.png")  # Saves the figure to a PNG file
    plt.close()  
    