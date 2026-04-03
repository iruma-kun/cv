"""
Contour detection and shape analysis.

Covers: Contour finding, bounding boxes, convex hulls,
shape approximation, area/perimeter computation.
"""

import cv2
import numpy as np
from src.utils import save_image, save_plot, create_comparison_figure


def find_contours(image, method='canny'):
    """
    Find contours in an image.

    Args:
        image: Input BGR or grayscale image.
        method: 'canny' for edge-based, 'threshold' for threshold-based.

    Returns:
        tuple: (contours, hierarchy)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if method == 'canny':
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    else:
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

    return contours, hierarchy


def draw_contours(image, contours, thickness=2):
    """Draw all contours on a copy of the image with random colors."""
    result = image.copy()
    for i, contour in enumerate(contours):
        color = (
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255))
        )
        cv2.drawContours(result, [contour], -1, color, thickness)
    return result


def draw_bounding_boxes(image, contours, min_area=500):
    """
    Draw bounding rectangles around contours.

    Args:
        image: Input image.
        contours: List of contours.
        min_area: Minimum contour area to include.

    Returns:
        numpy.ndarray: Image with bounding boxes drawn.
    """
    result = image.copy()
    count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, f"A:{int(area)}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        count += 1
    print(f"  Found {count} objects with area > {min_area}")
    return result


def draw_convex_hulls(image, contours, min_area=500):
    """Draw convex hulls around significant contours."""
    result = image.copy()
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        hull = cv2.convexHull(contour)
        cv2.drawContours(result, [hull], -1, (0, 0, 255), 2)
    return result


def approximate_shapes(image, contours, min_area=500):
    """
    Approximate contours to polygonal shapes and classify them.

    Classification based on number of vertices:
    3=Triangle, 4=Rectangle/Square, 5=Pentagon, >5=Circle-like
    """
    result = image.copy()
    shape_counts = {}

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        # Approximate the contour shape
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # Classify shape
        vertices = len(approx)
        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            # Check aspect ratio to distinguish rectangle vs square
            x, y, w, h = cv2.boundingRect(approx)
            ratio = float(w) / h
            shape = "Square" if 0.85 <= ratio <= 1.15 else "Rectangle"
        elif vertices == 5:
            shape = "Pentagon"
        elif vertices > 5:
            shape = "Circle"
        else:
            shape = "Unknown"

        shape_counts[shape] = shape_counts.get(shape, 0) + 1

        # Draw the approximated shape and label
        cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(result, shape, (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    print(f"  Shape detection: {shape_counts}")
    return result, shape_counts


def analyze_contours(contours, min_area=100):
    """
    Compute properties for each significant contour.

    Returns:
        list[dict]: Properties for each contour (area, perimeter, circularity, etc.)
    """
    properties = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        extent = area / (w * h) if (w * h) > 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        properties.append({
            'index': i,
            'area': area,
            'perimeter': perimeter,
            'circularity': round(circularity, 3),
            'aspect_ratio': round(aspect_ratio, 3),
            'extent': round(extent, 3),
            'solidity': round(solidity, 3),
            'bounding_box': (x, y, w, h),
        })

    return properties


def run_contour_pipeline(image, output_prefix="contours"):
    """
    Run contour detection and analysis pipeline.

    Args:
        image: Input BGR image.
        output_prefix: Prefix for output filenames.

    Returns:
        dict: Dictionary of result name -> result image or data.
    """
    print("\n=== Contour Detection & Analysis ===")

    results = {}

    # Find contours using both methods
    contours_canny, _ = find_contours(image, 'canny')
    contours_thresh, _ = find_contours(image, 'threshold')
    print(f"  Canny method: {len(contours_canny)} contours found")
    print(f"  Threshold method: {len(contours_thresh)} contours found")

    # Use canny contours for analysis
    contours = contours_canny

    # Draw contours
    results['all_contours'] = draw_contours(image, contours)

    # Bounding boxes
    results['bounding_boxes'] = draw_bounding_boxes(image, contours)

    # Convex hulls
    results['convex_hulls'] = draw_convex_hulls(image, contours)

    # Shape approximation
    shapes_img, shape_counts = approximate_shapes(image, contours)
    results['shapes'] = shapes_img

    # Analyze contour properties
    props = analyze_contours(contours)
    if props:
        print(f"  Top contour properties:")
        for p in sorted(props, key=lambda x: x['area'], reverse=True)[:5]:
            print(f"    #{p['index']}: area={p['area']:.0f}, "
                  f"circularity={p['circularity']}, "
                  f"solidity={p['solidity']}")

    # Save results
    for name, img in results.items():
        save_image(img, f"{output_prefix}_{name}.jpg", subdir="contours")

    fig = create_comparison_figure(
        [image, results['all_contours'], results['bounding_boxes'],
         results['convex_hulls'], results['shapes']],
        ['Original', 'All Contours', 'Bounding Boxes', 'Convex Hulls', 'Shapes'],
        cols=3
    )
    save_plot(fig, f"{output_prefix}_comparison.png", subdir="contours")

    print(f"  Contour analysis complete.")
    return results
