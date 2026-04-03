"""
Color-based segmentation using HSV color space.

Covers: BGR to HSV conversion, color range masking,
multi-color segmentation, dominant color extraction.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils import save_image, save_plot, create_comparison_figure


# Predefined color ranges in HSV
COLOR_RANGES = {
    'red_lower':  {'lower': np.array([0, 100, 100]),   'upper': np.array([10, 255, 255])},
    'red_upper':  {'lower': np.array([160, 100, 100]), 'upper': np.array([180, 255, 255])},
    'orange':     {'lower': np.array([10, 100, 100]),  'upper': np.array([25, 255, 255])},
    'yellow':     {'lower': np.array([25, 100, 100]),  'upper': np.array([35, 255, 255])},
    'green':      {'lower': np.array([35, 50, 50]),    'upper': np.array([85, 255, 255])},
    'blue':       {'lower': np.array([85, 50, 50]),    'upper': np.array([130, 255, 255])},
    'purple':     {'lower': np.array([130, 50, 50]),   'upper': np.array([160, 255, 255])},
}


def segment_by_color(image, lower_hsv, upper_hsv):
    """Segment image by HSV color range."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return mask, segmented


def segment_all_colors(image):
    """Segment all predefined colors from the image."""
    results = {}
    for color_name, range_dict in COLOR_RANGES.items():
        mask, segmented = segment_by_color(image, range_dict['lower'], range_dict['upper'])
        pixel_count = cv2.countNonZero(mask)
        if pixel_count > 0:
            results[color_name] = {
                'mask': mask, 'segmented': segmented,
                'pixel_count': pixel_count,
                'percentage': (pixel_count / mask.size) * 100
            }
    return results


def extract_dominant_colors(image, k=5):
    """Extract dominant colors using K-Means clustering."""
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    percentages = (counts / len(labels)) * 100
    sorted_idx = np.argsort(-percentages)
    colors = centers[sorted_idx].astype(np.uint8)
    percentages = percentages[sorted_idx]

    bar = np.zeros((80, 500, 3), dtype=np.uint8)
    x_start = 0
    for color, pct in zip(colors, percentages):
        x_end = x_start + int(500 * pct / 100)
        cv2.rectangle(bar, (x_start, 0), (x_end, 80), color.tolist(), -1)
        x_start = x_end

    print(f"  Dominant colors (BGR):")
    for i, (c, p) in enumerate(zip(colors, percentages)):
        print(f"    #{i+1}: BGR({c[0]},{c[1]},{c[2]}) — {p:.1f}%")
    return colors, percentages, bar


def visualize_color_spaces(image):
    """Visualize image in HSV, LAB, grayscale, and BGR channels."""
    results = {}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    results['hsv'] = {'channels': [hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]],
                      'names': ['Hue', 'Saturation', 'Value']}
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    results['lab'] = {'channels': [lab[:,:,0], lab[:,:,1], lab[:,:,2]],
                      'names': ['L', 'A', 'B']}
    b, g, r = cv2.split(image)
    results['bgr'] = {'channels': [b, g, r], 'names': ['Blue', 'Green', 'Red']}
    return results


def run_segmentation_pipeline(image, output_prefix="segment"):
    """Run color segmentation pipeline."""
    print("\n=== Color Segmentation ===")

    # Color spaces
    cs = visualize_color_spaces(image)
    imgs, titles = [], []
    for name, data in cs.items():
        for ch, ch_name in zip(data['channels'], data['names']):
            imgs.append(ch)
            titles.append(f"{name.upper()}: {ch_name}")
    fig = create_comparison_figure(imgs, titles, cols=3)
    save_plot(fig, f"{output_prefix}_color_spaces.png", subdir="segmentation")

    # Multi-color segmentation
    color_results = segment_all_colors(image)
    if color_results:
        seg_imgs, seg_titles = [image], ['Original']
        for cn, d in color_results.items():
            print(f"    {cn}: {d['percentage']:.1f}%")
            seg_imgs.append(d['segmented'])
            seg_titles.append(cn.replace('_', ' ').title())
        if len(seg_imgs) > 1:
            fig = create_comparison_figure(seg_imgs, seg_titles, cols=3)
            save_plot(fig, f"{output_prefix}_colors.png", subdir="segmentation")

    # Dominant colors
    _, _, bar = extract_dominant_colors(image)
    save_image(bar, f"{output_prefix}_dominant_colors.jpg", subdir="segmentation")
    print(f"  Color segmentation complete.")
    return color_results
