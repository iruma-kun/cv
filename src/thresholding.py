"""
Thresholding techniques.

Covers: Simple (Binary, Inv, Trunc, ToZero), Adaptive (Mean, Gaussian), Otsu's method.
"""

import cv2
import numpy as np
from src.utils import save_image, save_plot, create_comparison_figure


def apply_binary_threshold(image, thresh=127):
    """Apply simple binary threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return result


def apply_binary_inv_threshold(image, thresh=127):
    """Apply inverse binary threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    return result


def apply_truncate_threshold(image, thresh=127):
    """Apply truncate threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_TRUNC)
    return result


def apply_tozero_threshold(image, thresh=127):
    """Apply to-zero threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_TOZERO)
    return result


def apply_otsu_threshold(image):
    """
    Apply Otsu's automatic thresholding.
    Automatically determines optimal threshold value from the image histogram.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Apply Gaussian blur before Otsu for better results
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_val, result = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"  Otsu's threshold value: {thresh_val:.1f}")
    return result, thresh_val


def apply_adaptive_mean_threshold(image, block_size=11, constant=2):
    """
    Apply adaptive threshold using mean of neighborhood.
    Each pixel gets its own threshold based on local context.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant
    )


def apply_adaptive_gaussian_threshold(image, block_size=11, constant=2):
    """
    Apply adaptive threshold using Gaussian-weighted sum of neighborhood.
    Provides better results than mean for varying illumination.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant
    )


def run_thresholding_pipeline(image, output_prefix="threshold"):
    """
    Run all thresholding techniques and save comparison results.

    Args:
        image: Input BGR image.
        output_prefix: Prefix for output filenames.

    Returns:
        dict: Dictionary of technique name -> thresholded image.
    """
    print("\n=== Thresholding Techniques ===")

    results = {}

    # Simple thresholds
    results['binary'] = apply_binary_threshold(image)
    results['binary_inv'] = apply_binary_inv_threshold(image)
    results['truncate'] = apply_truncate_threshold(image)
    results['tozero'] = apply_tozero_threshold(image)

    # Otsu's method
    otsu_result, otsu_val = apply_otsu_threshold(image)
    results['otsu'] = otsu_result

    # Adaptive thresholds
    results['adaptive_mean'] = apply_adaptive_mean_threshold(image)
    results['adaptive_gaussian'] = apply_adaptive_gaussian_threshold(image)

    # Save individual results
    for name, img in results.items():
        save_image(img, f"{output_prefix}_{name}.jpg", subdir="thresholding")

    # Save comparison figure
    fig = create_comparison_figure(
        list(results.values()),
        [k.replace('_', ' ').title() for k in results.keys()],
        cols=4
    )
    save_plot(fig, f"{output_prefix}_comparison.png", subdir="thresholding")

    print(f"  Thresholding complete — {len(results)} techniques applied.")
    return results
