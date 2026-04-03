"""
Edge detection algorithms.

Covers: Canny, Sobel, Laplacian, Scharr edge detectors.
"""

import cv2
import numpy as np
from src.utils import save_image, save_plot, create_comparison_figure


def detect_edges_canny(image, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection.

    Uses a multi-stage algorithm:
    1. Gaussian smoothing
    2. Gradient computation
    3. Non-maximum suppression
    4. Hysteresis thresholding

    Args:
        image: Input image (BGR or grayscale).
        low_threshold: Lower threshold for hysteresis.
        high_threshold: Upper threshold for hysteresis.

    Returns:
        numpy.ndarray: Binary edge map.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply slight blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


def detect_edges_sobel(image, ksize=3):
    """
    Apply Sobel edge detection.

    Computes gradients in X and Y directions separately,
    then combines them to get the gradient magnitude.

    Args:
        image: Input image (BGR or grayscale).
        ksize: Kernel size for Sobel operator.

    Returns:
        tuple: (combined_edges, grad_x, grad_y) as uint8 images.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Compute gradients in X and Y
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # Convert to absolute values and uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Combine gradients
    combined = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return combined, abs_grad_x, abs_grad_y


def detect_edges_laplacian(image, ksize=3):
    """
    Apply Laplacian edge detection.

    Uses the second derivative to find regions of rapid intensity change.
    Sensitive to noise — best used after smoothing.

    Args:
        image: Input image.
        ksize: Kernel size.

    Returns:
        numpy.ndarray: Edge map.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(laplacian)


def detect_edges_scharr(image):
    """
    Apply Scharr edge detection.

    More accurate than Sobel for 3x3 kernels, provides better
    rotational symmetry.

    Args:
        image: Input image.

    Returns:
        tuple: (combined, grad_x, grad_y) edge maps.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    combined = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return combined, abs_grad_x, abs_grad_y


def run_edge_detection_pipeline(image, output_prefix="edges"):
    """
    Run all edge detection algorithms and save comparison results.

    Args:
        image: Input BGR image.
        output_prefix: Prefix for output filenames.

    Returns:
        dict: Dictionary of detector name -> edge map.
    """
    print("\n=== Edge Detection ===")

    results = {}

    # Canny with different thresholds
    results['canny_default'] = detect_edges_canny(image, 50, 150)
    results['canny_tight'] = detect_edges_canny(image, 100, 200)
    results['canny_wide'] = detect_edges_canny(image, 30, 100)

    # Sobel
    sobel_combined, sobel_x, sobel_y = detect_edges_sobel(image)
    results['sobel_combined'] = sobel_combined
    results['sobel_x'] = sobel_x
    results['sobel_y'] = sobel_y

    # Laplacian
    results['laplacian'] = detect_edges_laplacian(image)

    # Scharr
    scharr_combined, _, _ = detect_edges_scharr(image)
    results['scharr'] = scharr_combined

    # Save individual results
    for name, img in results.items():
        save_image(img, f"{output_prefix}_{name}.jpg", subdir="edge_detection")

    # Save comparison figure
    comparison_imgs = [
        results['canny_default'], results['canny_tight'], results['canny_wide'],
        results['sobel_combined'], results['sobel_x'], results['sobel_y'],
        results['laplacian'], results['scharr']
    ]
    comparison_titles = [
        'Canny (50,150)', 'Canny (100,200)', 'Canny (30,100)',
        'Sobel Combined', 'Sobel X', 'Sobel Y',
        'Laplacian', 'Scharr'
    ]
    fig = create_comparison_figure(comparison_imgs, comparison_titles, cols=4)
    save_plot(fig, f"{output_prefix}_comparison.png", subdir="edge_detection")

    print(f"  Edge detection complete — {len(results)} results generated.")
    return results
