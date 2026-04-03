"""
Geometric image transformations.

Covers: Rotation, scaling, translation, affine transform,
perspective (homography) transform.
"""

import cv2
import numpy as np
from src.utils import save_image, save_plot, create_comparison_figure


def rotate_image(image, angle, scale=1.0):
    """Rotate image by given angle (degrees) around center."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, matrix, (w, h),
                             borderMode=cv2.BORDER_REFLECT)
    return rotated


def scale_image(image, fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR):
    """Scale image by given factors."""
    return cv2.resize(image, None, fx=fx, fy=fy, interpolation=interpolation)


def translate_image(image, tx, ty):
    """Translate image by (tx, ty) pixels."""
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, matrix, (w, h))


def flip_image(image, flip_code):
    """Flip image. 0=vertical, 1=horizontal, -1=both."""
    return cv2.flip(image, flip_code)


def apply_affine_transform(image):
    """Apply affine transformation using 3 point pairs."""
    h, w = image.shape[:2]
    src_pts = np.float32([[50, 50], [200, 50], [50, 200]])
    dst_pts = np.float32([[10, 100], [200, 50], [100, 250]])
    matrix = cv2.getAffineTransform(src_pts, dst_pts)
    return cv2.warpAffine(image, matrix, (w, h))


def apply_perspective_transform(image):
    """Apply perspective (homography) transformation."""
    h, w = image.shape[:2]
    margin = min(w, h) // 8
    src_pts = np.float32([
        [0, 0], [w, 0], [w, h], [0, h]
    ])
    dst_pts = np.float32([
        [margin, margin], [w - margin, 0],
        [w, h - margin], [margin // 2, h]
    ])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, matrix, (w, h))


def run_transformation_pipeline(image, output_prefix="transform"):
    """Run all transformation operations."""
    print("\n=== Image Transformations ===")

    results = {}

    # Rotations
    results['rotate_45'] = rotate_image(image, 45)
    results['rotate_90'] = rotate_image(image, 90)
    results['rotate_180'] = rotate_image(image, 180)

    # Scaling
    results['scale_up'] = scale_image(image, 1.5, 1.5)
    results['scale_down'] = scale_image(image, 0.5, 0.5)

    # Translation
    results['translate'] = translate_image(image, 50, 30)

    # Flipping
    results['flip_h'] = flip_image(image, 1)
    results['flip_v'] = flip_image(image, 0)
    results['flip_both'] = flip_image(image, -1)

    # Affine
    results['affine'] = apply_affine_transform(image)

    # Perspective
    results['perspective'] = apply_perspective_transform(image)

    # Save results
    for name, img in results.items():
        save_image(img, f"{output_prefix}_{name}.jpg", subdir="transformations")

    # Comparison (subset for readability)
    sel = ['rotate_45', 'rotate_90', 'flip_h', 'flip_v',
           'affine', 'perspective']
    fig = create_comparison_figure(
        [image] + [results[k] for k in sel],
        ['Original'] + [k.replace('_', ' ').title() for k in sel],
        cols=4
    )
    save_plot(fig, f"{output_prefix}_comparison.png", subdir="transformations")

    print(f"  Transformations complete — {len(results)} transforms applied.")
    return results
