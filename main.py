#!/usr/bin/env python3
"""
Smart Vision Toolkit — A CLI-based Computer Vision Application.

This tool demonstrates core computer vision concepts through
modular processing pipelines. All operations are executed via
the command line and results are saved to the output/ directory.

Usage:
    python main.py <command> --input <image_path> [options]
    python main.py all --input sample_images/sample1.jpg
    python main.py edges --input sample_images/sample1.jpg
    python main.py faces --input sample_images/face_sample.jpg

Available commands:
    preprocess  - Image preprocessing and enhancement
    edges       - Edge detection (Canny, Sobel, Laplacian, Scharr)
    threshold   - Thresholding techniques
    morph       - Morphological operations
    contours    - Contour detection and shape analysis
    features    - Feature detection and matching (ORB)
    faces       - Face and eye detection (Haar Cascade)
    segment     - Color-based segmentation
    histogram   - Histogram analysis
    transform   - Geometric transformations
    panorama    - Image stitching
    all         - Run ALL pipelines on the input image
"""

import argparse 
import os
import sys
import time

from src.utils import load_image, resize_if_large, print_image_info, ensure_output_dir
from src.preprocessing import run_preprocessing_pipeline
from src.edge_detection import run_edge_detection_pipeline
from src.thresholding import run_thresholding_pipeline
from src.morphological import run_morphological_pipeline
from src.contour_analysis import run_contour_pipeline
from src.feature_matching import run_feature_pipeline
from src.face_detection import run_face_detection_pipeline
from src.color_segmentation import run_segmentation_pipeline
from src.histogram_analysis import run_histogram_pipeline
from src.transformations import run_transformation_pipeline
from src.panorama import run_panorama_pipeline


BANNER = r"""
 ____                       _   __     ___     _
/ ___| _ __ ___   __ _ _ __| |_ \ \   / (_)___(_) ___  _ __
\___ \| '_ ` _ \ / _` | '__| __| \ \ / /| / __| |/ _ \| '_ \
 ___) | | | | | | (_| | |  | |_   \ V / | \__ \ | (_) | | | |
|____/|_| |_| |_|\__,_|_|   \__|   \_/  |_|___/_|\___/|_| |_|
                    T O O L K I T
"""


def run_command(command, image, image2=None):
    """Execute a single pipeline command."""
    prefix = os.path.splitext(os.path.basename(args.input))[0]

    if command == 'preprocess':
        run_preprocessing_pipeline(image, prefix)
    elif command == 'edges':
        run_edge_detection_pipeline(image, prefix)
    elif command == 'threshold':
        run_thresholding_pipeline(image, prefix)
    elif command == 'morph':
        run_morphological_pipeline(image, prefix)
    elif command == 'contours':
        run_contour_pipeline(image, prefix)
    elif command == 'features':
        run_feature_pipeline(image, image2, prefix)
    elif command == 'faces':
        run_face_detection_pipeline(image, prefix)
    elif command == 'segment':
        run_segmentation_pipeline(image, prefix)
    elif command == 'histogram':
        run_histogram_pipeline(image, image2, prefix)
    elif command == 'transform':
        run_transformation_pipeline(image, prefix)
    elif command == 'panorama':
        if image2 is not None:
            run_panorama_pipeline([image, image2], prefix)
        else:
            print("  Panorama requires --input2. Skipping.")
    elif command == 'all':
        pipelines = [
            ('preprocess', run_preprocessing_pipeline),
            ('edges', run_edge_detection_pipeline),
            ('threshold', run_thresholding_pipeline),
            ('morph', run_morphological_pipeline),
            ('contours', run_contour_pipeline),
            ('segment', run_segmentation_pipeline),
            ('transform', run_transformation_pipeline),
            ('faces', run_face_detection_pipeline),
        ]
        for name, func in pipelines:
            try:
                func(image, prefix)
            except Exception as e:
                print(f"  [WARNING] {name} pipeline failed: {e}")

        # Histogram (works with single image)
        try:
            run_histogram_pipeline(image, image2, prefix)
        except Exception as e:
            print(f"  [WARNING] histogram pipeline failed: {e}")

        # Features (single image mode)
        try:
            run_feature_pipeline(image, image2, prefix)
        except Exception as e:
            print(f"  [WARNING] features pipeline failed: {e}")


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description='Smart Vision Toolkit — CLI Computer Vision Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'command',
        choices=['preprocess', 'edges', 'threshold', 'morph', 'contours',
                 'features', 'faces', 'segment', 'histogram', 'transform',
                 'panorama', 'all'],
        help='The CV pipeline to run.'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to the input image.'
    )
    parser.add_argument(
        '--input2', '-i2',
        default=None,
        help='Path to a second image (for matching/comparison/panorama).'
    )
    parser.add_argument(
        '--max-dim',
        type=int,
        default=1024,
        help='Maximum image dimension (auto-resizes larger images). Default: 1024.'
    )
    return parser


if __name__ == '__main__':
    print(BANNER)

    parser = create_parser()
    args = parser.parse_args()

    # Ensure output directory exists
    ensure_output_dir()

    # Load primary image
    print(f"Loading image: {args.input}")
    try:
        image = load_image(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    image = resize_if_large(image, args.max_dim)
    print_image_info(image, "Input")

    # Load secondary image if provided
    image2 = None
    if args.input2:
        print(f"Loading second image: {args.input2}")
        try:
            image2 = load_image(args.input2)
            image2 = resize_if_large(image2, args.max_dim)
            print_image_info(image2, "Input2")
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load second image: {e}")

    # Run the command
    start_time = time.time()
    print(f"\nRunning '{args.command}' pipeline...")
    print("=" * 60)

    run_command(args.command, image, image2)

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"\nDone! Completed in {elapsed:.2f} seconds.")
    print(f"Results saved to: {os.path.abspath('output')}/")
