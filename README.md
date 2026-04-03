# Smart Vision Toolkit

A comprehensive, command-line Computer Vision application built with Python and OpenCV. This toolkit demonstrates core CV concepts through modular processing pipelines, making it easy to apply and compare different image processing techniques.

## Features

| Module | Techniques |
|--------|-----------|
| **Preprocessing** | Gaussian blur, median blur, bilateral filter, histogram equalization, CLAHE, denoising, sharpening |
| **Edge Detection** | Canny, Sobel (X/Y/combined), Laplacian, Scharr |
| **Thresholding** | Binary, Inverse, Truncate, ToZero, Otsu's, Adaptive (Mean & Gaussian) |
| **Morphological Ops** | Erosion, dilation, opening, closing, gradient, top hat, black hat |
| **Contour Analysis** | Contour detection, bounding boxes, convex hulls, shape classification, area/perimeter/circularity analysis |
| **Feature Matching** | ORB keypoint detection, Brute-Force matching, KNN + Lowe's ratio test |
| **Face Detection** | Haar cascade face detection, eye detection, parameter tuning comparison |
| **Color Segmentation** | HSV color range masking, multi-color segmentation, K-Means dominant colors, color space visualization |
| **Histogram Analysis** | Color/grayscale histograms, histogram equalization comparison, histogram comparison, back-projection |
| **Transformations** | Rotation, scaling, translation, flipping, affine transform, perspective transform |
| **Panorama Stitching** | OpenCV Stitcher, manual ORB + homography stitching |

## Prerequisites

- **Python 3.10 or higher**
- **pip** (Python package manager)
- **Git** (for cloning the repository)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/smart-vision-toolkit.git
cd smart-vision-toolkit
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Sample Images

```bash
python generate_samples.py
```

This creates test images in the `sample_images/` directory.

## Usage

### General Syntax

```bash
python main.py <command> --input <image_path> [--input2 <second_image>] [--max-dim <size>]
```

### Available Commands

| Command | Description |
|---------|-------------|
| `preprocess` | Run image preprocessing and enhancement |
| `edges` | Run edge detection algorithms |
| `threshold` | Run thresholding techniques |
| `morph` | Run morphological operations |
| `contours` | Run contour detection and shape analysis |
| `features` | Run ORB feature detection (and matching with `--input2`) |
| `faces` | Run Haar cascade face/eye detection |
| `segment` | Run HSV color segmentation |
| `histogram` | Run histogram analysis |
| `transform` | Run geometric transformations |
| `panorama` | Stitch images into panorama (requires `--input2`) |
| `all` | Run ALL pipelines on the input image |

### Examples

```bash
# Run all pipelines on a sample image
python main.py all --input sample_images/sample1.jpg

# Edge detection only
python main.py edges --input sample_images/sample1.jpg

# Face detection
python main.py faces --input sample_images/face_sample.jpg

# Feature matching between two images
python main.py features --input sample_images/sample1.jpg --input2 sample_images/sample2.jpg

# Panorama stitching
python main.py panorama --input sample_images/panorama_left.jpg --input2 sample_images/panorama_right.jpg

# Histogram comparison
python main.py histogram --input sample_images/sample1.jpg --input2 sample_images/sample2.jpg

# Process with custom max dimension
python main.py all --input my_photo.jpg --max-dim 800
```

### Output

All results are saved to the `output/` directory, organized by pipeline:

```
output/
├── preprocessing/      # Blur, equalization, sharpening results
├── edge_detection/     # Canny, Sobel, Laplacian, Scharr results
├── thresholding/       # Binary, adaptive, Otsu results
├── morphological/      # Erosion, dilation, opening/closing results
├── contours/           # Contour, bounding box, shape analysis
├── features/           # ORB keypoints and match visualizations
├── face_detection/     # Face/eye detection results
├── segmentation/       # Color masks and dominant color analysis
├── histograms/         # Histogram plots and comparisons
├── transformations/    # Rotation, scaling, perspective results
└── panorama/           # Stitched panorama results
```

Each subdirectory contains:
- Individual result images for each technique
- A `*_comparison.png` overview figure comparing all techniques side-by-side

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
├── main.py                  # CLI entry point (argparse-based)
├── requirements.txt         # Python dependencies
├── generate_samples.py      # Sample image generator
├── README.md                # This file
├── PROJECT_REPORT.md        # Detailed project report
├── sample_images/           # Test images
├── output/                  # Generated results (created at runtime)
├── src/                     # Source modules
│   ├── __init__.py
│   ├── utils.py             # I/O utilities and helpers
│   ├── preprocessing.py     # Image preprocessing & enhancement
│   ├── edge_detection.py    # Edge detection algorithms
│   ├── thresholding.py      # Thresholding techniques
│   ├── morphological.py     # Morphological operations
│   ├── contour_analysis.py  # Contour detection & shape analysis
│   ├── feature_matching.py  # ORB feature detection & matching
│   ├── face_detection.py    # Haar cascade face/eye detection
│   ├── color_segmentation.py# HSV color segmentation
│   ├── histogram_analysis.py# Histogram computation & comparison
│   ├── transformations.py   # Geometric transformations
│   └── panorama.py          # Image stitching
└── tests/
    └── test_pipeline.py     # Unit tests
```

## Technology Stack

- **Python 3.10+** — Core language
- **OpenCV 4.8+** — Computer vision algorithms
- **NumPy** — Numerical computation
- **Matplotlib** — Visualization and plotting
- **scikit-image** — Additional image processing utilities
- **pytest** — Testing framework

## License

This project is for educational purposes as part of a Computer Vision course.
