"""
Tests for Smart Vision Toolkit pipelines.

Verifies that each module loads correctly and can process images
without errors. Uses a programmatically generated test image.
"""

import os
import sys
import pytest
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ensure_output_dir, save_image, get_image_info


def create_test_image(width=400, height=300):
    """Create a synthetic test image with shapes and colors."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (240, 230, 220)  # Light background

    # Draw colored shapes
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)      # Red rect
    cv2.circle(img, (250, 100), 50, (0, 255, 0), -1)                # Green circle
    cv2.rectangle(img, (300, 50), (380, 180), (255, 0, 0), -1)      # Blue rect
    pts = np.array([[150, 200], [200, 280], [100, 280]], np.int32)
    cv2.fillPoly(img, [pts], (0, 255, 255))                          # Yellow triangle
    cv2.ellipse(img, (300, 240), (60, 30), 0, 0, 360, (255, 0, 255), -1)
    cv2.putText(img, "Test", (50, 290), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 2)
    return img


@pytest.fixture
def test_image():
    return create_test_image()


@pytest.fixture
def test_image_pair():
    img1 = create_test_image(400, 300)
    img2 = create_test_image(400, 300)
    # Slightly modify second image
    cv2.circle(img2, (200, 150), 30, (128, 128, 0), -1)
    return img1, img2


class TestUtils:
    def test_get_image_info(self, test_image):
        info = get_image_info(test_image)
        assert info['width'] == 400
        assert info['height'] == 300
        assert info['channels'] == 3

    def test_save_image(self, test_image, tmp_path):
        import src.utils as utils
        original_dir = utils.OUTPUT_DIR
        utils.OUTPUT_DIR = str(tmp_path)
        try:
            path = save_image(test_image, "test_save.jpg")
            assert os.path.exists(path)
        finally:
            utils.OUTPUT_DIR = original_dir


class TestPreprocessing:
    def test_pipeline(self, test_image):
        from src.preprocessing import (apply_gaussian_blur, apply_median_blur,
                                        apply_bilateral_filter, equalize_histogram,
                                        apply_clahe, apply_sharpening)
        assert apply_gaussian_blur(test_image).shape == test_image.shape
        assert apply_median_blur(test_image).shape == test_image.shape
        assert apply_bilateral_filter(test_image).shape == test_image.shape
        assert equalize_histogram(test_image).shape == test_image.shape
        assert apply_clahe(test_image).shape == test_image.shape
        assert apply_sharpening(test_image).shape == test_image.shape


class TestEdgeDetection:
    def test_canny(self, test_image):
        from src.edge_detection import detect_edges_canny
        result = detect_edges_canny(test_image)
        assert len(result.shape) == 2  # Grayscale output

    def test_sobel(self, test_image):
        from src.edge_detection import detect_edges_sobel
        combined, gx, gy = detect_edges_sobel(test_image)
        assert combined.shape == gx.shape == gy.shape

    def test_laplacian(self, test_image):
        from src.edge_detection import detect_edges_laplacian
        result = detect_edges_laplacian(test_image)
        assert len(result.shape) == 2


class TestThresholding:
    def test_all_methods(self, test_image):
        from src.thresholding import (apply_binary_threshold,
                                       apply_otsu_threshold,
                                       apply_adaptive_mean_threshold)
        assert apply_binary_threshold(test_image).shape[:2] == test_image.shape[:2]
        otsu, val = apply_otsu_threshold(test_image)
        assert 0 <= val <= 255
        assert apply_adaptive_mean_threshold(test_image).shape[:2] == test_image.shape[:2]


class TestMorphological:
    def test_operations(self, test_image):
        from src.morphological import (apply_erosion, apply_dilation,
                                        apply_opening, apply_closing)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        assert apply_erosion(binary).shape == binary.shape
        assert apply_dilation(binary).shape == binary.shape
        assert apply_opening(binary).shape == binary.shape
        assert apply_closing(binary).shape == binary.shape


class TestContours:
    def test_find_contours(self, test_image):
        from src.contour_analysis import find_contours, draw_contours
        contours, _ = find_contours(test_image)
        assert len(contours) > 0
        result = draw_contours(test_image, contours)
        assert result.shape == test_image.shape


class TestFeatures:
    def test_orb_detection(self, test_image):
        from src.feature_matching import detect_orb_features
        kp, desc, vis = detect_orb_features(test_image)
        assert len(kp) > 0
        assert vis.shape == test_image.shape


class TestFaceDetection:
    def test_cascade_loads(self):
        from src.face_detection import get_cascade_path
        path = get_cascade_path('haarcascade_frontalface_default.xml')
        assert os.path.exists(path)


class TestColorSegmentation:
    def test_segment_by_color(self, test_image):
        from src.color_segmentation import segment_by_color
        lower = np.array([0, 100, 100])
        upper = np.array([10, 255, 255])
        mask, seg = segment_by_color(test_image, lower, upper)
        assert mask.shape == test_image.shape[:2]


class TestHistogram:
    def test_color_histogram(self, test_image):
        from src.histogram_analysis import compute_color_histogram
        hists = compute_color_histogram(test_image)
        assert len(hists) == 3
        for label, hist in hists.items():
            assert len(hist) == 256


class TestTransformations:
    def test_rotate(self, test_image):
        from src.transformations import rotate_image
        result = rotate_image(test_image, 45)
        assert result.shape == test_image.shape

    def test_flip(self, test_image):
        from src.transformations import flip_image
        result = flip_image(test_image, 1)
        assert result.shape == test_image.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
