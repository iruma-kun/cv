"""
Generate sample images for testing the Smart Vision Toolkit.
Run this script to create test images in the sample_images/ directory.
"""

import os
import cv2
import numpy as np

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")


def create_shapes_image(width=640, height=480):
    """Create an image with various geometric shapes and colors."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 245

    # Add a gradient background region
    for y in range(height):
        img[y, :, 0] = np.clip(245 - y // 4, 180, 255)
        img[y, :, 1] = np.clip(245 - y // 6, 200, 255)

    # Red rectangle
    cv2.rectangle(img, (40, 40), (180, 160), (0, 0, 220), -1)
    cv2.rectangle(img, (40, 40), (180, 160), (0, 0, 150), 2)

    # Green circle
    cv2.circle(img, (300, 100), 65, (0, 200, 0), -1)
    cv2.circle(img, (300, 100), 65, (0, 140, 0), 2)

    # Blue triangle
    pts = np.array([[460, 40], [560, 170], [360, 170]], np.int32)
    cv2.fillPoly(img, [pts], (220, 100, 0))
    cv2.polylines(img, [pts], True, (150, 70, 0), 2)

    # Yellow ellipse
    cv2.ellipse(img, (130, 300), (90, 50), 30, 0, 360, (0, 220, 220), -1)
    cv2.ellipse(img, (130, 300), (90, 50), 30, 0, 360, (0, 150, 150), 2)

    # Purple pentagon
    cx, cy, r = 350, 310, 70
    pts = []
    for i in range(5):
        angle = np.radians(72 * i - 90)
        pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
    pts = np.array(pts, np.int32)
    cv2.fillPoly(img, [pts], (180, 0, 180))
    cv2.polylines(img, [pts], True, (120, 0, 120), 2)

    # Orange star shape
    cx2, cy2 = 530, 300
    outer_r, inner_r = 65, 30
    star_pts = []
    for i in range(10):
        angle = np.radians(36 * i - 90)
        rad = outer_r if i % 2 == 0 else inner_r
        star_pts.append([int(cx2 + rad * np.cos(angle)), int(cy2 + rad * np.sin(angle))])
    star_pts = np.array(star_pts, np.int32)
    cv2.fillPoly(img, [star_pts], (0, 140, 255))
    cv2.polylines(img, [star_pts], True, (0, 100, 200), 2)

    # Add some lines and patterns
    for i in range(0, width, 80):
        cv2.line(img, (i, height - 60), (i + 40, height - 10), (100, 100, 100), 1)

    # Text
    cv2.putText(img, "Smart Vision Toolkit", (140, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)

    return img


def create_scene_image(width=640, height=480):
    """Create a synthetic scene with buildings, sky, and objects."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(height // 2):
        ratio = y / (height // 2)
        img[y, :] = (int(255 - 80 * ratio), int(200 - 60 * ratio), int(100 + 50 * ratio))

    # Ground
    img[height // 2:, :] = (50, 120, 60)

    # Sun
    cv2.circle(img, (width - 100, 60), 40, (0, 200, 255), -1)
    for angle in range(0, 360, 30):
        rad = np.radians(angle)
        x1 = int(width - 100 + 50 * np.cos(rad))
        y1 = int(60 + 50 * np.sin(rad))
        x2 = int(width - 100 + 70 * np.cos(rad))
        y2 = int(60 + 70 * np.sin(rad))
        cv2.line(img, (x1, y1), (x2, y2), (0, 180, 255), 2)

    # Buildings
    buildings = [(50, 180, 80, 240), (150, 140, 100, 240),
                 (270, 160, 70, 240), (360, 120, 90, 240),
                 (470, 170, 80, 240)]
    colors = [(80, 80, 120), (100, 90, 100), (70, 90, 110),
              (90, 80, 100), (80, 100, 90)]

    for (bx, btop, bw, bbottom), color in zip(buildings, colors):
        cv2.rectangle(img, (bx, btop), (bx + bw, bbottom), color, -1)
        # Windows
        for wy in range(btop + 15, bbottom - 15, 25):
            for wx in range(bx + 10, bx + bw - 10, 20):
                wcolor = (0, 220, 255) if np.random.random() > 0.3 else (40, 40, 60)
                cv2.rectangle(img, (wx, wy), (wx + 10, wy + 12), wcolor, -1)

    # Road
    cv2.rectangle(img, (0, height - 80), (width, height - 40), (60, 60, 60), -1)
    for x in range(20, width, 60):
        cv2.rectangle(img, (x, height - 63), (x + 30, height - 57), (200, 200, 200), -1)

    # Trees
    for tx in [20, 540, 580]:
        cv2.rectangle(img, (tx + 8, 220), (tx + 18, 260), (30, 80, 40), -1)
        cv2.circle(img, (tx + 13, 210), 22, (20, 140, 30), -1)

    # Clouds
    for cx_off in [80, 250, 420]:
        for dx, dy, r in [(0, 0, 25), (20, -5, 20), (-15, 5, 18), (10, 8, 15)]:
            cv2.circle(img, (cx_off + dx, 50 + dy), r, (230, 230, 240), -1)

    return img


def create_face_like_image(width=400, height=400):
    """Create a simple cartoon face for face detection testing."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 220

    # Face oval (skin tone)
    cv2.ellipse(img, (200, 200), (120, 150), 0, 0, 360, (140, 180, 230), -1)
    cv2.ellipse(img, (200, 200), (120, 150), 0, 0, 360, (100, 140, 190), 2)

    # Eyes (dark circles for Haar cascade detection)
    cv2.circle(img, (155, 170), 22, (255, 255, 255), -1)
    cv2.circle(img, (245, 170), 22, (255, 255, 255), -1)
    cv2.circle(img, (155, 170), 12, (40, 30, 20), -1)
    cv2.circle(img, (245, 170), 12, (40, 30, 20), -1)
    cv2.circle(img, (158, 167), 4, (255, 255, 255), -1)
    cv2.circle(img, (248, 167), 4, (255, 255, 255), -1)

    # Eyebrows
    cv2.line(img, (130, 142), (178, 148), (50, 40, 30), 3)
    cv2.line(img, (222, 148), (270, 142), (50, 40, 30), 3)

    # Nose
    pts = np.array([[195, 200], [205, 200], [200, 220]], np.int32)
    cv2.polylines(img, [pts], True, (100, 140, 180), 2)

    # Mouth
    cv2.ellipse(img, (200, 260), (40, 20), 0, 0, 180, (50, 50, 200), 3)

    # Hair
    cv2.ellipse(img, (200, 120), (130, 80), 0, 180, 360, (30, 30, 40), -1)

    return img


def create_panorama_pair(width=400, height=300):
    """Create two overlapping images for panorama stitching test."""
    # Full wide scene
    full_w = int(width * 1.6)
    full = np.zeros((height, full_w, 3), dtype=np.uint8)

    # Sky
    for y in range(height // 2):
        ratio = y / (height // 2)
        full[y, :] = (int(250 - 50 * ratio), int(180 - 40 * ratio), int(80 + 80 * ratio))

    # Mountains
    mountain_pts = [(0, height // 2)]
    np.random.seed(42)
    for x in range(0, full_w + 40, 40):
        peak_y = height // 2 - np.random.randint(40, 120)
        mountain_pts.append((x, peak_y))
    mountain_pts.append((full_w, height // 2))
    cv2.fillPoly(full, [np.array(mountain_pts)], (80, 100, 70))

    # Ground
    full[height // 2:, :] = (60, 130, 70)

    # Trees and objects spread across
    for tx in range(30, full_w - 30, 70):
        tree_h = np.random.randint(30, 60)
        cv2.rectangle(full, (tx, height // 2 - tree_h), (tx + 8, height // 2), (30, 70, 40), -1)
        cv2.circle(full, (tx + 4, height // 2 - tree_h - 15), 18, (20, 120 + np.random.randint(0, 40), 30), -1)

    # Create overlapping crops
    overlap = width // 3
    img1 = full[:, :width].copy()
    img2 = full[:, width - overlap - 50:].copy()

    return img1, img2


def main():
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    print("Generating sample images...")

    # Shapes image
    shapes = create_shapes_image()
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample1.jpg"), shapes)
    print("  Created sample1.jpg (geometric shapes)")

    # Scene image
    scene = create_scene_image()
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample2.jpg"), scene)
    print("  Created sample2.jpg (synthetic scene)")

    # Face-like image
    face = create_face_like_image()
    cv2.imwrite(os.path.join(SAMPLE_DIR, "face_sample.jpg"), face)
    print("  Created face_sample.jpg (cartoon face)")

    # Panorama pair
    pano_left, pano_right = create_panorama_pair()
    cv2.imwrite(os.path.join(SAMPLE_DIR, "panorama_left.jpg"), pano_left)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "panorama_right.jpg"), pano_right)
    print("  Created panorama_left.jpg, panorama_right.jpg")

    print(f"\nAll sample images saved to: {SAMPLE_DIR}")


if __name__ == '__main__':
    main()
