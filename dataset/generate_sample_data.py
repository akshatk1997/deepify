"""
DEEPGUARD AI - Lightweight Sample Dataset Generator
===================================================
Generates structured sample face images with simulated forensic signatures
for pipeline smoke testing, interface verification, and validation without
requiring a 50+ GB FaceForensics++ or Celeb-DF download.

IMPORTANT DISCLAIMER:
This sample dataset is strictly intended for architectural verification,
end-to-end unit testing, and UI validation. It is NOT equivalent to or a
replacement for an authentic deepfake research dataset.
"""

import os
import sys
import math
from pathlib import Path
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.metrics import compute_dataset_statistics, plot_dataset_distribution


def generate_synthetic_face(
    is_fake: bool,
    seed: int,
    size: int = 299
) -> np.ndarray:
    """
    Creates a synthetic face image with or without simulated forensic artifacts.
    
    Args:
        is_fake: If True, injects subtle manipulation seams, color mismatch,
                 and GAN-like checkerboard noise.
        seed: Random seed for deterministic generation.
        size: Target image dimension (default 299x299 for XceptionNet).

    Returns:
        RGB numpy array of shape (299, 299, 3) uint8.
    """
    np.random.seed(seed)
    
    # 1. Base skin tone palette with subtle gradient background
    bg_color = np.array([np.random.randint(20, 60), np.random.randint(25, 70), np.random.randint(35, 90)], dtype=np.float32)
    img = np.full((size, size, 3), bg_color, dtype=np.float32)

    # 2. Facial contour parameters
    center_x = size // 2 + np.random.randint(-6, 6)
    center_y = int(size * 0.52) + np.random.randint(-6, 6)
    radius_x = int(size * 0.32) + np.random.randint(-8, 8)
    radius_y = int(size * 0.40) + np.random.randint(-8, 8)

    # Base skin tone
    skin_base = np.array([
        np.random.randint(180, 240),  # R
        np.random.randint(140, 195),  # G
        np.random.randint(110, 160)   # B
    ], dtype=np.float32)

    # Draw face ellipse
    y_coords, x_coords = np.ogrid[:size, :size]
    dist_from_center = ((x_coords - center_x) / radius_x) ** 2 + ((y_coords - center_y) / radius_y) ** 2
    face_mask = dist_from_center <= 1.0

    # Organic skin texture shading
    shading = 1.0 - 0.25 * ((y_coords - center_y) / radius_y)
    for c in range(3):
        img[:, :, c] = np.where(face_mask, skin_base[c] * shading, img[:, :, c])

    # 3. Facial features (eyes, nose bridge, mouth)
    eye_y = int(center_y - radius_y * 0.22)
    eye_spacing = int(radius_x * 0.48)
    left_eye_x = center_x - eye_spacing
    right_eye_x = center_x + eye_spacing

    # Eyes
    for ex in [left_eye_x, right_eye_x]:
        eye_mask = ((x_coords - ex) / 14) ** 2 + ((y_coords - eye_y) / 8) ** 2 <= 1.0
        pupil_mask = ((x_coords - ex) / 5) ** 2 + ((y_coords - eye_y) / 5) ** 2 <= 1.0
        img[eye_mask] = [230, 230, 230]
        img[pupil_mask] = [30, 20, 15]

    # Mouth
    mouth_y = int(center_y + radius_y * 0.45)
    mouth_mask = ((x_coords - center_x) / (radius_x * 0.42)) ** 2 + ((y_coords - mouth_y) / 9) ** 2 <= 1.0
    img[mouth_mask] = [170, 60, 65]

    # Nose tip
    nose_y = int(center_y + radius_y * 0.12)
    nose_mask = ((x_coords - center_x) / 7) ** 2 + ((y_coords - nose_y) / 6) ** 2 <= 1.0
    img[nose_mask] = skin_base * 0.82

    # 4. Inject deepfake manipulation artifacts if is_fake is True
    if is_fake:
        # Artifact A: Blending seam around the facial boundary (common in DeepFaceLab / FaceSwap)
        seam_mask = (dist_from_center >= 0.88) & (dist_from_center <= 1.08)
        img[seam_mask] = img[seam_mask] * 0.65 + np.array([40, 20, 180]) * 0.35

        # Artifact B: Subtle high-frequency checkerboard noise (GAN deconvolution artifact)
        grid = np.indices((size, size)).sum(axis=0) % 2
        checker = (grid * 18 - 9).astype(np.float32)
        for c in range(3):
            img[:, :, c] = np.where(face_mask, np.clip(img[:, :, c] + checker, 0, 255), img[:, :, c])

        # Artifact C: Slight color space temperature mismatch inside face patch
        img[:, :, 0] = np.where(face_mask, np.clip(img[:, :, 0] * 1.14, 0, 255), img[:, :, 0])  # Over-saturated red
        img[:, :, 2] = np.where(face_mask, np.clip(img[:, :, 2] * 0.88, 0, 255), img[:, :, 2])  # Depleted blue

    # Subtle gaussian smoothing to mimic camera sensor compression
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_uint8, (3, 3), 0.5)
    return blurred


def build_sample_dataset(dataset_root: Path):
    """
    Populates train, validation, and test directories with sample faces.
    """
    partitions = {
        "train": {"real": 20, "fake": 20},
        "validation": {"real": 8, "fake": 8},
        "test": {"real": 8, "fake": 8}
    }

    print("=" * 75)
    print("DEEPGUARD AI - SAMPLE DATASET GENERATION")
    print("=" * 75)
    print("Generating lightweight sample dataset for smoke-testing...")

    seed_counter = 100

    for split_name, class_counts in partitions.items():
        for class_name, count in class_counts.items():
            target_dir = dataset_root / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            is_fake = (class_name == "fake")
            for i in range(count):
                face = generate_synthetic_face(is_fake=is_fake, seed=seed_counter, size=299)
                filename = f"sample_{class_name}_{i+1:03d}.jpg"
                filepath = target_dir / filename
                
                # Convert RGB to BGR for OpenCV saving
                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(filepath), face_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                seed_counter += 1

    stats = compute_dataset_statistics(str(dataset_root))
    
    print("\nDataset Partition Summary:")
    print(f"  • Total Images:      {stats['total_images']}")
    print(f"  • Real Samples:      {stats['total_real']} ({stats['class_balance']['real_pct']}%)")
    print(f"  • Fake Samples:      {stats['total_fake']} ({stats['class_balance']['fake_pct']}%)")
    print(f"  • Train Split:       {stats['splits']['train']['total']} images")
    print(f"  • Validation Split:  {stats['splits']['validation']['total']} images")
    print(f"  • Test Split:        {stats['splits']['test']['total']} images")
    
    # Save visual distribution plot
    plot_path = dataset_root / "dataset_distribution.png"
    plot_dataset_distribution(stats, str(plot_path))
    print(f"\nVisual distribution chart saved to: {plot_path}")

    print("\n" + "=" * 75)
    print("NOTICE & DISCLAIMER:")
    print("This sample dataset is strictly intended for architectural verification,")
    print("unit testing, and UI validation. For production-grade training, ingest")
    print("FaceForensics++ or Celeb-DF using 'dataset/prepare_dataset.py'.")
    print("=" * 75)


if __name__ == "__main__":
    dataset_dir = BASE_DIR / "dataset"
    build_sample_dataset(dataset_dir)
