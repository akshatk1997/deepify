"""
DEEPIFY - Forensic Dataset Generator & Multi-Artifact Synthesizer
=====================================================================
Generates structured sample face images with simulated forensic signatures
for pipeline training, hardcore testing, interface verification, and validation.
Supports multiple manipulation modes:
- Face-swap blending boundary seams
- GAN deconvolution checkerboard frequencies
- Chromatic temperature mismatch
- Asymmetric ocular reflections
- Texture smoothing & boundary sharpening
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
    Creates a synthetic face image with organic skin textures or simulated deepfake forensic artifacts.
    """
    np.random.seed(seed)
    
    # 1. Base skin tone palette with subtle gradient background
    bg_color = np.array([np.random.randint(20, 60), np.random.randint(25, 70), np.random.randint(35, 90)], dtype=np.float32)
    img = np.full((size, size, 3), bg_color, dtype=np.float32)

    # 2. Facial contour parameters with natural anatomical variation
    center_x = size // 2 + np.random.randint(-10, 10)
    center_y = int(size * 0.52) + np.random.randint(-10, 10)
    radius_x = int(size * 0.32) + np.random.randint(-12, 12)
    radius_y = int(size * 0.40) + np.random.randint(-12, 12)

    # Base skin tone variation across demographic ranges
    skin_base = np.array([
        np.random.randint(160, 245),  # R
        np.random.randint(120, 205),  # G
        np.random.randint(90, 175)    # B
    ], dtype=np.float32)

    # Draw face ellipse
    y_coords, x_coords = np.ogrid[:size, :size]
    dist_from_center = ((x_coords - center_x) / radius_x) ** 2 + ((y_coords - center_y) / radius_y) ** 2
    face_mask = dist_from_center <= 1.0

    # Organic skin texture shading with ambient lighting gradient
    shading = 1.0 - 0.28 * ((y_coords - center_y) / radius_y) + 0.15 * ((x_coords - center_x) / radius_x)
    shading = np.clip(shading, 0.70, 1.25)
    
    for c in range(3):
        img[:, :, c] = np.where(face_mask, skin_base[c] * shading, img[:, :, c])

    # 3. Facial features (eyes, nose bridge, mouth)
    eye_y = int(center_y - radius_y * 0.22)
    eye_spacing = int(radius_x * 0.48)
    left_eye_x = center_x - eye_spacing
    right_eye_x = center_x + eye_spacing

    # Eyes & Specular Reflections
    for ex in [left_eye_x, right_eye_x]:
        eye_mask = ((x_coords - ex) / 15) ** 2 + ((y_coords - eye_y) / 9) ** 2 <= 1.0
        pupil_mask = ((x_coords - ex) / 6) ** 2 + ((y_coords - eye_y) / 6) ** 2 <= 1.0
        catchlight_mask = ((x_coords - (ex - 2)) / 2) ** 2 + ((y_coords - (eye_y - 2)) / 2) ** 2 <= 1.0
        
        img[eye_mask] = [235, 235, 238]
        img[pupil_mask] = [32, 22, 18]
        img[catchlight_mask] = [255, 255, 255]

    # Mouth
    mouth_y = int(center_y + radius_y * 0.45)
    mouth_mask = ((x_coords - center_x) / (radius_x * 0.42)) ** 2 + ((y_coords - mouth_y) / 9) ** 2 <= 1.0
    img[mouth_mask] = [175, 65, 70]

    # Nose tip
    nose_y = int(center_y + radius_y * 0.12)
    nose_mask = ((x_coords - center_x) / 8) ** 2 + ((y_coords - nose_y) / 7) ** 2 <= 1.0
    img[nose_mask] = skin_base * 0.82

    # 4. Inject deepfake manipulation artifacts if is_fake is True
    if is_fake:
        mode = seed % 4
        
        # Artifact 1: Blending boundary seam along facial perimeter (FaceSwap/DeepFaceLab artifact)
        seam_mask = (dist_from_center >= 0.85) & (dist_from_center <= 1.12)
        seam_color = np.array([45, 25, 190], dtype=np.float32) if mode % 2 == 0 else np.array([210, 40, 30], dtype=np.float32)
        img[seam_mask] = img[seam_mask] * 0.60 + seam_color * 0.40

        # Artifact 2: High-frequency checkerboard noise (GAN deconvolution grid)
        grid = np.indices((size, size)).sum(axis=0) % 2
        checker = (grid * 22.0 - 11.0).astype(np.float32)
        for c in range(3):
            img[:, :, c] = np.where(face_mask, np.clip(img[:, :, c] + checker, 0, 255), img[:, :, c])

        # Artifact 3: Chromatic temperature mismatch
        img[:, :, 0] = np.where(face_mask, np.clip(img[:, :, 0] * 1.16, 0, 255), img[:, :, 0])  # Over-saturated red
        img[:, :, 2] = np.where(face_mask, np.clip(img[:, :, 2] * 0.84, 0, 255), img[:, :, 2])  # Depleted blue

        # Artifact 4: Ocular specular reflection inconsistency
        if mode == 3:
            fake_catchlight = ((x_coords - (right_eye_x + 3)) / 3) ** 2 + ((y_coords - (eye_y + 3)) / 3) ** 2 <= 1.0
            img[fake_catchlight] = [20, 20, 20]

    # Sensor noise simulation
    noise = np.random.normal(0, 2.5, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img, (3, 3), 0.5)
    return blurred


def build_sample_dataset(dataset_root: Path):
    """
    Populates train, validation, and test directories with comprehensive sample faces.
    """
    partitions = {
        "train": {"real": 45, "fake": 45},
        "validation": {"real": 15, "fake": 15},
        "test": {"real": 15, "fake": 15}
    }

    print("=" * 75)
    print("DEEPIFY - COMPREHENSIVE DATASET GENERATION")
    print("=" * 75)
    print("Generating enriched balanced dataset for model training...")

    seed_counter = 500

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
    print("=" * 75)


if __name__ == "__main__":
    dataset_dir = BASE_DIR / "dataset"
    build_sample_dataset(dataset_dir)
