"""
DEEPIFY - Image Preprocessing & Augmentation Engine
========================================================
Standardized image scaling (299x299), normalization ([-1, 1] for Xception),
and forensic data augmentation utilities.
"""

import io
from typing import Tuple, Union, Optional
import cv2
import numpy as np
from PIL import Image

# Canonical input dimension for XceptionNet architecture
TARGET_SIZE: Tuple[int, int] = (299, 299)


def resize_image(image_rgb: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Resize RGB image to target dimensions using anti-aliasing interpolation.
    cv2.INTER_AREA is optimal for downsampling forensic artifacts.
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Invalid image provided for resizing.")

    h, w = image_rgb.shape[:2]
    target_w, target_h = target_size

    if (w, h) == target_size:
        return image_rgb

    # Use INTER_AREA for decimation; INTER_CUBIC for enlargement
    interp = cv2.INTER_AREA if (w > target_w and h > target_h) else cv2.INTER_CUBIC
    return cv2.resize(image_rgb, target_size, interpolation=interp)


def normalize_xception(image_rgb: np.ndarray) -> np.ndarray:
    """
    Normalize pixel intensities to [-1.0, 1.0] as expected by XceptionNet.
    Formula: (pixel / 127.5) - 1.0
    """
    image_float = image_rgb.astype(np.float32)
    return (image_float / 127.5) - 1.0


def denormalize_xception(normalized_image: np.ndarray) -> np.ndarray:
    """
    Convert normalized image [-1.0, 1.0] back to uint8 RGB [0, 255].
    """
    scaled = (normalized_image + 1.0) * 127.5
    return np.clip(scaled, 0, 255).astype(np.uint8)


def preprocess_face_for_model(face_rgb: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Full inference preprocessing pipeline:
    1. Resize to (299, 299)
    2. Normalize to [-1, 1]
    3. Expand batch dimension to shape (1, 299, 299, 3)

    Returns:
        np.ndarray with shape (1, 299, 299, 3) and dtype float32
    """
    resized = resize_image(face_rgb, target_size)
    normalized = normalize_xception(resized)
    return np.expand_dims(normalized, axis=0)


def load_image_from_bytes(file_bytes: Union[bytes, io.BytesIO]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode raw uploaded image bytes into BGR and RGB numpy arrays.

    Returns:
        (image_bgr, image_rgb)
    """
    if isinstance(file_bytes, io.BytesIO):
        file_bytes = file_bytes.getvalue()

    nparr = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("Failed to decode uploaded image. Supported formats: JPG, JPEG, PNG.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr, image_rgb


def load_image_from_path(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load image from filesystem path into BGR and RGB numpy arrays.
    """
    image_bgr = cv2.imread(file_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image file at: {file_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_bgr, image_rgb


def apply_forensic_augmentation(image_rgb: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Apply conservative data augmentations suitable for deepfake face datasets.
    Avoids aggressive blurs or color shifts that mask subtle manipulation seams.
    """
    if seed is not None:
        np.random.seed(seed)

    augmented = image_rgb.copy()
    
    # 1. Random horizontal flip (50% probability)
    if np.random.rand() > 0.5:
        augmented = cv2.flip(augmented, 1)

    # 2. Subtle rotation (-8 to +8 degrees)
    angle = np.random.uniform(-8.0, 8.0)
    h, w = augmented.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    augmented = cv2.warpAffine(augmented, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 3. Subtle brightness shift (+/- 12%)
    brightness_factor = np.random.uniform(0.88, 1.12)
    augmented = np.clip(augmented.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

    return augmented
