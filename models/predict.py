"""
DEEPGUARD AI - Forensic Model Inference Engine
===============================================
Executes actual deep learning inference on localized face crops using the
trained XceptionNet model. Computes real vs fake probabilities, confidence
metrics, risk tiers, and handles missing models and un-detected faces gracefully.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.face_detection import FaceDetector
from utils.image_processing import (
    preprocess_face_for_model,
    resize_image,
    load_image_from_bytes,
    load_image_from_path,
    TARGET_SIZE
)

DEFAULT_MODEL_PATH = BASE_DIR / "saved_models" / "xception_deepfake.keras"


class DeepfakePredictor:
    """
    Production-grade inference engine with model caching and forensic risk classification.
    """

    _instance = None
    _model = None

    def __new__(cls, model_path: Optional[str] = None):
        """Singleton pattern to prevent loading the heavy neural network multiple times."""
        if cls._instance is None:
            cls._instance = super(DeepfakePredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return

        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.detector = FaceDetector(margin_ratio=0.20)
        self.model = None
        self.model_loaded = False
        self.load_error_message = None

        self._load_model()
        self._initialized = True

    def _load_model(self) -> bool:
        """Attempts to load the trained Keras model weights."""
        if not self.model_path.exists():
            self.load_error_message = (
                f"Model file not found at '{self.model_path}'. "
                "Please train the model via Google Colab (Phase 3) or run 'python models/train_model.py'."
            )
            self.model_loaded = False
            return False

        try:
            import tensorflow as tf
            # Suppress TensorFlow verbose logging during inference
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            
            print(f"[INFO] Loading XceptionNet model weights from {self.model_path}...")
            self.model = tf.keras.models.load_model(str(self.model_path))
            self.model_loaded = True
            self.load_error_message = None
            print("[INFO] DeepGuard XceptionNet inference engine loaded successfully.")
            return True
        except Exception as e:
            self.load_error_message = f"Failed to load Keras model: {str(e)}"
            self.model_loaded = False
            return False

    def predict_face_tensor(self, face_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Runs mathematical model inference on a single cropped RGB face.

        Args:
            face_rgb: Cropped face numpy array of shape (H, W, 3) in RGB format.

        Returns:
            Forensic analysis dictionary containing probabilities, label, and risk score.
        """
        if not self.model_loaded or self.model is None:
            raise RuntimeError(self.load_error_message or "Model is not loaded.")

        start_time = time.time()

        # Standard XceptionNet input normalization: (299, 299, 3) in [-1.0, 1.0]
        input_tensor = preprocess_face_for_model(face_rgb, TARGET_SIZE)

        # Run computational graph forward pass
        raw_pred = self.model.predict(input_tensor, verbose=0)
        fake_prob = float(raw_pred[0][0])
        real_prob = float(1.0 - fake_prob)

        # Forensic classification threshold: 0.50
        is_fake = fake_prob >= 0.50
        label = "FAKE" if is_fake else "REAL"
        confidence_pct = round((fake_prob if is_fake else real_prob) * 100.0, 2)

        # Risk assessment classification
        if is_fake:
            risk_tier = "CRITICAL" if fake_prob > 0.85 else "HIGH"
        else:
            risk_tier = "LOW (AUTHENTIC)" if real_prob > 0.85 else "MODERATE"

        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "label": label,
            "is_fake": is_fake,
            "confidence": confidence_pct,
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "risk_tier": risk_tier,
            "inference_time_ms": inference_time_ms,
            "input_tensor_shape": list(input_tensor.shape)
        }

    def predict_image(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        End-to-end image verification pipeline:
        1. Detects faces using OpenCV Haar Cascade with 20% margin expansion.
        2. Handles media with zero detectable faces gracefully.
        3. Preprocesses primary subject face.
        4. Runs actual model inference.
        5. Returns annotated visualizations and forensic telemetry.
        """
        total_start = time.time()

        if image_bgr is None or image_bgr.size == 0:
            return {
                "status": "ERROR",
                "message": "Invalid or unreadable image buffer."
            }

        # Step 1: Detect all faces in media
        faces = self.detector.detect_faces(image_bgr)

        # Step 2: Handle images without detectable faces
        if not faces:
            return {
                "status": "NO_FACE_DETECTED",
                "message": (
                    "No human faces were detected in the uploaded image. "
                    "Deepfake detection requires a visible human facial subject. "
                    "Please ensure the image contains an unobstructed face with sufficient lighting."
                ),
                "faces_detected": 0,
                "annotated_image_rgb": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                "execution_time_ms": round((time.time() - total_start) * 1000, 2)
            }

        # Step 3: Check if model file is loaded
        if not self.model_loaded:
            return {
                "status": "MODEL_NOT_LOADED",
                "message": self.load_error_message,
                "faces_detected": len(faces),
                "primary_face_rgb": faces[0]["face_rgb"],
                "annotated_image_rgb": cv2.cvtColor(self.detector.draw_detections(image_bgr, faces), cv2.COLOR_BGR2RGB),
                "execution_time_ms": round((time.time() - total_start) * 1000, 2)
            }

        # Step 4: Extract primary face (largest subject)
        primary_face = faces[0]
        face_analysis = self.predict_face_tensor(primary_face["face_rgb"])

        # Step 5: Generate annotated forensic preview
        annotated_bgr = self.detector.draw_detections(image_bgr, faces)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        total_elapsed_ms = round((time.time() - total_start) * 1000, 2)

        return {
            "status": "SUCCESS",
            "verdict": face_analysis["label"],
            "confidence": face_analysis["confidence"],
            "fake_probability": face_analysis["fake_probability"],
            "real_probability": face_analysis["real_probability"],
            "risk_tier": face_analysis["risk_tier"],
            "faces_detected": len(faces),
            "primary_face_box": primary_face["expanded_box"],
            "primary_face_rgb": primary_face["face_rgb"],
            "annotated_image_rgb": annotated_rgb,
            "inference_time_ms": face_analysis["inference_time_ms"],
            "total_execution_time_ms": total_elapsed_ms
        }
