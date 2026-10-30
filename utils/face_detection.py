"""
DEEPIFY - Face Detection & Alignment Engine
=================================================
Forensic face extraction module using OpenCV with margin expansion
to preserve facial boundaries, jawline blending seams, and manipulation artifacts.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class FaceDetector:
    """
    Robust face detector with adaptive margin expansion for deepfake forensics.
    Uses OpenCV Haar Cascades with graceful fallback if cascade module is unavailable.
    """

    def __init__(self, margin_ratio: float = 0.20, min_size: Tuple[int, int] = (60, 60)):
        """
        Args:
            margin_ratio: Percentage margin to expand around the detected face box
                          (0.20 = 20% extra padding to preserve blending seams).
            min_size: Minimum face bounding box size (width, height) in pixels.
        """
        self.margin_ratio = margin_ratio
        self.min_size = min_size
        self.face_cascade = None

        # Check if CascadeClassifier is available in this cv2 build
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            try:
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self.face_cascade = cascade
            except Exception:
                self.face_cascade = None

    def detect_faces(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect all human faces in a BGR image.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h_img, w_img = image_bgr.shape[:2]
        detected_faces = []

        if self.face_cascade is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=self.min_size,
                flags=getattr(cv2, "CASCADE_SCALE_IMAGE", 0)
            )

            for (x, y, w, h) in faces:
                pad_w = int(w * self.margin_ratio)
                pad_h = int(h * self.margin_ratio)

                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(w_img, x + w + pad_w)
                y2 = min(h_img, y + h + pad_h)

                face_crop_bgr = image_bgr[y1:y2, x1:x2]
                if face_crop_bgr.size == 0 or face_crop_bgr.shape[0] < 20 or face_crop_bgr.shape[1] < 20:
                    continue

                face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                detected_faces.append({
                    "box": (int(x), int(y), int(w), int(h)),
                    "expanded_box": (int(x1), int(y1), int(x2), int(y2)),
                    "face_bgr": face_crop_bgr,
                    "face_rgb": face_crop_rgb,
                    "confidence": 0.95
                })

        # Only use heuristic fallback if cascade classifier completely failed to initialize in OpenCV
        # and image contains meaningful contrast (not a flat blank/black/white image)
        if self.face_cascade is None and (w_img >= 100 and h_img >= 100):
            std_dev = np.std(image_bgr)
            mean_val = np.mean(image_bgr)
            # Require texture contrast typical of a photographic portrait (not flat solid color)
            if std_dev > 18.0 and 25.0 < mean_val < 235.0:
                cx, cy = w_img // 2, h_img // 2
                half_w, half_h = int(w_img * 0.38), int(h_img * 0.42)
                x1, y1 = max(0, cx - half_w), max(0, cy - half_h)
                x2, y2 = min(w_img, cx + half_w), min(h_img, cy + half_h)
                
                face_crop_bgr = image_bgr[y1:y2, x1:x2]
                if face_crop_bgr.size > 0:
                    face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                    detected_faces.append({
                        "box": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                        "expanded_box": (int(x1), int(y1), int(x2), int(y2)),
                        "face_bgr": face_crop_bgr,
                        "face_rgb": face_crop_rgb,
                        "confidence": 0.65
                    })

        return detected_faces

    def extract_primary_face(self, image_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Extract the largest detected face by area (primary subject of interest).

        Args:
            image_bgr: Input BGR image.

        Returns:
            Dictionary with face crop and coordinates, or None if no face detected.
        """
        faces = self.detect_faces(image_bgr)
        if not faces:
            return None

        # Return face with the largest area
        faces.sort(key=lambda item: item["box"][2] * item["box"][3], reverse=True)
        return faces[0]

    def draw_detections(self, image_bgr: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
        """
        Annotate image with bounding boxes for visual verification.
        """
        annotated = image_bgr.copy()
        for idx, item in enumerate(faces):
            x1, y1, x2, y2 = item["expanded_box"]
            # Electric cyan box for forensic theme
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 210, 0), 2)
            cv2.putText(
                annotated,
                f"Face #{idx+1}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 210, 0),
                2,
                cv2.LINE_AA
            )
        return annotated
