"""
DEEPGUARD AI - Forensic Video Processing & Temporal Aggregation Engine
=======================================================================
Processes video files (MP4, AVI, MOV):
1. Decodes video streams with OpenCV.
2. Extracts uniformly distributed or strided frames.
3. Detects faces across temporal frames.
4. Executes model inference on detected faces.
5. Implements multi-frame temporal probability aggregation.
6. Isolates suspicious frames and builds temporal verification timelines.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.face_detection import FaceDetector
from models.predict import DeepfakePredictor


class VideoProcessor:
    """
    Forensic video analysis engine supporting MP4, AVI, MOV with temporal aggregation.
    """

    def __init__(self, predictor: Optional[DeepfakePredictor] = None):
        self.detector = FaceDetector(margin_ratio=0.20)
        self.predictor = predictor or DeepfakePredictor()

    @staticmethod
    def get_video_metadata(video_path: str) -> Dict[str, Any]:
        """Extracts technical metadata from video container."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"valid": False, "error": "Failed to open video container."}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        duration_sec = round(frame_count / fps, 2) if fps > 0 and frame_count > 0 else 0.0

        cap.release()
        return {
            "valid": True,
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration_sec,
            "resolution": f"{width}x{height}"
        }

    def analyze_video_file(
        self,
        video_path: str,
        target_sample_count: int = 24,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-frame forensic verification on video.

        Aggregation Methodology:
        - A video is classified based on facial probabilities across all sampled frames.
        - Deepfake face-swaps are often localized to specific frame intervals.
        - A video is classified as FAKE if:
            1. The ratio of manipulated frames >= 20%, OR
            2. The average fake probability of the top 20% most suspicious frames >= 0.75.
        - If no faces are detected across the sampled frames, result is INCONCLUSIVE.
        """
        start_time = time.time()
        meta = self.get_video_metadata(video_path)

        if not meta["valid"] or meta["frame_count"] == 0:
            return {
                "status": "ERROR",
                "message": "Invalid or unreadable video file.",
                "duration_sec": 0,
                "frames_analyzed": 0
            }

        total_frames = meta["frame_count"]
        fps = meta["fps"]

        # Calculate uniform sampling frame indices
        sample_count = min(target_sample_count, total_frames)
        if sample_count <= 0:
            sample_count = 1
        frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        cap = cv2.VideoCapture(video_path)
        frame_timeline: List[Dict[str, Any]] = []
        suspicious_frames: List[Dict[str, Any]] = []

        total_faces_found = 0
        current_sample_idx = 0

        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue

            timestamp_sec = round(f_idx / fps, 2) if fps > 0 else 0.0
            faces = self.detector.detect_faces(frame_bgr)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_entry: Dict[str, Any] = {
                "frame_number": int(f_idx),
                "timestamp_sec": timestamp_sec,
                "has_face": len(faces) > 0,
                "fake_probability": 0.0,
                "real_probability": 1.0,
                "verdict": "NO_FACE",
                "face_crop_rgb": None
            }

            if faces and self.predictor.model_loaded:
                primary_face = faces[0]
                face_crop_rgb = primary_face["face_rgb"]
                total_faces_found += 1

                # Execute XceptionNet inference
                pred = self.predictor.predict_face_tensor(face_crop_rgb)

                frame_entry["fake_probability"] = pred["fake_probability"]
                frame_entry["real_probability"] = pred["real_probability"]
                frame_entry["verdict"] = pred["label"]
                frame_entry["confidence"] = pred["confidence"]
                frame_entry["face_crop_rgb"] = face_crop_rgb

                # Collect suspicious frames (fake prob >= 0.50)
                if pred["is_fake"]:
                    suspicious_frames.append({
                        "frame_number": int(f_idx),
                        "timestamp_sec": timestamp_sec,
                        "fake_probability": pred["fake_probability"],
                        "confidence": pred["confidence"],
                        "face_crop_rgb": face_crop_rgb,
                        "full_frame_rgb": cv2.cvtColor(
                            self.detector.draw_detections(frame_bgr, faces),
                            cv2.COLOR_BGR2RGB
                        )
                    })

            frame_timeline.append(frame_entry)
            current_sample_idx += 1

            if progress_callback:
                progress = min(1.0, current_sample_idx / len(frame_indices))
                progress_callback(progress, f"Analyzing frame {current_sample_idx}/{len(frame_indices)} at {timestamp_sec}s...")

        cap.release()
        total_time_sec = round(time.time() - start_time, 2)

        # Handle Edge Case: No usable faces detected
        if total_faces_found == 0:
            return {
                "status": "INCONCLUSIVE",
                "verdict": "INCONCLUSIVE",
                "message": (
                    "No usable human facial subjects were detected across the sampled video frames. "
                    "Media authenticity cannot be verified without clear facial subjects."
                ),
                "metadata": meta,
                "frames_analyzed": len(frame_timeline),
                "faces_detected": 0,
                "processing_time_sec": total_time_sec,
                "timeline": frame_timeline,
                "suspicious_frames": []
            }

        # Temporal Aggregation Mathematics
        face_frames = [f for f in frame_timeline if f["has_face"]]
        fake_probs = [f["fake_probability"] for f in face_frames]
        real_probs = [f["real_probability"] for f in face_frames]

        mean_fake_prob = float(np.mean(fake_probs))
        mean_real_prob = float(np.mean(real_probs))

        # Top-K 25% highest risk frames average
        k_top = max(1, int(len(fake_probs) * 0.25))
        top_k_fake_prob = float(np.mean(sorted(fake_probs, reverse=True)[:k_top]))

        # Manipulated frame ratio
        manipulated_count = sum(1 for p in fake_probs if p >= 0.50)
        manipulated_ratio = manipulated_count / len(face_frames)

        # Video-Level Forensic Classification Rule:
        # A video is a deepfake if >= 20% frames are manipulated or top-K burst >= 0.75
        is_video_fake = (manipulated_ratio >= 0.20) or (top_k_fake_prob >= 0.75)

        if is_video_fake:
            overall_verdict = "FAKE"
            overall_confidence = round(max(mean_fake_prob, top_k_fake_prob) * 100, 2)
            risk_tier = "CRITICAL" if top_k_fake_prob > 0.85 else "HIGH"
        else:
            overall_verdict = "REAL"
            overall_confidence = round(mean_real_prob * 100, 2)
            risk_tier = "LOW (AUTHENTIC)" if mean_real_prob > 0.85 else "MODERATE"

        # Sort suspicious frames from highest fake probability to lowest
        suspicious_frames.sort(key=lambda item: item["fake_probability"], reverse=True)

        return {
            "status": "SUCCESS",
            "verdict": overall_verdict,
            "confidence": overall_confidence,
            "mean_fake_probability": round(mean_fake_prob, 4),
            "mean_real_probability": round(mean_real_prob, 4),
            "top_k_fake_probability": round(top_k_fake_prob, 4),
            "manipulated_frames_count": manipulated_count,
            "manipulated_frames_ratio": round(manipulated_ratio * 100, 2),
            "risk_tier": risk_tier,
            "metadata": meta,
            "frames_analyzed": len(frame_timeline),
            "faces_detected": total_faces_found,
            "processing_time_sec": total_time_sec,
            "timeline": frame_timeline,
            "suspicious_frames": suspicious_frames[:6]  # Top 6 suspicious frames
        }
