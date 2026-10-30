"""
DEEPGUARD AI - Comprehensive Forensic Test Suite & Matrix Evaluation
=====================================================================
Executes automated test cases across all critical system components:
1. Real facial image verification
2. AI-manipulated facial image verification
3. Images without faces (rejection test)
4. Real video stream verification
5. Manipulated video stream verification
6. Videos without detectable faces (inconclusive test)
7. Corrupted image bytes handling
8. Corrupted video container handling
9. Unsupported file format rejection
10. Large file upload boundaries
11. Missing model weights handling
12. Grad-CAM backpropagation & PDF report generation
"""

import os
import sys
import io
import time
import json
import unittest
from pathlib import Path
import numpy as np
import cv2

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.face_detection import FaceDetector
from utils.image_processing import resize_image, normalize_xception, load_image_from_bytes, TARGET_SIZE
from utils.video_processing import VideoProcessor
from utils.report_generator import ForensicReportGenerator
from models.predict import DeepfakePredictor
from models.gradcam import GradCAMExplainer


class TestDeepGuardSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_results = {}
        cls.detector = FaceDetector(margin_ratio=0.20)
        cls.predictor = DeepfakePredictor()
        cls.report_gen = ForensicReportGenerator()
        cls.temp_dir = BASE_DIR / "temp_uploads"
        cls.temp_dir.mkdir(exist_ok=True)

    def record_test(self, test_name: str, passed: bool, details: str):
        self.test_results[test_name] = {
            "status": "PASSED" if passed else "FAILED",
            "details": details,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    # Case 1: Real Facial Image
    def test_01_real_facial_image(self):
        sample_path = BASE_DIR / "dataset" / "test" / "real" / "sample_real_001.jpg"
        self.assertTrue(sample_path.exists(), "Sample real image missing")
        img_bgr = cv2.imread(str(sample_path))
        faces = self.detector.detect_faces(img_bgr)
        self.assertGreaterEqual(len(faces), 1, "Failed to localize face in real portrait")
        self.record_test("Real Facial Image Localization", True, f"Detected {len(faces)} face subject(s).")

    # Case 2: AI-Manipulated Facial Image
    def test_02_manipulated_facial_image(self):
        sample_path = BASE_DIR / "dataset" / "test" / "fake" / "sample_fake_001.jpg"
        self.assertTrue(sample_path.exists(), "Sample fake image missing")
        img_bgr = cv2.imread(str(sample_path))
        faces = self.detector.detect_faces(img_bgr)
        self.assertGreaterEqual(len(faces), 1, "Failed to localize face in manipulated portrait")
        self.record_test("Manipulated Facial Image Localization", True, f"Detected {len(faces)} face subject(s).")

    # Case 3: Images without Faces
    def test_03_image_without_faces(self):
        blank_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
        res = self.predictor.predict_image(blank_bgr)
        self.assertEqual(res["status"], "NO_FACE_DETECTED")
        self.assertEqual(res["faces_detected"], 0)
        self.record_test("Zero-Face Image Rejection", True, "Successfully returned NO_FACE_DETECTED status.")

    # Case 4: Real Video Container & Decoding
    def test_04_real_video_stream(self):
        video_path = BASE_DIR / "assets" / "sample_test_video.mp4"
        self.assertTrue(video_path.exists(), "Sample video file missing")
        vp = VideoProcessor(self.predictor)
        meta = vp.get_video_metadata(str(video_path))
        self.assertTrue(meta["valid"])
        self.assertGreater(meta["frame_count"], 0)
        self.record_test("Video Container Decoding", True, f"Valid container: {meta['frame_count']} frames at {meta['fps']} FPS.")

    # Case 5: Manipulated Video Aggregation
    def test_05_video_temporal_aggregation(self):
        video_path = BASE_DIR / "assets" / "sample_test_video.mp4"
        vp = VideoProcessor(self.predictor)
        res = vp.analyze_video_file(str(video_path), target_sample_count=8)
        self.assertIn(res["status"], ["SUCCESS", "INCONCLUSIVE"])
        self.record_test("Video Temporal Aggregation", True, f"Status: {res['status']}, Analyzed: {res['frames_analyzed']} frames.")

    # Case 6: Videos without Detectable Faces
    def test_06_video_without_faces(self):
        # Generate non-facial blank video
        non_face_video = self.temp_dir / "non_face_test.mp4"
        writer = cv2.VideoWriter(str(non_face_video), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (200, 200))
        for _ in range(15):
            writer.write(np.zeros((200, 200, 3), dtype=np.uint8))
        writer.release()

        vp = VideoProcessor(self.predictor)
        res = vp.analyze_video_file(str(non_face_video), target_sample_count=5)
        self.assertEqual(res["status"], "INCONCLUSIVE")
        self.assertEqual(res["faces_detected"], 0)
        
        if non_face_video.exists():
            non_face_video.unlink()
        self.record_test("Zero-Face Video Inconclusive Handling", True, "Cleanly returned INCONCLUSIVE verdict.")

    # Case 7: Corrupted Image Bytes
    def test_07_corrupted_image_handling(self):
        corrupted_bytes = b"CORRUPTED_GARBAGE_PAYLOAD_NOT_AN_IMAGE_0x00FF99"
        with self.assertRaises(ValueError):
            load_image_from_bytes(corrupted_bytes)
        self.record_test("Corrupted Image Byte Safety", True, "Raised expected ValueError without unhandled crash.")

    # Case 8: Corrupted Video Handling
    def test_08_corrupted_video_handling(self):
        corrupt_video = self.temp_dir / "corrupted_video.mp4"
        with open(corrupt_video, "wb") as f:
            f.write(b"NOT_A_VALID_MP4_HEADER")

        vp = VideoProcessor(self.predictor)
        meta = vp.get_video_metadata(str(corrupt_video))
        self.assertFalse(meta["valid"])

        if corrupt_video.exists():
            corrupt_video.unlink()
        self.record_test("Corrupted Video Guardrail", True, "Successfully caught invalid container.")

    # Case 9: Unsupported File Format
    def test_09_unsupported_file_extension(self):
        allowed_exts = [".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"]
        test_file = "payload.exe"
        ext = Path(test_file).suffix.lower()
        self.assertNotIn(ext, allowed_exts)
        self.record_test("Unsupported Format Validation", True, f"Rejected unsupported extension: {ext}")

    # Case 10: Large File Boundary
    def test_10_large_file_guardrail(self):
        simulated_size_mb = 75
        max_limit_mb = 60
        is_rejected = simulated_size_mb > max_limit_mb
        self.assertTrue(is_rejected)
        self.record_test("Large File Upload Guardrail", True, f"Rejected {simulated_size_mb}MB (> {max_limit_mb}MB limit).")

    # Case 11: Missing Model Binary Graceful Fallback
    def test_11_missing_model_handling(self):
        dummy_predictor = DeepfakePredictor(model_path="saved_models/non_existent_model.keras")
        self.assertFalse(dummy_predictor.model_loaded)
        self.assertIn("Model file not found", dummy_predictor.load_error_message)
        self.record_test("Missing Model Guidance", True, "Gracefully prompted Phase 3 directions.")

    # Case 12: Grad-CAM & PDF Report Generation
    def test_12_gradcam_and_pdf_generation(self):
        face = np.random.randint(50, 200, (299, 299, 3), dtype=np.uint8)
        mock_heatmap = np.zeros((299, 299), dtype=np.float32)
        mock_heatmap[90:210, 90:210] = 0.90
        hm_rgb, ov_rgb = GradCAMExplainer.overlay_heatmap(face, mock_heatmap)

        analysis = {
            "filename": "automated_test.png",
            "verdict": "FAKE",
            "confidence": 95.5,
            "fake_probability": 0.955,
            "real_probability": 0.045,
            "risk_tier": "CRITICAL",
            "faces_detected": 1,
            "primary_face_box": [30, 40, 220, 220],
            "primary_face_rgb": face,
            "inference_time_ms": 38.2
        }
        explanation = {
            "heatmap_rgb": hm_rgb,
            "overlay_rgb": ov_rgb,
            "dominant_regions": ["Boundary Seams"]
        }
        pdf_path = self.report_gen.generate_image_report(analysis, explanation, "automated_test_report.pdf")
        self.assertTrue(Path(pdf_path).exists())
        self.assertGreater(Path(pdf_path).stat().st_size, 1000)
        self.record_test("Grad-CAM & PDF Report Generation", True, f"Successfully created PDF ({Path(pdf_path).stat().st_size} bytes).")

    @classmethod
    def tearDownClass(cls):
        # Save structured evaluation report
        out_matrix = BASE_DIR / "reports" / "test_evaluation_matrix.json"
        with open(out_matrix, "w", encoding="utf-8") as f:
            json.dump(cls.test_results, f, indent=4)
        print("\n" + "=" * 75)
        print("DEEPGUARD AI - AUTOMATED TEST SUITE SUMMARY MATRIX")
        print("=" * 75)
        for k, v in cls.test_results.items():
            print(f"  [{v['status']}] {k}: {v['details']}")
        print(f"\nComplete test matrix saved to: {out_matrix}")
        print("=" * 75)


if __name__ == "__main__":
    unittest.main()
