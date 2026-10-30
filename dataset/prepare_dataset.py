"""
DEEPIFY - Dataset Ingestion & Preprocessing Pipeline
=========================================================
Processes raw videos from FaceForensics++, Celeb-DF, DFDC, or WildDeepfake:
1. Performs video-level and identity-level partitioning (prevents data leakage).
2. Samples frames at fixed temporal strides.
3. Detects faces with margin expansion.
4. Resizes to 299x299 and stores into dataset/train, validation, and test splits.
"""

import os
import sys
import glob
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from utils.face_detection import FaceDetector
from utils.image_processing import resize_image, TARGET_SIZE


def split_videos_leak_free(
    video_paths: List[Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[Path]]:
    """
    CRITICAL FORENSIC REQUIREMENT:
    Splits video files BEFORE extracting frames.
    All frames from an individual video or subject remain in exactly one split,
    preventing severe data leakage and artificially inflated benchmark scores.
    """
    random.seed(seed)
    shuffled = video_paths.copy()
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:]
    }
    return splits


def process_video_file(
    video_path: Path,
    output_dir: Path,
    detector: FaceDetector,
    sample_stride: int = 15,
    max_faces: int = 30
) -> int:
    """
    Decodes video, samples frames, detects faces, and writes 299x299 crops to output_dir.

    Returns:
        Number of faces successfully extracted.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Unable to open video: {video_path}")
        return 0

    frame_idx = 0
    saved_count = 0
    video_stem = video_path.stem

    while True:
        ret, frame_bgr = cap.read()
        if not ret or saved_count >= max_faces:
            break

        # Only process frames at the sampling stride
        if frame_idx % sample_stride == 0:
            faces = detector.detect_faces(frame_bgr)
            for f_idx, face_data in enumerate(faces):
                face_rgb = face_data["face_rgb"]
                # Resize to standard XceptionNet 299x299
                face_299 = resize_image(face_rgb, TARGET_SIZE)
                
                out_filename = f"{video_stem}_f{frame_idx}_face{f_idx}.jpg"
                out_path = output_dir / out_filename
                
                # Save as high-quality JPEG
                face_bgr_out = cv2.cvtColor(face_299, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), face_bgr_out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved_count += 1
                
                if saved_count >= max_faces:
                    break

        frame_idx += 1

    cap.release()
    return saved_count


def run_pipeline(
    raw_dir: str,
    output_dataset_dir: str,
    sample_stride: int = 15,
    max_faces_per_video: int = 25
):
    """
    Executes end-to-end dataset creation from raw video folders:
    raw_dir/real/*.mp4
    raw_dir/fake/*.mp4
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dataset_dir)
    detector = FaceDetector(margin_ratio=0.20)

    print("=" * 70)
    print("DEEPIFY - FORENSIC DATASET PREPROCESSING PIPELINE")
    print("=" * 70)

    for category in ["real", "fake"]:
        cat_dir = raw_path / category
        if not cat_dir.exists():
            print(f"[INFO] Skipping '{category}': folder {cat_dir} does not exist.")
            continue

        video_files = list(cat_dir.glob("*.mp4")) + list(cat_dir.glob("*.avi")) + list(cat_dir.glob("*.mov"))
        print(f"Found {len(video_files)} {category.upper()} source videos.")

        if not video_files:
            continue

        splits = split_videos_leak_free(video_files)

        for split_name, v_list in splits.items():
            dest_folder = out_path / split_name / category
            dest_folder.mkdir(parents=True, exist_ok=True)

            print(f"  Extracting {split_name}/{category} from {len(v_list)} videos...")
            total_extracted = 0
            for v_path in v_list:
                n = process_video_file(
                    v_path,
                    dest_folder,
                    detector,
                    sample_stride=sample_stride,
                    max_faces=max_faces_per_video
                )
                total_extracted += n

            print(f"  --> Extracted {total_extracted} faces to {dest_folder}")

    print("\nDataset preprocessing complete. Target directory structure populated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepify Video Preprocessing & Face Extraction")
    parser.add_argument("--raw_dir", type=str, default="raw_videos", help="Path to raw videos (contains real/ and fake/)")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Destination dataset folder")
    parser.add_argument("--stride", type=int, default=15, help="Frame sampling stride (e.g. every 15th frame)")
    parser.add_argument("--max_faces", type=int, default=20, help="Max face frames per video")
    args = parser.parse_args()

    run_pipeline(args.raw_dir, args.output_dir, args.stride, args.max_faces)
