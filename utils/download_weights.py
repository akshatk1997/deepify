"""
DEEPGUARD AI - Automated Model Weights Downloader
=================================================
Automates secure downloading of the pre-trained XceptionNet deepfake model
(xception_deepfake.keras) for cloud deployments (Streamlit Community Cloud,
Docker, Hugging Face Spaces) where large binary model files are not committed
directly to Git version control.
"""

import os
import sys
from pathlib import Path
import urllib.request

BASE_DIR = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
MODEL_PATH = SAVED_MODELS_DIR / "xception_deepfake.keras"

# Default Model Hosting URL (e.g. GitHub Releases, HuggingFace, or S3 bucket)
# Replace with your actual direct model release URL
DEFAULT_MODEL_URL = os.environ.get(
    "DEEPGUARD_MODEL_URL",
    "https://github.com/your-username/deepguard-ai/releases/download/v1.0.0/xception_deepfake.keras"
)


def download_model_if_missing(model_url: str = DEFAULT_MODEL_URL) -> bool:
    """
    Downloads model weights if not already present locally.
    """
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        print(f"[INFO] Model binary already exists at: {MODEL_PATH}")
        return True

    print(f"[INFO] Model binary missing. Initiating download from: {model_url}")
    try:
        def _progress(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\rDownloading XceptionNet weights: {percent}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(model_url, str(MODEL_PATH), reporthook=_progress)
        print(f"\n[SUCCESS] Model successfully saved to: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"\n[WARN] Automated download failed: {str(e)}")
        print("Please train via Google Colab (notebooks/deepfake_training.ipynb) or place xception_deepfake.keras in saved_models/.")
        return False


if __name__ == "__main__":
    download_model_if_missing()
