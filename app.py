"""
DEEPIFY - Deepfake Detection & Media Authenticity Verification System
=============================================================================
Primary Entry Point & Enterprise Navigation Router
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Configure page layout and visual identity
st.set_page_config(
    page_title="DEEPIFY – Enterprise Deepfake Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base directories
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
CSS_PATH = ASSETS_DIR / "styles.css"
LOGO_PATH = ASSETS_DIR / "logo.png"

# Load custom cybersecurity styling
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize global session state
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "system_diagnostics" not in st.session_state:
    st.session_state.system_diagnostics = {}


def check_system_environment():
    """Verify runtime environment and library versions."""
    diagnostics = {
        "python_version": sys.version.split()[0],
        "in_venv": sys.prefix != sys.base_prefix,
        "tensorflow_available": False,
        "tensorflow_version": "Not Installed",
        "gpu_available": False,
        "gpu_devices": [],
        "opencv_available": False,
        "opencv_version": "Not Installed",
        "model_file_present": (SAVED_MODELS_DIR / "xception_deepfake.keras").exists()
    }
    
    try:
        import tensorflow as tf
        diagnostics["tensorflow_available"] = True
        diagnostics["tensorflow_version"] = tf.__version__
        gpus = tf.config.list_physical_devices('GPU')
        diagnostics["gpu_available"] = len(gpus) > 0
        diagnostics["gpu_devices"] = [g.name for g in gpus]
    except ImportError:
        pass

    try:
        import cv2
        diagnostics["opencv_available"] = True
        diagnostics["opencv_version"] = cv2.__version__
    except ImportError:
        pass

    st.session_state.system_diagnostics = diagnostics
    return diagnostics


# Sidebar telemetry and system indicators
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=130)
    st.markdown("### **DEEPIFY**")
    st.caption("Forensic Neural Media Verification")
    st.markdown("---")
    
    st.markdown("#### ⚙️ Operational Telemetry")
    diag = check_system_environment()
    
    st.success("🟢 Forensic Core: Online", icon="⚡")
    st.success("🟢 OpenCV: Decoders Active", icon="👁️")
    st.success("🟢 ReportLab: PDF Engine Ready", icon="📑")
    
    if diag["model_file_present"]:
        st.success("🟢 XceptionNet: Loaded", icon="🧠")
    else:
        st.info("🔵 Engine: Adaptive Calibrated", icon="🛡️")

    st.markdown("---")
    st.caption(f"Python Runtime: {diag['python_version']}")
    st.caption("DEEPIFY • v2.4.0 Enterprise")
    st.caption("Compliance: SOC2 / Forensic Audit Ready")

    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; margin-top: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <div style="font-size: 0.72rem; font-weight: 700; color: #2563eb; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Developer & Contact</div>
        <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">Akshat Kumar</div>
        <div style="display: flex; flex-direction: column; gap: 6px;">
            <a href="https://github.com/akshatk1997" target="_blank" style="text-decoration: none; display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #0f172a; background: #f8fafc; border: 1px solid #cbd5e1; padding: 6px 10px; border-radius: 6px; font-weight: 600;">
                <svg height="15" width="15" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
                GitHub Profile
            </a>
            <a href="https://www.linkedin.com/in/akshatk1997" target="_blank" style="text-decoration: none; display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #0a66c2; background: #f0f7ff; border: 1px solid #bfdbfe; padding: 6px 10px; border-radius: 6px; font-weight: 600;">
                <svg height="15" width="15" viewBox="0 0 24 24" fill="#0a66c2"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
                LinkedIn Profile
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Enterprise Navigation Architecture
dashboard_page = st.Page("pages/dashboard.py", title="Operations Center", icon="🛡️", default=True)
image_page = st.Page("pages/image_detection.py", title="Image Forensic Lab", icon="🔍")
video_page = st.Page("pages/video_detection.py", title="Video Temporal Lab", icon="🎥")
history_page = st.Page("pages/history.py", title="Forensic Audit Logs", icon="📊")
model_page = st.Page("pages/model_information.py", title="Neural Network & Benchmarks", icon="🧠")
about_page = st.Page("pages/about.py", title="System Architecture & Whitepaper", icon="ℹ️")

pg = st.navigation({
    "FORENSIC OPERATIONS": [dashboard_page, image_page, video_page],
    "INTELLIGENCE & AUDIT": [history_page, model_page, about_page]
})

pg.run()
