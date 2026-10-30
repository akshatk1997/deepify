"""
DEEPGUARD AI - Single-Image Deepfake Forensics Page
====================================================
Interactive Streamlit interface for image upload, face localization,
XceptionNet model inference, confidence telemetry, and forensic reporting.
"""

import time
from datetime import datetime
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Ensure project root paths
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
CSS_PATH = ASSETS_DIR / "styles.css"
LOGO_PATH = ASSETS_DIR / "logo.png"

# Load styles
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from models.predict import DeepfakePredictor
from models.gradcam import GradCAMExplainer
from utils.image_processing import load_image_from_bytes
from utils.report_generator import ForensicReportGenerator


def render_image_detection_page():
    # Page Header
    st.markdown('<div class="brand-title">IMAGE FORENSIC INSPECTION</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">'
        'Upload portrait imagery to detect AI facial synthesis, deepfake face-swaps, '
        'and blending seam anomalies via the deep XceptionNet neural engine.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Initialize predictor singleton
    predictor = DeepfakePredictor()

    # Upload Container
    col_upload, col_guidance = st.columns([5, 3])

    with col_upload:
        st.markdown("### 📤 Upload Forensic Target")
        uploaded_file = st.file_uploader(
            "Select image for deepfake verification (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="High-resolution frontal portrait images yield optimal forensic accuracy."
        )

    with col_guidance:
        st.markdown("### 🔍 Forensic Protocol")
        st.markdown("""
        <div class="deepguard-card" style="padding: 16px; margin-top: 10px;">
            <p style="margin: 0; font-size: 0.9rem; color: #94a3b8;">
                <strong style="color: #00d2ff;">1. Face Isolation:</strong> OpenCV crops subject with 20% margin.<br>
                <strong style="color: #00d2ff;">2. Normalization:</strong> Scaled to 299×299 [-1, 1] range.<br>
                <strong style="color: #00d2ff;">3. Logit Prediction:</strong> Depthwise separable convolution score.<br>
                <strong style="color: #00d2ff;">4. Verification:</strong> Classified as REAL or FAKE.
            </p>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("👋 Upload a facial image above to initiate deepfake verification analysis.")
        return

    # Validate file size (max 15MB)
    if uploaded_file.size > 15 * 1024 * 1024:
        st.error("Uploaded file exceeds 15MB limit. Please upload an optimized forensic image.")
        return

    # Read bytes and decode
    try:
        file_bytes = uploaded_file.read()
        image_bgr, image_rgb = load_image_from_bytes(file_bytes)
    except Exception as e:
        st.error(f"Error reading image: {str(e)}")
        return

    st.markdown("<br>", unsafe_allow_html=True)

    # Preview uploaded media
    st.subheader("📸 Media Visual & Localization Preview")
    prev_col1, prev_col2 = st.columns([1, 1])

    with prev_col1:
        st.image(image_rgb, caption="Raw Uploaded Media", use_container_width=True)

    # Check model availability
    if not predictor.model_loaded:
        with prev_col2:
            st.markdown("""
            <div class="cyber-alert-warning">
                <h4>⚠️ Model Binary Missing (Phase 3 Required)</h4>
                <p>The trained XceptionNet weights file was not detected at <code>saved_models/xception_deepfake.keras</code>.</p>
                <p><strong>Next Steps:</strong></p>
                <ol>
                    <li>Execute the Google Colab Notebook at <code>notebooks/deepfake_training.ipynb</code> on GPU.</li>
                    <li>Download the trained <code>xception_deepfake.keras</code> file.</li>
                    <li>Place it inside the local <code>saved_models/</code> directory.</li>
                    <li>Or run local CPU training: <code>python models/train_model.py</code>.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            # Still demonstrate face detection capability
            faces = predictor.detector.detect_faces(image_bgr)
            if faces:
                st.success(f"Detected {len(faces)} human facial subject(s) via OpenCV Haar Cascade.")
                st.image(faces[0]["face_rgb"], caption="Primary Subject Face Crop (20% Expanded Margin)", width=240)
            else:
                st.warning("No human face detected in uploaded media.")
        return

    # Run actual model inference
    with st.spinner("Executing forensic neural graph inference & boundary artifact scan..."):
        analysis = predictor.predict_image(image_bgr)

    # Handle edge case: No face detected
    if analysis["status"] == "NO_FACE_DETECTED":
        st.warning(f"⚠️ {analysis['message']}")
        return

    # Handle normal successful prediction
    if analysis["status"] == "SUCCESS":
        verdict = analysis["verdict"]
        confidence = analysis["confidence"]
        fake_prob = analysis["fake_probability"]
        real_prob = analysis["real_probability"]
        risk_tier = analysis["risk_tier"]

        with prev_col2:
            st.image(analysis["annotated_image_rgb"], caption=f"Forensic Face Localization ({analysis['faces_detected']} subject(s))", use_container_width=True)

        st.markdown("---")
        st.subheader("🛡️ Forensic Verification Results")

        # Top Verdict Banner
        res_col1, res_col2 = st.columns([3, 2])

        with res_col1:
            if verdict == "FAKE":
                st.markdown("""
                <div class="deepguard-card" style="border-color: rgba(239, 68, 68, 0.4);">
                    <div class="badge-fake">🚨 FAKE / MANIPULATED MEDIA DETECTED</div>
                    <h2 style="color: #ef4444; margin-top: 14px; margin-bottom: 6px;">HIGH PROBABILITY OF AI SYNTHESIS</h2>
                    <p style="color: #cbd5e1; font-size: 0.95rem;">
                        The XceptionNet forensic engine detected characteristic facial manipulation signatures,
                        including blending boundary discontinuities, high-frequency texture distortions, or facial warping artifacts.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="deepguard-card" style="border-color: rgba(16, 185, 129, 0.4);">
                    <div class="badge-real">🛡️ REAL / AUTHENTIC MEDIA VERIFIED</div>
                    <h2 style="color: #10b981; margin-top: 14px; margin-bottom: 6px;">NATURAL BIOMETRIC INTEGRITY CONFIRMED</h2>
                    <p style="color: #cbd5e1; font-size: 0.95rem;">
                        The uploaded portrait displays organic facial skin tone continuity, authentic ocular reflections,
                        and natural boundary frequencies consistent with unmodified authentic camera captures.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="deepguard-card">
                <h4 style="color: #00d2ff; margin-top: 0;">Forensic Telemetry</h4>
                <div style="margin-bottom: 8px;">
                    <span style="color: #94a3b8;">Confidence Score:</span> 
                    <strong style="color: #ffffff; font-size: 1.2rem;">{confidence}%</strong>
                </div>
                <div style="margin-bottom: 8px;">
                    <span style="color: #94a3b8;">Risk Classification:</span> 
                    <strong style="color: {'#ef4444' if verdict == 'FAKE' else '#10b981'};">{risk_tier}</strong>
                </div>
                <div style="margin-bottom: 8px;">
                    <span style="color: #94a3b8;">Inference Latency:</span> 
                    <strong style="color: #cbd5e1;">{analysis['inference_time_ms']} ms</strong>
                </div>
                <div>
                    <span style="color: #94a3b8;">Faces Localized:</span> 
                    <strong style="color: #cbd5e1;">{analysis['faces_detected']}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence Distribution Meters
        st.markdown("#### 📊 Prediction Probability Distribution")
        p_col1, p_col2 = st.columns(2)

        with p_col1:
            st.markdown(f"**REAL (Authentic) Probability: `{round(real_prob * 100, 2)}%`**")
            st.markdown(f"""
            <div class="forensic-progress-bar">
                <div class="forensic-progress-fill-real" style="width: {min(100, max(0, real_prob * 100))}%;"></div>
            </div>
            """, unsafe_allow_html=True)

        with p_col2:
            st.markdown(f"**FAKE (Manipulated) Probability: `{round(fake_prob * 100, 2)}%`**")
            st.markdown(f"""
            <div class="forensic-progress-bar">
                <div class="forensic-progress-fill-fake" style="width: {min(100, max(0, fake_prob * 100))}%;"></div>
            </div>
            """, unsafe_allow_html=True)

        # Subject Face Crop Details
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🔬 Extracted Subject Facial Region")
        face_col1, face_col2 = st.columns([1, 3])
        with face_col1:
            st.image(analysis["primary_face_rgb"], caption="Input to XceptionNet (299×299)", width=220)
        with face_col2:
            st.markdown(f"""
            <div class="deepguard-card">
                <h4 style="color: #00d2ff; margin-top: 0;">Feature Map Analysis Details</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6;">
                    • <strong>Primary Bounding Box (Expanded):</strong> <code>{analysis['primary_face_box']}</code><br>
                    • <strong>Input Normalization:</strong> <code>[-1.0, 1.0]</code> across 3 RGB channels.<br>
                    • <strong>Convolutional Depth:</strong> 36 Depthwise Separable Convolutional stages.<br>
                    • <strong>Classification Logit:</strong> <code>{fake_prob:.6f}</code> (Threshold: 0.50).
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Explainable AI (Grad-CAM) Visual Evidence Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🧠 Explainable AI: Grad-CAM Feature Activation")
        st.caption("Visualizing gradient backpropagation through final convolutional block (block14_sepconv2_act).")

        try:
            explainer = GradCAMExplainer(predictor.model)
            explanation = explainer.explain(analysis["primary_face_rgb"], alpha=0.45)

            cam_c1, cam_c2, cam_c3 = st.columns(3)
            with cam_c1:
                st.image(analysis["primary_face_rgb"], caption="1. Subject Face Crop", use_container_width=True)
            with cam_c2:
                st.image(explanation["heatmap_rgb"], caption="2. Grad-CAM Activation Heatmap", use_container_width=True)
            with cam_c3:
                st.image(explanation["overlay_rgb"], caption="3. Forensic Superimposed Overlay", use_container_width=True)

            dominant_str = ", ".join(explanation["dominant_regions"]) if explanation["dominant_regions"] else "General facial structure"
            st.markdown(f"""
            <div class="cyber-alert-info">
                <strong>🔍 Attention Interpretation:</strong> Model attention is concentrated heavily on: 
                <strong style="color: #00d2ff;">{dominant_str}</strong>.<br>
                <small style="color: #94a3b8;">
                    Warmer colors (red/yellow) indicate higher neural weight contributing toward the classification verdict;
                    cooler tones (blue/violet) indicate baseline or suppressed feature responses.
                </small>
            </div>
            """, unsafe_allow_html=True)
        except Exception as cam_err:
            st.warning(f"Grad-CAM visualization notice: {str(cam_err)}")

        # Record in Session State History for persistence across pages
        record_id = f"IMG-{int(time.time())}"
        history_entry = {
            "id": record_id,
            "filename": uploaded_file.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "media_type": "Image",
            "verdict": verdict,
            "confidence": f"{confidence}%",
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "faces_detected": analysis["faces_detected"],
            "latency_ms": analysis["inference_time_ms"]
        }
        
        # Avoid duplicate consecutive entries
        if not any(entry.get("id") == record_id for entry in st.session_state.analysis_history):
            st.session_state.analysis_history.insert(0, history_entry)

        st.success(f"Forensic record saved to History log (ID: {record_id}).")

        # Download Official Forensic PDF Report Button
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_gen = ForensicReportGenerator()
        analysis_data_report = {
            "filename": uploaded_file.name,
            "verdict": verdict,
            "confidence": confidence,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "risk_tier": risk_tier,
            "faces_detected": analysis["faces_detected"],
            "primary_face_box": analysis["primary_face_box"],
            "primary_face_rgb": analysis["primary_face_rgb"],
            "inference_time_ms": analysis["inference_time_ms"]
        }
        
        explanation_report = None
        try:
            if 'explanation' in locals():
                explanation_report = explanation
        except Exception:
            pass

        pdf_path = pdf_gen.generate_image_report(analysis_data_report, explanation_report, f"{record_id}.pdf")
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        st.download_button(
            label="📄 Download Official Forensic Audit Report (PDF)",
            data=pdf_bytes,
            file_name=f"DeepGuard_Forensic_Report_{record_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )


if __name__ == "__main__":
    render_image_detection_page()
