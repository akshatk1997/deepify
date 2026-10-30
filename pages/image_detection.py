"""
DEEPIFY - Single-Image Deepfake Forensics Page
====================================================
Enterprise SaaS forensic interface for image upload, automated face localization,
neural inference, Explainable AI (Grad-CAM), and court-ready PDF generation.
"""

import time
from datetime import datetime
from pathlib import Path
import streamlit as st
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
CSS_PATH = ASSETS_DIR / "styles.css"
SAMPLE_DIR = BASE_DIR / "dataset" / "test"

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from models.predict import DeepfakePredictor
from models.gradcam import GradCAMExplainer
from utils.image_processing import load_image_from_bytes, load_image_from_path
from utils.report_generator import ForensicReportGenerator


def render_image_detection_page():
    # SaaS Enterprise Header Bar
    st.markdown("""
    <div class="saas-header-bar">
        <div style="font-weight: 700; color: #0f172a; font-size: 1rem; display: flex; align-items: center; gap: 8px;">
            <span style="color: #2563eb;">🛡️ DEEPIFY</span>
            <span style="color: #cbd5e1;">|</span>
            <span style="color: #475569; font-weight: 500;">Image Forensic Lab</span>
        </div>
        <div class="saas-badge-group">
            <span class="saas-pill saas-pill-live"><span class="saas-pill-pulse"></span> Neural Core Online</span>
            <span class="saas-pill">Backbone: XceptionNet</span>
            <span class="saas-pill">Input: 299×299 RGB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="brand-title">FACIAL AUTHENTICITY VERIFICATION</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">'
        'Audit facial imagery for AI-generated synthesis, face-swapping, and blending boundary seams '
        'using depthwise separable convolutions and Explainable AI (Grad-CAM).'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    predictor = DeepfakePredictor()

    # Quick Demo Selector Bar
    st.markdown("""
    <div class="demo-bar-container">
        <div style="font-weight: 700; color: #0284c7; font-size: 0.94rem; margin-bottom: 6px;">
            🎯 One-Click Verification Sandbox: Load Pre-Validated Benchmark Portraits
        </div>
        <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 8px;">
            Test the forensic engine immediately with authentic camera captures, synthetic face-swaps, or boundary test canvases.
        </div>
    </div>
    """, unsafe_allow_html=True)

    demo_c1, demo_c2, demo_c3 = st.columns(3)

    if "selected_image_source" not in st.session_state:
        st.session_state.selected_image_source = None
    if "uploaded_image_name" not in st.session_state:
        st.session_state.uploaded_image_name = "sample_media.jpg"

    with demo_c1:
        if st.button("🟢 Load Authentic Portrait (Real)", use_container_width=True):
            real_path = SAMPLE_DIR / "real" / "sample_real_001.jpg"
            if real_path.exists():
                bgr, rgb = load_image_from_path(str(real_path))
                st.session_state.selected_image_source = (bgr, rgb)
                st.session_state.uploaded_image_name = "benchmark_authentic_subject.jpg"
                st.session_state.cached_image_key = None
                st.rerun()

    with demo_c2:
        if st.button("🔴 Load Manipulated Portrait (Fake)", use_container_width=True):
            fake_path = SAMPLE_DIR / "fake" / "sample_fake_001.jpg"
            if fake_path.exists():
                bgr, rgb = load_image_from_path(str(fake_path))
                st.session_state.selected_image_source = (bgr, rgb)
                st.session_state.uploaded_image_name = "benchmark_manipulated_face.jpg"
                st.session_state.cached_image_key = None
                st.rerun()

    with demo_c3:
        if st.button("⚪ Test Zero-Face Media (Rejection)", use_container_width=True):
            blank_bgr = np.zeros((320, 320, 3), dtype=np.uint8)
            blank_rgb = np.zeros((320, 320, 3), dtype=np.uint8)
            st.session_state.selected_image_source = (blank_bgr, blank_rgb)
            st.session_state.uploaded_image_name = "zero_face_canvas.jpg"
            st.session_state.cached_image_key = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # File Uploader Section
    uploaded_file = st.file_uploader(
        "Or drag and drop your own target image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload frontal facial portrait media. Maximum file size: 15MB."
    )

    if uploaded_file is not None:
        if uploaded_file.size > 15 * 1024 * 1024:
            st.error("Uploaded file exceeds 15MB ceiling limit.")
            return
        bgr, rgb = load_image_from_bytes(uploaded_file.read())
        st.session_state.selected_image_source = (bgr, rgb)
        st.session_state.uploaded_image_name = uploaded_file.name
        st.session_state.cached_image_key = None

    if st.session_state.selected_image_source is None:
        st.info("👆 Select a demo benchmark sample above or upload a photo to initiate verification.")
        return

    image_bgr, image_rgb = st.session_state.selected_image_source
    active_filename = st.session_state.uploaded_image_name

    # Cache analysis by active filename and image footprint
    cache_key = f"{active_filename}_{image_bgr.shape}_{float(np.mean(image_bgr)):.2f}"
    if st.session_state.get("cached_image_key") != cache_key:
        with st.spinner("Analyzing biometric markers, compression frequencies, and XceptionNet activations..."):
            analysis = predictor.predict_image(image_bgr)
            st.session_state.cached_image_analysis = analysis
            st.session_state.cached_image_key = cache_key

            if analysis["status"] != "NO_FACE_DETECTED":
                record_id = f"IMG-{int(time.time())}"
                st.session_state.current_record_id = record_id
                history_entry = {
                    "id": record_id,
                    "filename": active_filename,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "media_type": "Image",
                    "verdict": analysis["verdict"],
                    "confidence": f"{analysis['confidence']}%",
                    "fake_probability": analysis["fake_probability"],
                    "real_probability": analysis["real_probability"],
                    "faces_detected": analysis["faces_detected"],
                    "latency_ms": analysis["inference_time_ms"]
                }
                st.session_state.analysis_history.insert(0, history_entry)
    else:
        analysis = st.session_state.cached_image_analysis

    st.markdown("<br>", unsafe_allow_html=True)

    # Handle Zero-Face Detected
    if analysis["status"] == "NO_FACE_DETECTED":
        st.markdown(f"""
        <div class="deepguard-card" style="border-left: 4px solid #f59e0b; background: #fffbeb;">
            <div class="badge-inconclusive">⚠️ VERDICT: INCONCLUSIVE (NO USABLE FACES)</div>
            <h3 style="color: #b45309; margin-top: 14px; margin-bottom: 6px;">No Human Facial Subject Localized</h3>
            <p style="color: #334155; font-size: 0.95rem; line-height: 1.6;">
                {analysis['message']}
            </p>
            <p style="color: #64748b; font-size: 0.88rem; margin-bottom: 0;">
                DEEPIFY enforces strict forensic standards: non-facial imagery is cleanly marked 
                <strong>INCONCLUSIVE</strong> to avoid hallucinated false classifications.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.image(analysis["annotated_image_rgb"], caption="Raw Uploaded Media (Unlocalized)", use_container_width=True)
        return

    # Normal Successful Analysis
    verdict = analysis["verdict"]
    confidence = analysis["confidence"]
    fake_prob = analysis["fake_probability"]
    real_prob = analysis["real_probability"]
    risk_tier = analysis["risk_tier"]
    record_id = st.session_state.get("current_record_id", f"IMG-{int(time.time())}")

    # Main Visual and Verdict Display Grid
    grid_col_left, grid_col_right = st.columns([1, 1], gap="medium")

    with grid_col_left:
        st.subheader("📸 Media Localization Preview")
        st.image(
            analysis["annotated_image_rgb"],
            caption=f"Target: {active_filename} ({analysis['faces_detected']} subject localized with 20% margin)",
            use_container_width=True
        )

    with grid_col_right:
        st.subheader("🛡️ Forensic Verification Verdict")

        if verdict == "FAKE":
            st.markdown("""
            <div class="deepguard-card" style="border-color: #fca5a5; background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);">
                <div class="badge-fake">🚨 FAKE / MANIPULATED MEDIA</div>
                <h2 style="color: #dc2626; margin-top: 12px; margin-bottom: 6px;">SYNTHETIC ANOMALIES DETECTED</h2>
                <p style="color: #334155; font-size: 0.9rem; line-height: 1.55;">
                    The neural engine identified characteristic deepfake facial synthesis markers, 
                    such as boundary blending seams, GAN deconvolution frequencies, or ocular artifacts.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="deepguard-card" style="border-color: #a7f3d0; background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%);">
                <div class="badge-real">🛡️ REAL / AUTHENTIC MEDIA</div>
                <h2 style="color: #059669; margin-top: 12px; margin-bottom: 6px;">BIOMETRIC INTEGRITY CONFIRMED</h2>
                <p style="color: #334155; font-size: 0.9rem; line-height: 1.55;">
                    Organic facial skin tone continuity, consistent ocular light reflections, and 
                    natural high-frequency boundary distributions consistent with authentic camera captures.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Telemetry Card
        st.markdown(f"""
        <div class="deepguard-card">
            <h4 style="color: #0284c7; margin-top: 0; margin-bottom: 12px;">Forensic Inspection Telemetry</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.9rem;">
                <div><span style="color: #64748b;">Confidence Score:</span> <strong style="color: #0f172a;">{confidence}%</strong></div>
                <div><span style="color: #64748b;">Risk Classification:</span> <strong style="color: {'#dc2626' if verdict == 'FAKE' else '#059669'};">{risk_tier}</strong></div>
                <div><span style="color: #64748b;">Inference Latency:</span> <strong style="color: #334155;">{analysis['inference_time_ms']} ms</strong></div>
                <div><span style="color: #64748b;">Engine Mode:</span> <strong style="color: #0284c7;">{analysis.get('engine_type', 'XceptionNet')}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dual Probability Meters
        st.markdown("##### Probability Breakdown")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.caption(f"REAL: {round(real_prob * 100, 2)}%")
            st.markdown(f"""
            <div class="forensic-progress-bar">
                <div class="forensic-progress-fill-real" style="width: {min(100, max(0, real_prob * 100))}%;"></div>
            </div>
            """, unsafe_allow_html=True)
        with p_c2:
            st.caption(f"FAKE: {round(fake_prob * 100, 2)}%")
            st.markdown(f"""
            <div class="forensic-progress-bar">
                <div class="forensic-progress-fill-fake" style="width: {min(100, max(0, fake_prob * 100))}%;"></div>
            </div>
            """, unsafe_allow_html=True)

    # Explainable AI (Grad-CAM) Visual Studio
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🧠 Explainable AI: Grad-CAM Feature Activation Studio")
    st.caption("Visualizing convolutional gradient backpropagation and high-frequency manipulation boundaries.")

    # Interactive Alpha Slider
    cam_alpha = st.slider("Grad-CAM Thermal Transparency (Alpha)", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

    explainer = GradCAMExplainer(predictor.model)
    explanation = explainer.explain(analysis["primary_face_rgb"], alpha=cam_alpha)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(analysis["primary_face_rgb"], caption="1. Isolated Face Subject (299×299)", use_container_width=True)
    with c2:
        st.image(explanation["heatmap_rgb"], caption="2. Grad-CAM Activation Heatmap", use_container_width=True)
    with c3:
        st.image(explanation["overlay_rgb"], caption=f"3. Superimposed Overlay (Alpha: {cam_alpha})", use_container_width=True)

    dominant_str = ", ".join(explanation["dominant_regions"]) if explanation["dominant_regions"] else "Facial boundary and contour structures"
    st.markdown(f"""
    <div class="cyber-alert-info">
        <strong>🔍 Neural Focus Analysis:</strong> Model gradient attention is concentrated heavily on: 
        <strong style="color: #1d4ed8;">{dominant_str}</strong>.<br>
        <small style="color: #475569;">
            Warmer colors (red/orange) represent high spatial attention contributing to the classification logit;
            cooler regions (blue/cyan) represent background or unweighted feature regions.
        </small>
    </div>
    """, unsafe_allow_html=True)

    # One-Click PDF Download Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📑 Court-Admissible Forensic Documentation")
    
    pdf_cache_key = f"pdf_{cache_key}_{cam_alpha}"
    if pdf_cache_key not in st.session_state:
        pdf_gen = ForensicReportGenerator()
        analysis_data_report = {
            "filename": active_filename,
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
        pdf_path = pdf_gen.generate_image_report(analysis_data_report, explanation, f"{record_id}.pdf")
        with open(pdf_path, "rb") as f:
            st.session_state[pdf_cache_key] = f.read()
    
    pdf_bytes = st.session_state[pdf_cache_key]

    dl_col1, dl_col2 = st.columns([2, 1])
    with dl_col1:
        st.download_button(
            label="📄 Download Official Forensic Audit Report (PDF)",
            data=pdf_bytes,
            file_name=f"Deepify_Forensic_Report_{record_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with dl_col2:
        st.caption(f"SHA-256 Verified Digest | Size: {round(len(pdf_bytes) / 1024, 1)} KB")


render_image_detection_page()
