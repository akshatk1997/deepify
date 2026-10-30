"""
DEEPGUARD AI - Video Deepfake Forensics Page
============================================
Interactive Streamlit interface for multi-frame video ingestion (MP4, AVI, MOV),
configurable temporal sampling, frame-level probability aggregation,
suspicious frame isolation, and forensic timeline visualization.
"""

import os
import time
import tempfile
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project paths
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
CSS_PATH = ASSETS_DIR / "styles.css"

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from utils.video_processing import VideoProcessor
from models.predict import DeepfakePredictor
from models.gradcam import GradCAMExplainer
from utils.report_generator import ForensicReportGenerator


def render_video_detection_page():
    st.markdown('<div class="brand-title">VIDEO TEMPORAL FORENSICS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">'
        'Verify temporal video streams for synthetic facial manipulation, deepfake replacements, '
        'and temporal flickering artifacts across uniformly sampled frame intervals.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    predictor = DeepfakePredictor()
    video_processor = VideoProcessor(predictor=predictor)

    # Configuration Controls
    col_upload, col_params = st.columns([5, 3])

    with col_upload:
        st.markdown("### 📤 Upload Video Forensic Target")
        uploaded_video = st.file_uploader(
            "Select video for deepfake verification (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"],
            help="Supported video containers: MP4, AVI, MOV (maximum 60MB)."
        )

    with col_params:
        st.markdown("### ⚙️ Sampling Configuration")
        sample_count = st.slider(
            "Sampled Frame Density",
            min_value=10,
            max_value=50,
            value=24,
            step=2,
            help="Uniformly samples N frames across the video timeline for temporal aggregation."
        )
        st.caption("Higher density increases forensic sensitivity at the cost of processing time.")

    if uploaded_video is None:
        st.info("👋 Upload a video file above to initiate multi-frame temporal forensic analysis.")
        return

    # Check file size limit (max 60MB)
    if uploaded_video.size > 60 * 1024 * 1024:
        st.error("Uploaded video exceeds 60MB limit. Please upload an optimized forensic clip.")
        return

    # Save uploaded video to a secure temporary file
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    suffix = Path(uploaded_video.name).suffix or ".mp4"
    temp_video_path = temp_dir / f"temp_{int(time.time())}{suffix}"

    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Display video preview and container metadata
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎬 Video Preview & Container Analysis")
    prev_col1, prev_col2 = st.columns([1, 1])

    with prev_col1:
        st.video(str(temp_video_path))

    meta = video_processor.get_video_metadata(str(temp_video_path))
    with prev_col2:
        st.markdown(f"""
        <div class="deepguard-card">
            <h4 style="color: #00d2ff; margin-top: 0;">Container Telemetry</h4>
            <div style="margin-bottom: 6px;"><span style="color: #94a3b8;">Filename:</span> <strong>{uploaded_video.name}</strong></div>
            <div style="margin-bottom: 6px;"><span style="color: #94a3b8;">Resolution:</span> <strong>{meta.get('resolution', 'N/A')}</strong></div>
            <div style="margin-bottom: 6px;"><span style="color: #94a3b8;">Framerate:</span> <strong>{meta.get('fps', 'N/A')} FPS</strong></div>
            <div style="margin-bottom: 6px;"><span style="color: #94a3b8;">Total Frames:</span> <strong>{meta.get('frame_count', 'N/A')}</strong></div>
            <div><span style="color: #94a3b8;">Duration:</span> <strong>{meta.get('duration_sec', 'N/A')} seconds</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # Check if model binary is present
    if not predictor.model_loaded:
        st.markdown("""
        <div class="cyber-alert-warning">
            <h4>⚠️ Model Binary Missing (Phase 3 Required)</h4>
            <p>The trained XceptionNet model weights (<code>saved_models/xception_deepfake.keras</code>) were not found.</p>
            <p>Please complete model training in Google Colab (Phase 3) or run <code>python models/train_model.py</code>.</p>
        </div>
        """, unsafe_allow_html=True)
        # Clean up temporary video
        if temp_video_path.exists():
            temp_video_path.unlink()
        return

    # Trigger Video Analysis
    st.markdown("---")
    st.subheader("⚡ Temporal Aggregation Verification")

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_progress(ratio: float, msg: str):
        progress_bar.progress(ratio)
        status_text.text(msg)

    try:
        results = video_processor.analyze_video_file(
            str(temp_video_path),
            target_sample_count=sample_count,
            progress_callback=update_progress
        )
    finally:
        # Cleanup temporary uploaded video
        if temp_video_path.exists():
            temp_video_path.unlink()

    status_text.empty()
    progress_bar.empty()

    # Handle Edge Case: INCONCLUSIVE (No usable faces)
    if results["status"] == "INCONCLUSIVE":
        st.markdown("""
        <div class="deepguard-card" style="border-color: rgba(245, 158, 11, 0.5);">
            <div class="badge-inconclusive">⚠️ VERDICT: INCONCLUSIVE</div>
            <h3 style="color: #f59e0b; margin-top: 14px;">NO USABLE FACIAL SUBJECTS LOCALIZED</h3>
            <p style="color: #cbd5e1;">
                The video stream was decoded across all sampled frames, but no clear human facial features
                could be isolated for neural inference. Authentic forensic verification requires unobstructed facial subjects.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Display Verdict
    verdict = results["verdict"]
    confidence = results["confidence"]
    risk_tier = results["risk_tier"]

    res_col1, res_col2 = st.columns([3, 2])
    with res_col1:
        if verdict == "FAKE":
            st.markdown("""
            <div class="deepguard-card" style="border-color: rgba(239, 68, 68, 0.4);">
                <div class="badge-fake">🚨 OVERALL VERDICT: FAKE VIDEO DETECTED</div>
                <h2 style="color: #ef4444; margin-top: 14px; margin-bottom: 6px;">TEMPORAL MANIPULATION CONFIRMED</h2>
                <p style="color: #cbd5e1; font-size: 0.95rem;">
                    Multi-frame probability aggregation identified recurring AI-generated facial artifacts
                    and blending anomalies across sampled video frames.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="deepguard-card" style="border-color: rgba(16, 185, 129, 0.4);">
                <div class="badge-real">🛡️ OVERALL VERDICT: REAL VIDEO VERIFIED</div>
                <h2 style="color: #10b981; margin-top: 14px; margin-bottom: 6px;">AUTHENTIC TEMPORAL STREAM CONFIRMED</h2>
                <p style="color: #cbd5e1; font-size: 0.95rem;">
                    Natural biometric continuity verified across all temporal frames without facial replacement artifacts.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="deepguard-card">
            <h4 style="color: #00d2ff; margin-top: 0;">Multi-Frame Aggregation Telemetry</h4>
            <div style="margin-bottom: 6px;"><span style="color: #94a3b8;">Aggregated Confidence:</span> <strong>{confidence}%</strong></div>
            <div style="margin-bottom: 6px;"><span style="color: #94a3b8;">Risk Classification:</span> <strong style="color: {'#ef4444' if verdict == 'FAKE' else '#10b981'};">{risk_tier}</strong></div>
            <div style="margin-bottom: 6px;"><span style="color: #94a3b8;">Manipulated Frame Ratio:</span> <strong>{results['manipulated_frames_count']} / {results['faces_detected']} ({results['manipulated_frames_ratio']}%)</strong></div>
            <div><span style="color: #94a3b8;">Processing Latency:</span> <strong>{results['processing_time_sec']}s</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # Frame Analysis Timeline Chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 Temporal Manipulation Timeline")
    st.caption("Distribution of deepfake manipulation probability across video playback timestamp (seconds).")

    timeline_data = []
    for item in results["timeline"]:
        if item["has_face"]:
            timeline_data.append({
                "Timestamp (s)": item["timestamp_sec"],
                "Fake Probability": item["fake_probability"],
                "Verdict": item["verdict"]
            })

    if timeline_data:
        df_timeline = pd.DataFrame(timeline_data)
        st.line_chart(df_timeline, x="Timestamp (s)", y="Fake Probability", color="#ef4444")

    # Suspicious Frames Gallery
    suspicious = results.get("suspicious_frames", [])
    if suspicious:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🚨 Suspicious Frames Gallery (Top Anomalies)")
        st.caption("Frames exhibiting highest neural manipulation probabilities and boundary seams.")

        gallery_cols = st.columns(min(3, len(suspicious)))
        for idx, s_frame in enumerate(suspicious[:3]):
            col_target = gallery_cols[idx]
            with col_target:
                st.image(
                    s_frame["full_frame_rgb"],
                    caption=f"Frame #{s_frame['frame_number']} at {s_frame['timestamp_sec']}s (Risk: {round(s_frame['fake_probability'] * 100, 1)}%)",
                    use_container_width=True
                )

        # Grad-CAM explanation for #1 most suspicious frame
        if predictor.model_loaded and suspicious[0].get("face_crop_rgb") is not None:
            st.markdown("#### 🔬 Representative Frame Grad-CAM Analysis")
            st.caption(f"Feature activation map for highest anomaly frame #{suspicious[0]['frame_number']}.")
            try:
                explainer = GradCAMExplainer(predictor.model)
                face_crop = suspicious[0]["face_crop_rgb"]
                cam_exp = explainer.explain(face_crop, alpha=0.45)
                
                v_c1, v_c2, v_c3 = st.columns(3)
                with v_c1:
                    st.image(face_crop, caption=f"Frame #{suspicious[0]['frame_number']} Face Crop", use_container_width=True)
                with v_c2:
                    st.image(cam_exp["heatmap_rgb"], caption="Grad-CAM Activation Heatmap", use_container_width=True)
                with v_c3:
                    st.image(cam_exp["overlay_rgb"], caption="Superimposed Forensic Overlay", use_container_width=True)
            except Exception as e:
                st.caption(f"Grad-CAM frame visualization notice: {str(e)}")

    # Record in Session History
    record_id = f"VID-{int(time.time())}"
    history_entry = {
        "id": record_id,
        "filename": uploaded_video.name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "media_type": "Video",
        "verdict": verdict,
        "confidence": f"{confidence}%",
        "fake_probability": results.get("mean_fake_probability", 0.0),
        "real_probability": results.get("mean_real_probability", 1.0),
        "faces_detected": results["faces_detected"],
        "latency_ms": int(results["processing_time_sec"] * 1000)
    }
    if not any(entry.get("id") == record_id for entry in st.session_state.analysis_history):
        st.session_state.analysis_history.insert(0, history_entry)

    # Download Official Video Forensic PDF Report
    st.markdown("<br>", unsafe_allow_html=True)
    pdf_gen = ForensicReportGenerator()
    results["filename"] = uploaded_video.name
    pdf_path = pdf_gen.generate_video_report(results, f"{record_id}.pdf")
    with open(pdf_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    st.download_button(
        label="📄 Download Official Video Forensic Audit Report (PDF)",
        data=pdf_bytes,
        file_name=f"DeepGuard_Video_Report_{record_id}.pdf",
        mime="application/pdf",
        use_container_width=True
    )


if __name__ == "__main__":
    render_video_detection_page()
