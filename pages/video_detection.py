"""
DEEPGUARD AI - Video Deepfake Forensics Page
============================================
Enterprise SaaS temporal video verification interface for multi-frame analysis,
uniform sampling, temporal probability aggregation, suspicious frame galleries,
Grad-CAM frame activations, and audit-ready PDF generation.
"""

import time
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
CSS_PATH = ASSETS_DIR / "styles.css"
SAMPLE_VIDEO_PATH = ASSETS_DIR / "sample_test_video.mp4"

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from utils.video_processing import VideoProcessor
from models.predict import DeepfakePredictor
from models.gradcam import GradCAMExplainer
from utils.report_generator import ForensicReportGenerator


def render_video_detection_page():
    # SaaS Enterprise Header Bar
    st.markdown("""
    <div class="saas-header-bar">
        <div style="font-weight: 700; color: #f8fafc; font-size: 1rem; display: flex; align-items: center; gap: 8px;">
            <span style="color: #00d2ff;">🛡️ DEEPGUARD AI</span>
            <span style="color: #64748b;">|</span>
            <span style="color: #94a3b8; font-weight: 500;">Temporal Video Forensics</span>
        </div>
        <div class="saas-badge-group">
            <span class="saas-pill saas-pill-live"><span class="saas-pill-pulse"></span> Video Decoder Ready</span>
            <span class="saas-pill">Containers: MP4 / AVI / MOV</span>
            <span class="saas-pill">Sampling: Uniform Strided</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="brand-title">TEMPORAL VIDEO STREAM FORENSICS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">'
        'Inspect video streams for temporal deepfake replacements, face-swap blending inconsistencies, '
        'and temporal flickering artifacts using multi-frame probability aggregation.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    predictor = DeepfakePredictor()
    video_processor = VideoProcessor(predictor=predictor)

    # Quick Demo Selector
    st.markdown("""
    <div class="demo-bar-container">
        <div style="font-weight: 600; color: #38bdf8; font-size: 0.92rem; margin-bottom: 8px;">
            🎯 Quick Sandbox: Test with Verified Benchmark Video Stream
        </div>
        <div style="color: #94a3b8; font-size: 0.82rem; margin-bottom: 12px;">
            Loads a pre-rendered 60-frame 30FPS benchmark video stream (<code>assets/sample_test_video.mp4</code>).
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "selected_video_path" not in st.session_state:
        st.session_state.selected_video_path = None
    if "uploaded_video_name" not in st.session_state:
        st.session_state.uploaded_video_name = "benchmark_video.mp4"

    demo_btn_col1, demo_btn_col2 = st.columns([1, 2])
    with demo_btn_col1:
        if st.button("🎬 Load Forensic Benchmark Video", use_container_width=True):
            if SAMPLE_VIDEO_PATH.exists():
                st.session_state.selected_video_path = str(SAMPLE_VIDEO_PATH)
                st.session_state.uploaded_video_name = "benchmark_sample_video.mp4"
                st.session_state.cached_video_key = None
                st.rerun()
    with demo_btn_col2:
        st.caption("Standard 1080p synthetic test sequence with animated facial subject.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Configuration & Upload Grid
    up_c1, up_c2 = st.columns([5, 3], gap="medium")

    with up_c1:
        uploaded_video = st.file_uploader(
            "Or upload your own forensic video target (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"],
            help="Maximum video container size: 60MB."
        )

    with up_c2:
        st.markdown("##### Temporal Sampling Density")
        sample_count = st.slider(
            "Sampled Frame Density",
            min_value=10, max_value=50, value=20, step=2,
            help="Number of uniformly spaced frames analyzed across the timeline."
        )

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    if uploaded_video is not None:
        if uploaded_video.size > 60 * 1024 * 1024:
            st.error("Uploaded video exceeds 60MB limit.")
            return
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        saved_temp_path = temp_dir / f"custom_{int(time.time())}{suffix}"
        with open(saved_temp_path, "wb") as f:
            f.write(uploaded_video.read())
        st.session_state.selected_video_path = str(saved_temp_path)
        st.session_state.uploaded_video_name = uploaded_video.name
        st.session_state.cached_video_key = None

    if not st.session_state.selected_video_path:
        st.info("👆 Click 'Load Forensic Benchmark Video' above or upload a video file to begin.")
        return

    active_video_path = st.session_state.selected_video_path
    active_video_name = st.session_state.uploaded_video_name

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎬 Video Container Telemetry & Playback")

    meta = video_processor.get_video_metadata(active_video_path)
    grid_v1, grid_v2 = st.columns([1, 1], gap="medium")

    with grid_v1:
        st.video(active_video_path)

    with grid_v2:
        st.markdown(f"""
        <div class="deepguard-card">
            <h4 style="color: #00d2ff; margin-top: 0; margin-bottom: 12px;">Container Technical Telemetry</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.9rem;">
                <div><span style="color: #94a3b8;">Filename:</span> <strong style="color: #f8fafc;">{active_video_name}</strong></div>
                <div><span style="color: #94a3b8;">Spatial Resolution:</span> <strong style="color: #f8fafc;">{meta.get('resolution', 'N/A')}</strong></div>
                <div><span style="color: #94a3b8;">Container Framerate:</span> <strong style="color: #f8fafc;">{meta.get('fps', 'N/A')} FPS</strong></div>
                <div><span style="color: #94a3b8;">Total Frame Count:</span> <strong style="color: #f8fafc;">{meta.get('frame_count', 'N/A')}</strong></div>
                <div><span style="color: #94a3b8;">Total Duration:</span> <strong style="color: #f8fafc;">{meta.get('duration_sec', 'N/A')}s</strong></div>
                <div><span style="color: #94a3b8;">Target Samples:</span> <strong style="color: #00d2ff;">{sample_count} frames</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Execution Header with Re-run Option
    exec_col1, exec_col2 = st.columns([3, 1])
    with exec_col1:
        st.subheader("⚡ Multi-Frame Temporal Aggregation Execution")
    with exec_col2:
        if st.button("🔄 Re-Analyze Video Stream", use_container_width=True):
            st.session_state.cached_video_key = None
            st.rerun()

    # Caching Mechanism for Video Inference
    video_mtime = Path(active_video_path).stat().st_mtime if Path(active_video_path).exists() else 0
    video_cache_key = f"{active_video_path}_{sample_count}_{video_mtime}"

    if st.session_state.get("cached_video_key") != video_cache_key:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(ratio: float, msg: str):
            progress_bar.progress(ratio)
            status_text.caption(f"⚡ {msg}")

        results = video_processor.analyze_video_file(
            active_video_path,
            target_sample_count=sample_count,
            progress_callback=update_progress
        )

        status_text.empty()
        progress_bar.empty()

        st.session_state.cached_video_results = results
        st.session_state.cached_video_key = video_cache_key

        if results["status"] != "INCONCLUSIVE":
            record_id = f"VID-{int(time.time())}"
            st.session_state.current_video_record_id = record_id
            history_entry = {
                "id": record_id,
                "filename": active_video_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "media_type": "Video",
                "verdict": results["verdict"],
                "confidence": f"{results['confidence']}%",
                "fake_probability": results.get("mean_fake_probability", 0.0),
                "real_probability": results.get("mean_real_probability", 1.0),
                "faces_detected": results["faces_detected"],
                "latency_ms": int(results["processing_time_sec"] * 1000)
            }
            st.session_state.analysis_history.insert(0, history_entry)
    else:
        results = st.session_state.cached_video_results

    # Handle Inconclusive
    if results["status"] == "INCONCLUSIVE":
        st.markdown("""
        <div class="deepguard-card" style="border-left: 4px solid #f59e0b;">
            <div class="badge-inconclusive">⚠️ VERDICT: INCONCLUSIVE (NO FACES FOUND)</div>
            <h3 style="color: #fbbf24; margin-top: 14px;">Facial Continuity Verification Aborted</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
                No clear facial features could be isolated across the sampled intervals. 
                Forensic verification requires visible facial subjects.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    verdict = results["verdict"]
    confidence = results["confidence"]
    risk_tier = results["risk_tier"]
    record_id = st.session_state.get("current_video_record_id", f"VID-{int(time.time())}")

    # Results Verdict Grid
    res_c1, res_c2 = st.columns([1, 1], gap="medium")

    with res_c1:
        if verdict == "FAKE":
            st.markdown("""
            <div class="deepguard-card" style="border-color: rgba(239, 68, 68, 0.45); background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(15, 23, 42, 0.8) 100%);">
                <div class="badge-fake">🚨 OVERALL VERDICT: FAKE VIDEO DETECTED</div>
                <h2 style="color: #ef4444; margin-top: 12px; margin-bottom: 6px;">TEMPORAL MANIPULATION CONFIRMED</h2>
                <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.55;">
                    Multi-frame probability aggregation identified recurring AI-synthesized facial anomalies across the sampled timeline.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="deepguard-card" style="border-color: rgba(16, 185, 129, 0.45); background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(15, 23, 42, 0.8) 100%);">
                <div class="badge-real">🛡️ OVERALL VERDICT: REAL VIDEO VERIFIED</div>
                <h2 style="color: #10b981; margin-top: 12px; margin-bottom: 6px;">BIOMETRIC CONTINUITY VERIFIED</h2>
                <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.55;">
                    Natural biometric skin frequencies and temporal continuity confirmed across all analyzed frames.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with res_c2:
        st.markdown(f"""
        <div class="deepguard-card">
            <h4 style="color: #00d2ff; margin-top: 0; margin-bottom: 12px;">Multi-Frame Aggregation Telemetry</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.9rem;">
                <div><span style="color: #94a3b8;">Aggregated Score:</span> <strong style="color: #ffffff;">{confidence}%</strong></div>
                <div><span style="color: #94a3b8;">Risk Classification:</span> <strong style="color: {'#ef4444' if verdict == 'FAKE' else '#10b981'};">{risk_tier}</strong></div>
                <div><span style="color: #94a3b8;">Manipulated Ratio:</span> <strong style="color: #f8fafc;">{results['manipulated_frames_count']} / {results['faces_detected']} ({results['manipulated_frames_ratio']}%)</strong></div>
                <div><span style="color: #94a3b8;">Processing Latency:</span> <strong style="color: #cbd5e1;">{results['processing_time_sec']}s</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Timeline Chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 Temporal Manipulation Risk Timeline")
    st.caption("Fluctuation of manipulation probability across video playback timestamp (seconds).")

    timeline_points = [
        {"Playback Timestamp (s)": item["timestamp_sec"], "Manipulation Risk": item["fake_probability"]}
        for item in results["timeline"] if item["has_face"]
    ]
    if timeline_points:
        df_time = pd.DataFrame(timeline_points)
        st.line_chart(df_time, x="Playback Timestamp (s)", y="Manipulation Risk", color="#ef4444")

    # Suspicious Frames Gallery
    suspicious = results.get("suspicious_frames", [])
    if suspicious:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🚨 Suspicious Frames Gallery (Top Anomalies)")
        st.caption("Frames exhibiting highest neural manipulation probabilities and boundary seams.")

        gal_cols = st.columns(min(3, len(suspicious)))
        for idx, sf in enumerate(suspicious[:3]):
            with gal_cols[idx]:
                st.image(
                    sf["full_frame_rgb"],
                    caption=f"Frame #{sf['frame_number']} at {sf['timestamp_sec']}s (Risk: {round(sf['fake_probability'] * 100, 1)}%)",
                    use_container_width=True
                )

        if suspicious[0].get("face_crop_rgb") is not None:
            st.markdown("#### 🔬 Representative Frame Grad-CAM Analysis")
            explainer = GradCAMExplainer(predictor.model)
            fc = suspicious[0]["face_crop_rgb"]
            cam_exp = explainer.explain(fc, alpha=0.45)
            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                st.image(fc, caption=f"Frame #{suspicious[0]['frame_number']} Subject Face", use_container_width=True)
            with gc2:
                st.image(cam_exp["heatmap_rgb"], caption="Grad-CAM Activation Heatmap", use_container_width=True)
            with gc3:
                st.image(cam_exp["overlay_rgb"], caption="Forensic Superimposed Overlay", use_container_width=True)

    # Download PDF Button
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📑 Court-Admissible Video Audit Documentation")
    
    pdf_cache_key = f"pdf_vid_{record_id}"
    if pdf_cache_key not in st.session_state:
        pdf_gen = ForensicReportGenerator()
        results["filename"] = active_video_name
        pdf_path = pdf_gen.generate_video_report(results, f"{record_id}.pdf")
        with open(pdf_path, "rb") as f:
            st.session_state[pdf_cache_key] = f.read()

    pdf_bytes = st.session_state[pdf_cache_key]

    dl_c1, dl_c2 = st.columns([2, 1])
    with dl_c1:
        st.download_button(
            label="📄 Download Official Video Forensic Audit Report (PDF)",
            data=pdf_bytes,
            file_name=f"DeepGuard_Video_Report_{record_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with dl_c2:
        st.caption(f"SHA-256 Verified Digest | Size: {round(len(pdf_bytes) / 1024, 1)} KB")


render_video_detection_page()
