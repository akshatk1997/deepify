"""
DEEPIFY - Forensic Detection History Page
==============================================
Provides persistent session tracking, tabular filtering, search,
and export capabilities for all media analyses conducted in the active session.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
CSS_PATH = ASSETS_DIR / "styles.css"

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


def seed_demo_history():
    now = datetime.now()
    demo_entries = [
        {"id": "IMG-1730331001", "filename": "suspect_interview_crop.png", "timestamp": (now - timedelta(minutes=14)).strftime("%Y-%m-%d %H:%M:%S"), "media_type": "Image", "verdict": "FAKE", "confidence": "96.42%", "fake_probability": 0.9642, "real_probability": 0.0358, "faces_detected": 1, "latency_ms": 28.4},
        {"id": "VID-1730330890", "filename": "executive_announcement.mp4", "timestamp": (now - timedelta(minutes=32)).strftime("%Y-%m-%d %H:%M:%S"), "media_type": "Video", "verdict": "FAKE", "confidence": "89.20%", "fake_probability": 0.8920, "real_probability": 0.1080, "faces_detected": 24, "latency_ms": 1420},
        {"id": "IMG-1730330540", "filename": "kyc_id_card_photo.jpg", "timestamp": (now - timedelta(hours=1, minutes=5)).strftime("%Y-%m-%d %H:%M:%S"), "media_type": "Image", "verdict": "REAL", "confidence": "94.80%", "fake_probability": 0.0520, "real_probability": 0.9480, "faces_detected": 1, "latency_ms": 22.1},
        {"id": "IMG-1730329910", "filename": "conference_panelist_01.png", "timestamp": (now - timedelta(hours=2, minutes=18)).strftime("%Y-%m-%d %H:%M:%S"), "media_type": "Image", "verdict": "REAL", "confidence": "98.15%", "fake_probability": 0.0185, "real_probability": 0.9815, "faces_detected": 1, "latency_ms": 24.0},
        {"id": "VID-1730328400", "filename": "press_briefing_stream.mp4", "timestamp": (now - timedelta(hours=3, minutes=45)).strftime("%Y-%m-%d %H:%M:%S"), "media_type": "Video", "verdict": "REAL", "confidence": "93.50%", "fake_probability": 0.0650, "real_probability": 0.9350, "faces_detected": 30, "latency_ms": 1820}
    ]
    st.session_state.analysis_history = demo_entries


def render_history_page():
    # SaaS Header Bar
    history = st.session_state.analysis_history
    st.markdown(f"""
    <div class="saas-header-bar">
        <div style="font-weight: 700; color: #0f172a; font-size: 1rem; display: flex; align-items: center; gap: 8px;">
            <span style="color: #2563eb;">🛡️ DEEPIFY</span>
            <span style="color: #cbd5e1;">|</span>
            <span style="color: #475569; font-weight: 500;">Forensic Audit Trail</span>
        </div>
        <div class="saas-badge-group">
            <span class="saas-pill saas-pill-live"><span class="saas-pill-pulse"></span> Audit Logging Active</span>
            <span class="saas-pill">Records: {len(history)}</span>
            <span class="saas-pill">Storage: In-Memory Encrypted</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="brand-title">FORENSIC AUDIT HISTORY</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">'
        'Comprehensive chronological log of all image and video media verifications '
        'conducted across the current operational session.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    if not history:
        st.markdown("""
        <div class="deepguard-card" style="text-align: center; padding: 36px;">
            <h3 style="color: #334155; margin-top: 0;">No Active Investigation Records Found</h3>
            <p style="color: #64748b; font-size: 0.95rem; line-height: 1.6;">
                Run single-image or video forensic analyses to generate immutable session audit logs,
                or click below to load pre-configured demonstration investigation records.
            </p>
        </div>
        """, unsafe_allow_html=True)
        btn_c1, btn_c2, btn_c3 = st.columns([1, 2, 1])
        with btn_c2:
            if st.button("📊 Seed Demonstration Forensic Audit Log", use_container_width=True):
                seed_demo_history()
                st.rerun()
        return

    # Filter Toolbar
    f_c1, f_c2, f_c3 = st.columns([3, 2, 2], gap="small")
    with f_c1:
        search_query = st.text_input("🔍 Search by Target Filename or Audit ID", placeholder="e.g. IMG- or interview...")
    with f_c2:
        verdict_filter = st.selectbox("Verdict Filter", ["ALL", "REAL", "FAKE"])
    with f_c3:
        media_filter = st.selectbox("Media Type Filter", ["ALL", "Image", "Video"])

    filtered = history.copy()
    if search_query:
        q = search_query.lower()
        filtered = [i for i in filtered if q in i.get("id", "").lower() or q in i.get("filename", "").lower()]
    if verdict_filter != "ALL":
        filtered = [i for i in filtered if i.get("verdict") == verdict_filter]
    if media_filter != "ALL":
        filtered = [i for i in filtered if i.get("media_type") == media_filter]

    # Metrics on Filtered Results
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="stat-pill"><span class="stat-value">{len(filtered)}</span><span class="stat-label">Matching Audit Records</span></div>', unsafe_allow_html=True)
    with m2:
        reals = sum(1 for i in filtered if i.get("verdict") == "REAL")
        st.markdown(f'<div class="stat-pill"><span class="stat-value" style="color: #059669;">{reals}</span><span class="stat-label">Authentic (REAL) Matches</span></div>', unsafe_allow_html=True)
    with m3:
        fakes = sum(1 for i in filtered if i.get("verdict") == "FAKE")
        st.markdown(f'<div class="stat-pill"><span class="stat-value" style="color: #dc2626;">{fakes}</span><span class="stat-label">Manipulated (FAKE) Alerts</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Data Table
    df_history = pd.DataFrame(filtered)
    cols = ["id", "timestamp", "filename", "media_type", "verdict", "confidence", "faces_detected", "latency_ms"]
    display_cols = [c for c in cols if c in df_history.columns]

    st.dataframe(
        df_history[display_cols],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Export & Reset Toolbar
    act_c1, act_c2, act_c3 = st.columns([2, 2, 2], gap="medium")
    with act_c1:
        csv_bytes = df_history.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export Audit Trail (CSV)", data=csv_bytes, file_name="deepify_audit_trail.csv", mime="text/csv", use_container_width=True)
    with act_c2:
        json_bytes = json.dumps(filtered, indent=4).encode("utf-8")
        st.download_button("📥 Export Audit Trail (JSON)", data=json_bytes, file_name="deepify_audit_trail.json", mime="application/json", use_container_width=True)
    with act_c3:
        if st.button("🗑️ Reset Session Audit Trail", use_container_width=True):
            st.session_state.analysis_history = []
            st.rerun()


render_history_page()
