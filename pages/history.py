"""
DEEPGUARD AI - Forensic Detection History Page
==============================================
Provides persistent session tracking, tabular filtering, search,
and export capabilities for all media analyses conducted in the active session.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
CSS_PATH = ASSETS_DIR / "styles.css"
LOGO_PATH = ASSETS_DIR / "logo.png"

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


def render_history_page():
    st.markdown('<div class="brand-title">FORENSIC AUDIT HISTORY</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">'
        'Comprehensive chronological log of all image and video media verifications '
        'conducted across the current operational session.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    history = st.session_state.analysis_history

    if not history:
        st.markdown("""
        <div class="deepguard-card" style="text-align: center; padding: 48px;">
            <h3 style="color: #94a3b8;">No Investigation Records Found</h3>
            <p style="color: #64748b;">
                Run single-image or video forensic analyses to generate immutable session audit logs.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Filter & Search Controls
    filter_col1, filter_col2, filter_col3 = st.columns([3, 2, 2])

    with filter_col1:
        search_query = st.text_input("🔍 Search by Filename or Investigation ID", placeholder="e.g. IMG- or sample_...")

    with filter_col2:
        verdict_filter = st.selectbox("Filter by Verdict", ["ALL", "REAL", "FAKE"])

    with filter_col3:
        media_filter = st.selectbox("Filter by Media Type", ["ALL", "Image", "Video"])

    # Apply filters
    filtered_history = history.copy()

    if search_query:
        q = search_query.lower()
        filtered_history = [
            item for item in filtered_history
            if q in item.get("id", "").lower() or q in item.get("filename", "").lower()
        ]

    if verdict_filter != "ALL":
        filtered_history = [item for item in filtered_history if item.get("verdict") == verdict_filter]

    if media_filter != "ALL":
        filtered_history = [item for item in filtered_history if item.get("media_type") == media_filter]

    # Metrics Overview on Filtered Results
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Matched Records", len(filtered_history))
    with m2:
        reals = sum(1 for i in filtered_history if i.get("verdict") == "REAL")
        st.metric("Authentic (REAL) Matches", reals)
    with m3:
        fakes = sum(1 for i in filtered_history if i.get("verdict") == "FAKE")
        st.metric("Manipulated (FAKE) Matches", fakes)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabular Display
    df_history = pd.DataFrame(filtered_history)
    display_cols = ["id", "timestamp", "filename", "media_type", "verdict", "confidence", "faces_detected", "latency_ms"]
    available_cols = [c for c in display_cols if c in df_history.columns]

    st.dataframe(
        df_history[available_cols],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Actions: Export & Reset
    act_col1, act_col2, act_col3 = st.columns([2, 2, 3])

    with act_col1:
        csv_data = df_history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export History (CSV)",
            data=csv_data,
            file_name="deepguard_forensic_history.csv",
            mime="text/csv",
            use_container_width=True
        )

    with act_col2:
        json_data = json.dumps(filtered_history, indent=4).encode("utf-8")
        st.download_button(
            "📥 Export History (JSON)",
            data=json_data,
            file_name="deepguard_forensic_history.json",
            mime="application/json",
            use_container_width=True
        )

    with act_col3:
        if st.button("🗑️ Clear Active Session History", use_container_width=True):
            st.session_state.analysis_history = []
            st.rerun()


if __name__ == "__main__":
    render_history_page()
