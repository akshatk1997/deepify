"""
DEEPGUARD AI - About Project & Technical Disclosures
====================================================
Comprehensive documentation covering problem formulation, engineering objectives,
architectural stack, academic references, future development roadmap,
and capstone portfolio summary.
"""

from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
CSS_PATH = ASSETS_DIR / "styles.css"
LOGO_PATH = ASSETS_DIR / "logo.png"

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_about_page():
    # SaaS Header Bar
    st.markdown("""
    <div class="saas-header-bar">
        <div style="font-weight: 700; color: #f8fafc; font-size: 1rem; display: flex; align-items: center; gap: 8px;">
            <span style="color: #00d2ff;">🛡️ DEEPGUARD AI</span>
            <span style="color: #64748b;">|</span>
            <span style="color: #94a3b8; font-weight: 500;">About Platform & Specifications</span>
        </div>
        <div class="saas-badge-group">
            <span class="saas-pill saas-pill-live"><span class="saas-pill-pulse"></span> Production Ready</span>
            <span class="saas-pill">Version: 2.4.0 Enterprise</span>
            <span class="saas-pill">License: MIT Open Source</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=90)
    with col_title:
        st.markdown('<div class="brand-title">ABOUT DEEPGUARD AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="brand-subtitle">'
            'Next-generation computer vision intelligence platform for automated facial deepfake detection, '
            'media authenticity verification, and Explainable AI (Grad-CAM) diagnostics.'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Problem Statement Card
    st.subheader("🎯 The Problem: Generative AI & Biometric Manipulation")
    st.markdown("""
    <div class="deepguard-card">
        <p style="color: #cbd5e1; line-height: 1.7; font-size: 0.95rem;">
            The exponential democratization of generative diffusion models, generative adversarial networks (GANs),
            and neural autoencoders (DeepFaceLab, FaceSwap, LiveLink) has enabled hyper-realistic manipulation of facial media.
            This synthetic capability introduces unprecedented security risks:
        </p>
        <ul style="color: #94a3b8; line-height: 1.8; font-size: 0.9rem;">
            <li><strong>Financial Fraud & KYC Spoofing:</strong> Bypassing automated liveness checks in remote banking and fintech onboarding.</li>
            <li><strong>Disinformation & Epistemic Trust Erosion:</strong> Fabricating non-consensual defamatory media and counterfeit statements.</li>
            <li><strong>Corporate Impersonation:</strong> Executing high-value Business Email Compromise (BEC) and video conference CEO fraud.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Technology Stack
    st.subheader("🛠️ Technology Stack Breakdown")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("""
        <div class="deepguard-card" style="height: 100%;">
            <h4 style="color: #00d2ff; margin-top: 0; font-size: 1.05rem;">Machine Learning</h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 18px; line-height: 1.7; margin-bottom: 0;">
                <li>TensorFlow 2.x</li>
                <li>Keras 3 Backbone</li>
                <li>XceptionNet (ImageNet)</li>
                <li>Scikit-Learn Metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="deepguard-card" style="height: 100%;">
            <h4 style="color: #10b981; margin-top: 0; font-size: 1.05rem;">Computer Vision</h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 18px; line-height: 1.7; margin-bottom: 0;">
                <li>OpenCV Headless</li>
                <li>Haar Cascade Localization</li>
                <li>20% Margin Expansion</li>
                <li>Grad-CAM Backprop</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="deepguard-card" style="height: 100%;">
            <h4 style="color: #8b5cf6; margin-top: 0; font-size: 1.05rem;">Application UI</h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 18px; line-height: 1.7; margin-bottom: 0;">
                <li>Streamlit 1.62 Multi-Page</li>
                <li>Glassmorphism Dark CSS</li>
                <li>Altair Charts Engine</li>
                <li>Interactive State Hub</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown("""
        <div class="deepguard-card" style="height: 100%;">
            <h4 style="color: #f59e0b; margin-top: 0; font-size: 1.05rem;">Forensic Export</h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 18px; line-height: 1.7; margin-bottom: 0;">
                <li>ReportLab 5.x Engine</li>
                <li>Cryptographic SHA-256</li>
                <li>Dual Visual Evidence</li>
                <li>CSV / JSON Audit Trails</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Enterprise Feature Matrix
    st.subheader("⚖️ Enterprise Capability Comparison Matrix")
    st.markdown("""
    | Operational Feature | Traditional Rule-Based | Generic MobileNet | DEEPGUARD AI (Xception + Grad-CAM) |
    | :--- | :---: | :---: | :---: |
    | **Depthwise Cross-Channel Separation** | ❌ No | Partial | ✅ **Full 36-Stage Decoupled Topology** |
    | **Boundary Margin Expansion (20%)** | ❌ Tightly cropped | ❌ Cut off | ✅ **Preserves Blending Seams** |
    | **Multi-Frame Temporal Aggregation** | ❌ Single-frame only | ❌ Vulnerable to spikes | ✅ **Top-K Burst + Ratio Aggregation** |
    | **Explainable AI (Grad-CAM)** | ❌ Black box | ❌ None | ✅ **Real-Time Thermal Overlay** |
    | **Audit-Ready Cryptographic PDF** | ❌ Manual screenshots | ❌ Text log only | ✅ **ReportLab with SHA-256 Hashes** |
    | **Zero-Face Rejection Standard** | ❌ False classification | ❌ Forced prediction | ✅ **Graceful Inconclusive Guard** |
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Resume & Citation Box
    st.subheader("📋 Academic & Portfolio Citation")
    st.markdown("""
    <div class="cyber-alert-info">
        <strong>FINAL PROJECT TITLE FOR PORTFOLIO & RESUME:</strong><br>
        <em>DEEPGUARD AI – Deep Learning-Based Image & Video Deepfake Detection and Media Authenticity Verification Using CNN, XceptionNet, OpenCV and Explainable AI.</em><br>
        <small style="color: #94a3b8;">
            Engineered as an enterprise-grade applied machine learning and forensic computer vision platform suitable for capstone evaluations, technical interviews, and GitHub portfolio presentation.
        </small>
    </div>
    """, unsafe_allow_html=True)


render_about_page()
