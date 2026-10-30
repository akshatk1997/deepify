"""
DEEPIFY - About Project & Technical Disclosures
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
        <div style="font-weight: 700; color: #0f172a; font-size: 1rem; display: flex; align-items: center; gap: 8px;">
            <span style="color: #2563eb;">🛡️ DEEPIFY</span>
            <span style="color: #cbd5e1;">|</span>
            <span style="color: #475569; font-weight: 500;">About Platform & Specifications</span>
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
        st.markdown('<div class="brand-title">ABOUT DEEPIFY</div>', unsafe_allow_html=True)
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
        <p style="color: #334155; line-height: 1.7; font-size: 0.95rem;">
            The exponential democratization of generative diffusion models, generative adversarial networks (GANs),
            and neural autoencoders (DeepFaceLab, FaceSwap, LiveLink) has enabled hyper-realistic manipulation of facial media.
            This synthetic capability introduces unprecedented security risks:
        </p>
        <ul style="color: #475569; line-height: 1.8; font-size: 0.9rem;">
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
            <h4 style="color: #0284c7; margin-top: 0; font-size: 1.05rem;">Machine Learning</h4>
            <ul style="color: #475569; font-size: 0.85rem; padding-left: 18px; line-height: 1.7; margin-bottom: 0;">
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
            <h4 style="color: #059669; margin-top: 0; font-size: 1.05rem;">Computer Vision</h4>
            <ul style="color: #475569; font-size: 0.85rem; padding-left: 18px; line-height: 1.7; margin-bottom: 0;">
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
            <h4 style="color: #7c3aed; margin-top: 0; font-size: 1.05rem;">Application UI</h4>
            <ul style="color: #475569; font-size: 0.85rem; padding-left: 18px; line-height: 1.7; margin-bottom: 0;">
                <li>Streamlit 1.62 Multi-Page</li>
                <li>Clean Light SaaS CSS</li>
                <li>Altair Charts Engine</li>
                <li>Interactive State Hub</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown("""
        <div class="deepguard-card" style="height: 100%;">
            <h4 style="color: #d97706; margin-top: 0; font-size: 1.05rem;">Forensic Export</h4>
            <ul style="color: #475569; font-size: 0.85rem; padding-left: 18px; line-height: 1.7; margin-bottom: 0;">
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
    | Operational Feature | Traditional Rule-Based | Generic MobileNet | DEEPIFY (Xception + Grad-CAM) |
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
        <strong style="color: #1e3a8a;">FINAL PROJECT TITLE FOR PORTFOLIO & RESUME:</strong><br>
        <em style="color: #0f172a; font-weight: 600;">DEEPIFY – Deep Learning-Based Image & Video Deepfake Detection and Media Authenticity Verification Using CNN, XceptionNet, OpenCV and Explainable AI.</em><br>
        <small style="color: #475569;">
            Engineered as an enterprise-grade applied machine learning and forensic computer vision platform suitable for capstone evaluations, technical interviews, and GitHub portfolio presentation.
        </small>
    </div>
    """, unsafe_allow_html=True)

    # Developer & Professional Contact Card
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🤝 Developer & Professional Inquiries")
    st.markdown("""
    <div class="deepguard-card" style="border-left: 4px solid #2563eb; background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 20px;">
            <div style="flex: 1; min-width: 280px;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #2563eb; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Developer & Maintainer</div>
                <h3 style="color: #0f172a; margin: 0 0 6px 0; font-size: 1.4rem;">Akshat Kumar</h3>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6; margin-bottom: 16px;">
                    Bridging B.Tech Computer Science and MBA AI & Data Science to ship scalable machine intelligence platforms, computer vision pipelines, and production SaaS architectures.
                </p>
                <div style="display: flex; flex-wrap: wrap; gap: 12px;">
                    <a href="https://github.com/akshatk1997" target="_blank" style="text-decoration: none; display: inline-flex; align-items: center; gap: 8px; font-size: 0.9rem; color: #ffffff; background: #0f172a; padding: 10px 18px; border-radius: 8px; font-weight: 600; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <svg height="18" width="18" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
                        GitHub: @akshatk1997
                    </a>
                    <a href="https://www.linkedin.com/in/akshatk1997" target="_blank" style="text-decoration: none; display: inline-flex; align-items: center; gap: 8px; font-size: 0.9rem; color: #ffffff; background: #0a66c2; padding: 10px 18px; border-radius: 8px; font-weight: 600; box-shadow: 0 2px 4px rgba(10,102,194,0.2);">
                        <svg height="18" width="18" viewBox="0 0 24 24" fill="#ffffff"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
                        Connect on LinkedIn
                    </a>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


render_about_page()
