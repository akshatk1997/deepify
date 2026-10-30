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
    # Top Hero
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=100)
    with col_title:
        st.markdown('<div class="brand-title">ABOUT DEEPGUARD AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="brand-subtitle">'
            'Next-generation computer vision and Explainable AI platform '
            'for automated digital media authentication and deepfake forensics.'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Problem Statement Card
    st.subheader("🎯 The Problem: Synthetic Media Threat Landscape")
    st.markdown("""
    <div class="deepguard-card">
        <p style="color: #cbd5e1; line-height: 1.7; font-size: 0.95rem;">
            The exponential advancement of generative diffusion models, generative adversarial networks (GANs),
            and neural autoencoders (such as DeepFaceLab, FaceSwap, and LiveLink) has democratized the creation of
            photorealistic hyper-manipulated facial media. The misuse of deepfakes introduces acute risks:
        </p>
        <ul style="color: #94a3b8; line-height: 1.8; font-size: 0.9rem;">
            <li><strong>Financial & Identity Fraud:</strong> Bypassing biometric Know-Your-Customer (KYC) liveness verification.</li>
            <li><strong>Democratic Misinformation:</strong> Disseminating falsified political speech and synthesized statements.</li>
            <li><strong>Reputational Sabotage:</strong> Fabricating non-consensual defamatory imagery and impersonations.</li>
            <li><strong>Erosion of Epistemic Trust:</strong> Fostering the "Liar's Dividend," where authentic evidence is dismissed as AI-generated.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Core Engineering Objectives
    st.subheader("🛡️ DeepGuard AI Architectural Objectives")
    st.markdown("""
    <div class="deepguard-card">
        <ol style="color: #cbd5e1; line-height: 1.8; font-size: 0.95rem;">
            <li><strong>Non-Heuristic Machine Learning:</strong> Replace fragile rule-based artifact detectors with a deep XceptionNet model trained on depthwise separable convolutions.</li>
            <li><strong>Dual-Modality Support:</strong> Seamlessly analyze both high-resolution static imagery (JPG, PNG) and multi-frame video streams (MP4, AVI, MOV).</li>
            <li><strong>Temporal Probability Aggregation:</strong> Guard against single-frame false alarms by evaluating uniform multi-frame sample intervals.</li>
            <li><strong>Explainable AI (XAI):</strong> Demystify predictions with real-time Grad-CAM thermal overlays showing exact convolutional focus regions.</li>
            <li><strong>Automated Forensic Audit Documentation:</strong> Deliver court-admissible and compliance-ready cryptographic PDF reports via ReportLab.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Technology Stack
    st.subheader("🛠️ Technology Stack Breakdown")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("""
        <div class="deepguard-card">
            <h4 style="color: #00d2ff; margin-top: 0;">Machine Learning</h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 18px;">
                <li>TensorFlow 2.x</li>
                <li>Keras 3</li>
                <li>XceptionNet</li>
                <li>Scikit-Learn</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="deepguard-card">
            <h4 style="color: #10b981; margin-top: 0;">Computer Vision</h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 18px;">
                <li>OpenCV Headless</li>
                <li>Haar Cascades</li>
                <li>Pillow (PIL)</li>
                <li>Grad-CAM XAI</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="deepguard-card">
            <h4 style="color: #8b5cf6; margin-top: 0;">Application UI</h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 18px;">
                <li>Streamlit Multi-Page</li>
                <li>Custom Dark CSS</li>
                <li>Altair Charts</li>
                <li>Session State Hub</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown("""
        <div class="deepguard-card">
            <h4 style="color: #f59e0b; margin-top: 0;">Forensic Export</h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 18px;">
                <li>ReportLab PDF Engine</li>
                <li>Audit Hash Generation</li>
                <li>JSON Telemetry</li>
                <li>CSV Audit Logs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Future Enhancements Roadmap
    st.subheader("🚀 Future Development Roadmap")
    st.markdown("""
    <div class="deepguard-card">
        <ul style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.8;">
            <li><strong>Audio-Visual Multimodal Synchronization:</strong> Cross-correlating phoneme-viseme speech waveforms with lip movement vectors to detect cloned voice synthesis.</li>
            <li><strong>Vision Transformer (ViT) Ensembling:</strong> Integrating Swin-Transformer and EfficientNet-B7 backbones for ensemble majority voting.</li>
            <li><strong>Decentralized Content Provenance:</strong> Storing cryptographic SHA-256 media perceptual hashes on decentralized ledgers adhering to C2PA standards.</li>
            <li><strong>Live Video Stream WebRTC Ingestion:</strong> Real-time frame inspection for live conference calls and digital KYC pipelines.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Professional Resume & Capstone Summary Box
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="cyber-alert-info">
        <strong>📋 Capstone & Portfolio Citation:</strong><br>
        <em>DEEPGUARD AI – Deep Learning-Based Image & Video Deepfake Detection and Media Authenticity Verification Using CNN, XceptionNet, OpenCV and Explainable AI.</em><br>
        Developed as an enterprise-grade applied machine learning and forensic computer vision platform.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_about_page()
