# DEEPGUARD AI – Deepfake Detection & Media Authenticity Verification System

[![GitHub Repo](https://img.shields.io/badge/GitHub-akshatk1997%2Fdeepguard--ai-181717.svg?style=flat&logo=github)](https://github.com/akshatk1997/deepguard-ai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Akshat%20Kumar-0A66C2.svg?style=flat&logo=linkedin)](https://www.linkedin.com/in/akshatk1997)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Architecture-XceptionNet-red.svg)](https://keras.io/)
[![Explainability](https://img.shields.io/badge/XAI-Grad--CAM-purple.svg)](https://github.com/ramprs/grad-cam)
[![Frontend](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Portfolio & Capstone Resume Title:**  
> *DEEPGUARD AI – Deep Learning-Based Image & Video Deepfake Detection and Media Authenticity Verification Using CNN, XceptionNet, OpenCV and Explainable AI.*

---

## 1. Project Overview

**DEEPGUARD AI** is an enterprise-grade forensic analysis system designed to verify the authenticity of facial imagery and video streams. By leveraging **XceptionNet** (Extreme Inception) feature extractors with depthwise separable convolutions, coupled with **Haar/DNN Face Detection** and **Gradient-weighted Class Activation Mapping (Grad-CAM)**, DEEPGUARD AI provides transparent, explainable predictions classifying media as **REAL** (authentic) or **FAKE** (AI-synthesized / manipulated).

### Key Highlights
- **Dual-Mode Analysis**: Real-time image forensics and frame-sampled temporal video verification.
- **Explainable AI (XAI)**: Integrated Grad-CAM heatmaps highlight manipulated facial artifacts (e.g., blending boundaries, unnatural texture frequencies, eye/mouth synthesis flaws).
- **Forensic PDF Reports**: One-click generation of court- and audit-ready PDF investigation reports with ReportLab.
- **Cybersecurity Dark Aesthetic**: Custom-engineered Streamlit dashboard featuring glassmorphic controls, interactive metrics, and analysis history.
- **Hardware Agnostic**: Google Colab GPU-optimized training pipelines alongside lightweight CPU-compatible local inference fallbacks.

---

## 2. System Architecture

```
                  ┌────────────────────────────────────────┐
                  │            User Media Upload           │
                  │         (JPG, PNG, MP4, AVI, MOV)      │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │     Face Detection & Alignment         │
                  │        (OpenCV Cascade / DNN)          │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ Preprocessing & Normalization (299x299)│
                  │       (Xception Scaling: [-1, 1])      │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    Trained XceptionNet Deep Model      │
                  │   (Depthwise Separable Convolutions)   │
                  └─────────┬────────────────────┬─────────┘
                            │                    │
                            ▼                    ▼
             ┌─────────────────────────┐  ┌─────────────────────────┐
             │ Softmax / Sigmoid Logit │  │  Grad-CAM Backprop      │
             │   REAL vs FAKE Score    │  │  Attention Heatmap Gen  │
             └──────────────┬──────────┘  └──────────────┬──────────┘
                            │                            │
                            └─────────────┬──────────────┘
                                          │
                                          ▼
                  ┌────────────────────────────────────────┐
                  │   Streamlit Forensic Cybersecurity UI  │
                  │       & ReportLab PDF Export           │
                  └────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
deepguard-ai/
├── app.py                      # Primary Streamlit application entry point
├── requirements.txt            # Pinned, conflict-tested dependencies
├── README.md                   # Comprehensive documentation & setup guide
├── .gitignore                  # Git tracking exclusions
│
├── models/
│   ├── __init__.py
│   ├── xception_model.py       # Transfer learning architecture definition
│   ├── train_model.py          # Training pipeline with callbacks & metrics
│   ├── predict.py              # Robust inference engine (image & video)
│   └── gradcam.py              # Visual explainability gradient generator
│
├── dataset/                    # Dataset directory structure
│   ├── train/                  # Training set (real / fake)
│   ├── validation/             # Validation set (real / fake)
│   └── test/                   # Test evaluation set (real / fake)
│
├── pages/                      # Multi-page Streamlit application modules
│   ├── dashboard.py            # Platform metrics & overview
│   ├── image_detection.py      # Single-image forensic inspection
│   ├── video_detection.py      # Video frame extraction & aggregation
│   ├── history.py              # Local persistent inspection log
│   ├── model_information.py    # Architecture & evaluation metrics
│   └── about.py                # Objectives & technical disclosures
│
├── utils/                      # Core functional utilities
│   ├── __init__.py
│   ├── face_detection.py       # OpenCV face cropping & bounding box logic
│   ├── image_processing.py     # Resizing, alignment, & normalization
│   ├── video_processing.py     # Frame extraction & aggregation algorithms
│   ├── report_generator.py     # Automated forensic PDF generator
│   └── metrics.py              # Confusion matrix, ROC, & metric utilities
│
├── assets/                     # Visual design assets
│   ├── logo.png                # High-res cybersecurity branding icon
│   └── styles.css              # Custom dark-navy glassmorphism stylesheet
│
├── saved_models/               # Model weights & evaluation artifacts
│   └── xception_deepfake.keras # Saved Keras model file
│
├── reports/                    # Generated forensic PDF exports
│
└── notebooks/
    └── deepfake_training.ipynb # Google Colab end-to-end training notebook
```

---

## 4. Installation & Local Setup

### Recommended Environment
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: **Python 3.10, 3.11, or 3.12** (*Note: TensorFlow does not yet support Python 3.14+*)

### Step 1: Open PowerShell and Navigate to Workspace
```powershell
cd d:\Project\guio
```

### Step 2: Create a Clean Python 3.12 Virtual Environment
```powershell
py -3.12 -m venv .venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 4: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Running the Application

Launch the local Streamlit application:
```powershell
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`.

---

## 6. Model Training Workflow

Training heavy deep learning models on local consumer laptops without high-end dedicated GPUs is discouraged. DEEPGUARD AI provides an end-to-end **Google Colab Notebook** (`notebooks/deepfake_training.ipynb`) configured for free cloud T4/V100 GPUs.

1. Upload `notebooks/deepfake_training.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Mount Google Drive or upload dataset.
4. Execute training cells; the notebook exports `xception_deepfake.keras` and metric summaries.
5. Place the downloaded `xception_deepfake.keras` into `saved_models/`.

---

## 7. Forensic Verification & Explainability

DEEPGUARD AI ensures transparent predictions:
- **No Mock Results**: Inference executes through the XceptionNet computational graph.
- **Grad-CAM Integration**: Class activation maps project gradients onto final convolutional feature maps (`block14_sepconv2_act`), showing where the network focused.
- **Inconclusive Handling**: When no human face is detected in the uploaded media, the system safely halts and prompts the investigator rather than hallucinating classifications.

---

## 8. Author & Developer Contact

**Akshat Kumar**  
*AI Developer & Product Builder*  
Bridging B.Tech Computer Science and MBA in AI & Data Science to ship production-grade machine intelligence and computer vision systems.

- 🌐 **GitHub Profile**: [@akshatk1997](https://github.com/akshatk1997)
- 💼 **LinkedIn Profile**: [linkedin.com/in/akshatk1997](https://www.linkedin.com/in/akshatk1997)
- ✉️ **Email**: [akshatk1997@gmail.com](mailto:akshatk1997@gmail.com)
- 📂 **Project Repository**: [github.com/akshatk1997/deepguard-ai](https://github.com/akshatk1997/deepguard-ai)

---

## 9. License & Disclaimers

This software is released under the **MIT License**. Deepfake detection models are probabilistic statistical classifiers; predictions should be evaluated alongside forensic chain-of-custody protocols in legal and critical applications.
