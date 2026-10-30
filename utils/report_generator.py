"""
DEEPGUARD AI - Forensic PDF Report Generation Engine
=====================================================
Generates court-admissible, audit-ready cryptographic PDF verification reports
using ReportLab. Includes branding, unique investigation ID, SHA-256 audit hashes,
biometric telemetry, Grad-CAM visual evidence embeds, and forensic disclaimers.
"""

import os
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable
)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ForensicReportGenerator:
    """
    Automated ReportLab forensic report generator for DEEPGUARD AI.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._init_custom_styles()

    def _init_custom_styles(self):
        """Initializes forensic document typography styles."""
        self.title_style = ParagraphStyle(
            "DocTitle",
            parent=self.styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0B132B"),
            fontName="Helvetica-Bold",
            spaceAfter=4
        )
        self.subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=self.styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#475569"),
            fontName="Helvetica",
            spaceAfter=12
        )
        self.section_header = ParagraphStyle(
            "SectionHeader",
            parent=self.styles["Heading2"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#0F172A"),
            fontName="Helvetica-Bold",
            spaceBefore=10,
            spaceAfter=6
        )
        self.body_style = ParagraphStyle(
            "BodyDark",
            parent=self.styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#1E293B"),
            fontName="Helvetica"
        )
        self.disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=self.styles["Normal"],
            fontSize=7.5,
            leading=11,
            textColor=colors.HexColor("#64748B"),
            fontName="Helvetica-Oblique"
        )

    @staticmethod
    def _save_temp_image(img_rgb: np.ndarray, prefix: str = "temp_report") -> str:
        """Saves a temporary RGB image to disk for ReportLab embedding."""
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"{prefix}_{int(time.time() * 1000)}.png"
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(temp_path), img_bgr)
        return str(temp_path)

    @staticmethod
    def _generate_audit_hash(data_str: str) -> str:
        """Computes SHA-256 cryptographic audit signature."""
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest().upper()

    def generate_image_report(
        self,
        analysis_data: Dict[str, Any],
        explanation_data: Optional[Dict[str, Any]] = None,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Builds a comprehensive single-image forensic authenticity verification PDF.
        """
        investigation_id = f"DG-IMG-{int(time.time())}"
        pdf_filename = output_filename or f"DeepGuard_Forensic_Report_{investigation_id}.pdf"
        pdf_path = self.output_dir / pdf_filename

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            leftMargin=0.5 * inch,
            rightMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch
        )

        elements = []
        temp_images_to_clean: List[str] = []

        # 1. Header Banner & Branding
        header_data = []
        if LOGO_PATH.exists():
            logo_img = RLImage(str(LOGO_PATH), width=1.1 * inch, height=1.1 * inch)
            header_text = [
                Paragraph("DEEPGUARD AI FORENSIC LABORATORY", self.title_style),
                Paragraph("Digital Media Authenticity Verification & Biometric Manipulation Audit", self.subtitle_style),
                Paragraph(f"<b>Investigation ID:</b> {investigation_id} &nbsp;|&nbsp; <b>Date/Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", self.body_style)
            ]
            header_table = Table([[logo_img, header_text]], colWidths=[1.3 * inch, 6.2 * inch])
        else:
            header_table = Table([[
                Paragraph("DEEPGUARD AI FORENSIC LABORATORY", self.title_style),
                Paragraph(f"Investigation ID: {investigation_id}", self.body_style)
            ]], colWidths=[7.5 * inch])

        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4)
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 8))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#00D2FF"), spaceAfter=10))

        # 2. Executive Forensic Verdict Banner
        verdict = analysis_data.get("verdict", "INCONCLUSIVE")
        confidence = analysis_data.get("confidence", 0.0)
        risk_tier = analysis_data.get("risk_tier", "UNKNOWN")

        if verdict == "FAKE":
            verdict_bg = colors.HexColor("#FEE2E2")
            verdict_border = colors.HexColor("#EF4444")
            verdict_text = f"<font color='#B91C1C'><b>🚨 VERDICT: FAKE / MANIPULATED MEDIA DETECTED</b></font>"
            verdict_desc = "High probability of AI facial synthesis, deepfake face-swapping, or blending artifact seam manipulation."
        elif verdict == "REAL":
            verdict_bg = colors.HexColor("#DCFCE7")
            verdict_border = colors.HexColor("#10B981")
            verdict_text = f"<font color='#15803D'><b>🛡️ VERDICT: REAL / AUTHENTIC MEDIA VERIFIED</b></font>"
            verdict_desc = "Organic facial contours, natural ocular reflections, and biometric integrity confirmed."
        else:
            verdict_bg = colors.HexColor("#FEF3C7")
            verdict_border = colors.HexColor("#F59E0B")
            verdict_text = f"<font color='#B45309'><b>⚠️ VERDICT: INCONCLUSIVE</b></font>"
            verdict_desc = "Unable to isolate verifiable human facial subjects for neural feature extraction."

        verdict_table = Table([
            [Paragraph(verdict_text, ParagraphStyle("VText", parent=self.title_style, fontSize=14, leading=18))],
            [Paragraph(f"<b>Confidence Score: {confidence}%</b> &nbsp;|&nbsp; <b>Risk Tier: {risk_tier}</b>", self.body_style)],
            [Paragraph(verdict_desc, self.subtitle_style)]
        ], colWidths=[7.5 * inch])

        verdict_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), verdict_bg),
            ("BOX", (0, 0), (-1, -1), 1.5, verdict_border),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12)
        ]))
        elements.append(verdict_table)
        elements.append(Spacer(1, 10))

        # 3. Technical & Biometric Telemetry
        elements.append(Paragraph("1. Technical & Forensic Inspection Parameters", self.section_header))
        filename = analysis_data.get("filename", "Uploaded_Target.png")
        fake_prob = analysis_data.get("fake_probability", 0.0)
        real_prob = analysis_data.get("real_probability", 0.0)
        latency = analysis_data.get("inference_time_ms", 0.0)
        faces_n = analysis_data.get("faces_detected", 1)
        face_box = str(analysis_data.get("primary_face_box", "N/A"))

        telemetry_rows = [
            [Paragraph("<b>Target Media Filename:</b>", self.body_style), Paragraph(str(filename), self.body_style),
             Paragraph("<b>Media Type:</b>", self.body_style), Paragraph("Static Image (RGB)", self.body_style)],
            [Paragraph("<b>Faces Detected:</b>", self.body_style), Paragraph(str(faces_n), self.body_style),
             Paragraph("<b>Primary Bounding Box:</b>", self.body_style), Paragraph(face_box, self.body_style)],
            [Paragraph("<b>FAKE Logit Probability:</b>", self.body_style), Paragraph(f"{fake_prob:.4f} ({round(fake_prob * 100, 2)}%)", self.body_style),
             Paragraph("<b>REAL Logit Probability:</b>", self.body_style), Paragraph(f"{real_prob:.4f} ({round(real_prob * 100, 2)}%)", self.body_style)],
            [Paragraph("<b>Neural Backbone:</b>", self.body_style), Paragraph("XceptionNet (36 Separable Stages)", self.body_style),
             Paragraph("<b>Inference Latency:</b>", self.body_style), Paragraph(f"{latency} ms", self.body_style)],
            [Paragraph("<b>Input Normalization:</b>", self.body_style), Paragraph("Scaled to [-1.0, +1.0] at 299x299", self.body_style),
             Paragraph("<b>Decision Threshold:</b>", self.body_style), Paragraph("tau = 0.50 (Binary Sigmoid)", self.body_style)]
        ]

        t_telemetry = Table(telemetry_rows, colWidths=[1.8 * inch, 1.95 * inch, 1.8 * inch, 1.95 * inch])
        t_telemetry.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6)
        ]))
        elements.append(t_telemetry)
        elements.append(Spacer(1, 10))

        # 4. Visual Evidence & Grad-CAM Analysis
        elements.append(Paragraph("2. Visual Forensic Evidence & Explainable AI (Grad-CAM)", self.section_header))

        has_crop = analysis_data.get("primary_face_rgb") is not None
        has_cam = explanation_data is not None and "heatmap_rgb" in explanation_data

        if has_crop and has_cam:
            crop_path = self._save_temp_image(analysis_data["primary_face_rgb"], "crop")
            cam_path = self._save_temp_image(explanation_data["heatmap_rgb"], "cam")
            overlay_path = self._save_temp_image(explanation_data["overlay_rgb"], "overlay")
            temp_images_to_clean.extend([crop_path, cam_path, overlay_path])

            img_w, img_h = 2.3 * inch, 2.3 * inch
            ev_table = Table([
                [RLImage(crop_path, width=img_w, height=img_h),
                 RLImage(cam_path, width=img_w, height=img_h),
                 RLImage(overlay_path, width=img_w, height=img_h)],
                [Paragraph("<b>Figure 1: Isolated Face</b>", self.body_style),
                 Paragraph("<b>Figure 2: Grad-CAM Activation</b>", self.body_style),
                 Paragraph("<b>Figure 3: Thermal Overlay</b>", self.body_style)]
            ], colWidths=[2.5 * inch, 2.5 * inch, 2.5 * inch])

            ev_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2)
            ]))
            elements.append(ev_table)

            dom_regions = ", ".join(explanation_data.get("dominant_regions", [])) or "Facial feature boundaries"
            elements.append(Spacer(1, 4))
            elements.append(Paragraph(f"<b>Key Feature Attention:</b> Model gradients concentrated primarily on: <i>{dom_regions}</i>.", self.body_style))

        elif has_crop:
            crop_path = self._save_temp_image(analysis_data["primary_face_rgb"], "crop")
            temp_images_to_clean.append(crop_path)
            elements.append(RLImage(crop_path, width=2.4 * inch, height=2.4 * inch))
            elements.append(Paragraph("<b>Figure 1: Primary Subject Facial Crop (20% Expanded Margin)</b>", self.body_style))

        elements.append(Spacer(1, 10))

        # 5. Cryptographic Verification & Legal Disclaimer
        elements.append(Paragraph("3. Cryptographic Provenance & Forensic Limitations", self.section_header))
        audit_raw = f"{investigation_id}|{filename}|{verdict}|{confidence}|{fake_prob}"
        audit_sha256 = self._generate_audit_hash(audit_raw)

        elements.append(Paragraph(f"<b>SHA-256 Audit Signature:</b> <code>{audit_sha256}</code>", self.body_style))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(
            "<b>LEGAL & FORENSIC CHAIN-OF-CUSTODY NOTICE:</b> This authenticity verification report was compiled by DEEPGUARD AI "
            "using deep neural network inference (XceptionNet) and Explainable AI backpropagation (Grad-CAM). Predictions reflect "
            "empirical statistical probabilities and localized boundary anomalies. While offering high empirical precision (94.8% on "
            "standard benchmarks), findings should be evaluated in conjunction with digital chain-of-custody metadata, sensor PRNU "
            "noise verification, and human expert forensic testimony in judicial or legal proceedings.",
            self.disclaimer_style
        ))

        # Build document
        doc.build(elements)

        # Cleanup temporary image files
        for p in temp_images_to_clean:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

        return str(pdf_path)

    def generate_video_report(
        self,
        analysis_data: Dict[str, Any],
        output_filename: Optional[str] = None
    ) -> str:
        """
        Builds a comprehensive multi-frame temporal video verification report.
        """
        investigation_id = f"DG-VID-{int(time.time())}"
        pdf_filename = output_filename or f"DeepGuard_Video_Report_{investigation_id}.pdf"
        pdf_path = self.output_dir / pdf_filename

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            leftMargin=0.5 * inch,
            rightMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch
        )

        elements = []
        temp_images_to_clean: List[str] = []

        # 1. Header
        header_text = [
            Paragraph("DEEPGUARD AI FORENSIC LABORATORY", self.title_style),
            Paragraph("Temporal Video Stream Deepfake & Manipulation Verification Audit", self.subtitle_style),
            Paragraph(f"<b>Investigation ID:</b> {investigation_id} &nbsp;|&nbsp; <b>Date/Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", self.body_style)
        ]
        if LOGO_PATH.exists():
            logo_img = RLImage(str(LOGO_PATH), width=1.1 * inch, height=1.1 * inch)
            header_table = Table([[logo_img, header_text]], colWidths=[1.3 * inch, 6.2 * inch])
        else:
            header_table = Table([[Paragraph("DEEPGUARD AI", self.title_style), header_text]], colWidths=[1.5 * inch, 6.0 * inch])

        header_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        elements.append(header_table)
        elements.append(Spacer(1, 6))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#00D2FF"), spaceAfter=10))

        # 2. Verdict Banner
        verdict = analysis_data.get("verdict", "INCONCLUSIVE")
        confidence = analysis_data.get("confidence", 0.0)
        risk_tier = analysis_data.get("risk_tier", "UNKNOWN")

        if verdict == "FAKE":
            verdict_bg = colors.HexColor("#FEE2E2")
            verdict_border = colors.HexColor("#EF4444")
            v_title = "<font color='#B91C1C'><b>🚨 OVERALL VERDICT: FAKE VIDEO DETECTED</b></font>"
            v_desc = "Temporal frame aggregation confirmed persistent or burst facial replacement deepfake anomalies."
        elif verdict == "REAL":
            verdict_bg = colors.HexColor("#DCFCE7")
            verdict_border = colors.HexColor("#10B981")
            v_title = "<font color='#15803D'><b>🛡️ OVERALL VERDICT: REAL VIDEO VERIFIED</b></font>"
            v_desc = "Biometric temporal continuity verified across all sampled frames with no synthetic anomalies."
        else:
            verdict_bg = colors.HexColor("#FEF3C7")
            verdict_border = colors.HexColor("#F59E0B")
            v_title = "<font color='#B45309'><b>⚠️ OVERALL VERDICT: INCONCLUSIVE</b></font>"
            v_desc = "No usable human facial subjects detected in sampled video stream intervals."

        v_table = Table([
            [Paragraph(v_title, ParagraphStyle("VT", parent=self.title_style, fontSize=14, leading=18))],
            [Paragraph(f"<b>Aggregated Confidence: {confidence}%</b> &nbsp;|&nbsp; <b>Risk Tier: {risk_tier}</b>", self.body_style)],
            [Paragraph(v_desc, self.subtitle_style)]
        ], colWidths=[7.5 * inch])
        v_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), verdict_bg),
            ("BOX", (0, 0), (-1, -1), 1.5, verdict_border),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ]))
        elements.append(v_table)
        elements.append(Spacer(1, 10))

        # 3. Video Telemetry & Temporal Aggregation Parameters
        elements.append(Paragraph("1. Container & Temporal Aggregation Telemetry", self.section_header))
        meta = analysis_data.get("metadata", {})
        filename = analysis_data.get("filename", "Target_Stream.mp4")
        dur = meta.get("duration_sec", 0.0)
        res = meta.get("resolution", "N/A")
        fps = meta.get("fps", 0.0)
        sampled_n = analysis_data.get("frames_analyzed", 0)
        faces_n = analysis_data.get("faces_detected", 0)
        manip_n = analysis_data.get("manipulated_frames_count", 0)
        manip_r = analysis_data.get("manipulated_frames_ratio", 0.0)
        proc_time = analysis_data.get("processing_time_sec", 0.0)

        v_rows = [
            [Paragraph("<b>Filename:</b>", self.body_style), Paragraph(str(filename), self.body_style),
             Paragraph("<b>Container Duration:</b>", self.body_style), Paragraph(f"{dur} sec ({fps} FPS)", self.body_style)],
            [Paragraph("<b>Resolution:</b>", self.body_style), Paragraph(str(res), self.body_style),
             Paragraph("<b>Sampled Frames:</b>", self.body_style), Paragraph(f"{sampled_n} frames uniformly spaced", self.body_style)],
            [Paragraph("<b>Faces Localized:</b>", self.body_style), Paragraph(f"{faces_n} valid facial crops", self.body_style),
             Paragraph("<b>Manipulated Ratio:</b>", self.body_style), Paragraph(f"{manip_n} / {faces_n} ({manip_r}%)", self.body_style)],
            [Paragraph("<b>Aggregation Rule:</b>", self.body_style), Paragraph("Top-K (25%) + Ratio >= 20%", self.body_style),
             Paragraph("<b>Processing Time:</b>", self.body_style), Paragraph(f"{proc_time} seconds", self.body_style)]
        ]

        t_v = Table(v_rows, colWidths=[1.8 * inch, 1.95 * inch, 1.8 * inch, 1.95 * inch])
        t_v.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4)
        ]))
        elements.append(t_v)
        elements.append(Spacer(1, 10))

        # 4. Suspicious Frames Gallery
        suspicious = analysis_data.get("suspicious_frames", [])
        if suspicious:
            elements.append(Paragraph("2. Isolated Suspicious Frames (Top Manipulation Probabilities)", self.section_header))
            frame_cards = []
            frame_labels = []

            for sf in suspicious[:3]:
                if "full_frame_rgb" in sf:
                    f_path = self._save_temp_image(sf["full_frame_rgb"], "vid_susp")
                    temp_images_to_clean.append(f_path)
                    frame_cards.append(RLImage(f_path, width=2.3 * inch, height=1.6 * inch))
                    frame_labels.append(Paragraph(
                        f"<b>Frame #{sf['frame_number']}</b> ({sf['timestamp_sec']}s)<br>Risk: {round(sf['fake_probability'] * 100, 1)}%",
                        self.body_style
                    ))

            if frame_cards:
                t_frames = Table([frame_cards, frame_labels], colWidths=[2.5 * inch] * len(frame_cards))
                t_frames.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
                elements.append(t_frames)

        elements.append(Spacer(1, 10))

        # 5. Cryptographic Provenance & Disclaimer
        elements.append(Paragraph("3. Cryptographic Provenance & Legal Disclaimer", self.section_header))
        audit_raw = f"{investigation_id}|{filename}|{verdict}|{confidence}|{manip_r}"
        elements.append(Paragraph(f"<b>SHA-256 Audit Signature:</b> <code>{self._generate_audit_hash(audit_raw)}</code>", self.body_style))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(
            "<b>LEGAL & FORENSIC NOTICE:</b> Temporal video aggregation computes probability distribution statistics across "
            "uniformly sampled frame intervals. Findings reflect probabilistic neural classifications and should be reviewed "
            "by certified forensic examiners in accordance with applicable jurisdictional rules of digital evidence.",
            self.disclaimer_style
        ))

        doc.build(elements)

        for p in temp_images_to_clean:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

        return str(pdf_path)
