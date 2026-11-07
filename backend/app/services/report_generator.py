"""
AquaGuard AI — Inspection Results and Report Generator.
Generates comprehensive inspection reports in both JSON and PDF formats.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fpdf import FPDF

from backend.app.schemas.models import (
    InspectionConfig,
    InspectionResultFull,
    InspectionSummary,
)

logger = logging.getLogger(__name__)


class AquaGuardPDF(FPDF):
    """Custom PDF template with dark-mode header / footer styling."""

    def __init__(self, inspection_id: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.inspection_id = inspection_id
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        # Header banner
        self.set_fill_color(15, 23, 42)  # Navy dark
        self.rect(0, 0, 210, 20, "F")
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 5)
        self.cell(0, 10, "AquaGuard AI - Underwater Inspection Report", 0, new_x="RIGHT", new_y="TOP", align="L")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(148, 163, 184)
        self.set_xy(10, 5)
        self.cell(0, 10, f"ID: {self.inspection_id[:12]}...", 0, new_x="RIGHT", new_y="TOP", align="R")
        self.ln(18)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | AquaGuard AI - Confidential Inspection Data", 0, new_x="RIGHT", new_y="TOP", align="C")


class ReportGenerator:
    """Generates JSON and PDF inspection reports."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_json_report(self, result: InspectionResultFull) -> Path:
        """Export full inspection results as formatted JSON."""
        file_path = self.output_dir / "report.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        logger.info(f"JSON report saved: {file_path}")
        return file_path

    def generate_csv_report(self, result: InspectionResultFull) -> Path:
        """Export inspection detections as a clean CSV spreadsheet."""
        file_path = self.output_dir / "report.csv"
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Metadata header
            writer.writerow(["# AquaGuard AI - Inspection Telemetry Export"])
            writer.writerow(["# Inspection ID", result.summary.inspection_id])
            writer.writerow(["# Target File", result.summary.filename])
            writer.writerow(["# Total Detections", result.summary.total_detections])
            writer.writerow([])
            # Column headers
            writer.writerow([
                "Frame Number",
                "Timestamp (s)",
                "Crack Index",
                "Confidence (%)",
                "Severity Level",
                "Severity Score",
                "Width (px)",
                "Height (px)",
                "Bounding Box [x1, y1, x2, y2]",
                "Physical Diagnostic Reasons"
            ])
            for fr in result.frame_results:
                for idx, dr in enumerate(fr.detections, 1):
                    bbox = dr.detection.bbox
                    w = round(abs(bbox.x2 - bbox.x1), 1)
                    h = round(abs(bbox.y2 - bbox.y1), 1)
                    reasons = " | ".join(dr.severity.reasons)
                    writer.writerow([
                        fr.frame_number,
                        round(fr.timestamp, 2),
                        idx,
                        round(dr.detection.confidence * 100, 1),
                        dr.severity.severity.value,
                        round(dr.severity.severity_score, 1),
                        w,
                        h,
                        f"[{bbox.x1:.1f}, {bbox.y1:.1f}, {bbox.x2:.1f}, {bbox.y2:.1f}]",
                        reasons
                    ])
        logger.info(f"CSV report saved: {file_path}")
        return file_path

    def generate_pdf_report(
        self,
        result: InspectionResultFull,
        annotated_images: Optional[list[Path]] = None,
    ) -> Path:
        """Export inspection report as a professional multi-page PDF."""
        pdf_path = self.output_dir / "inspection_report.pdf"

        pdf = AquaGuardPDF(inspection_id=result.summary.inspection_id)
        pdf.alias_nb_pages()
        pdf.add_page()

        summary: InspectionSummary = result.summary
        config: InspectionConfig = result.config

        # ── Document Title ────────────────────────────────────────────
        pdf.set_text_color(15, 23, 42)
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 10, "Infrastructure Inspection Summary", 0, new_x="LMARGIN", new_y="NEXT", align="L")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | Target: {summary.filename}", 0, new_x="LMARGIN", new_y="NEXT", align="L")
        pdf.ln(5)

        # ── Summary Metadata Box ──────────────────────────────────────
        pdf.set_fill_color(241, 245, 249)
        pdf.rect(10, pdf.get_y(), 190, 42, "F")
        pdf.set_xy(14, pdf.get_y() + 3)

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(90, 6, "INSPECTION METADATA", 0, new_x="RIGHT", new_y="TOP")
        pdf.cell(90, 6, "PROCESSING STATISTICS", 0, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(71, 85, 105)
        y = pdf.get_y()

        # Col 1
        pdf.set_xy(14, y)
        pdf.cell(90, 5, f"Inspection ID: {summary.inspection_id}", 0, new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(14)
        pdf.cell(90, 5, f"Video Duration: {summary.video_duration:.1f} s", 0, new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(14)
        pdf.cell(90, 5, f"Enhancement Method: {summary.enhancement_method.upper()}", 0, new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(14)
        pdf.cell(90, 5, f"Confidence Threshold: {config.confidence_threshold:.2f}", 0, new_x="LMARGIN", new_y="NEXT")

        # Col 2
        pdf.set_xy(105, y)
        pdf.cell(90, 5, f"Total Extracted Frames: {summary.total_frames_extracted}", 0, new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(105)
        pdf.cell(90, 5, f"Selected / Rejected: {summary.selected_frames} / {summary.rejected_frames}", 0, new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(105)
        pdf.cell(90, 5, f"Total Crack Detections: {summary.total_detections}", 0, new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(105)
        pdf.cell(90, 5, f"Average Confidence: {summary.avg_confidence * 100:.1f}%", 0, new_x="LMARGIN", new_y="NEXT")

        pdf.set_y(y + 25)
        pdf.ln(5)

        # ── Severity Assessment Breakdown ─────────────────────────────
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, "Defect Severity Breakdown", 0, new_x="LMARGIN", new_y="NEXT", align="L")

        dist = summary.severity_distribution
        col_w = 45
        headers = ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]
        colors = [(239, 68, 68), (249, 115, 22), (34, 197, 94), (156, 163, 175)]

        # Header row
        for i, h in enumerate(headers):
            pdf.set_fill_color(*colors[i])
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(col_w, 7, f"{h} SEVERITY", 0, new_x="RIGHT", new_y="TOP", align="C", fill=True)
            if i < len(headers) - 1:
                pdf.cell(3, 7, "", 0, new_x="RIGHT", new_y="TOP")
        pdf.ln(8)

        # Count row
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 41, 59)
        for i, h in enumerate(headers):
            count = dist.get(h, 0)
            pdf.set_fill_color(248, 250, 252)
            pdf.cell(col_w, 10, str(count), 1, new_x="RIGHT", new_y="TOP", align="C", fill=True)
            if i < len(headers) - 1:
                pdf.cell(3, 10, "", 0, new_x="RIGHT", new_y="TOP")
        pdf.ln(15)

        # ── Important Disclaimer ──────────────────────────────────────
        pdf.set_fill_color(254, 242, 242)
        pdf.set_draw_color(239, 68, 68)
        pdf.rect(10, pdf.get_y(), 190, 14, "DF")
        pdf.set_xy(14, pdf.get_y() + 2)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(185, 28, 28)
        pdf.cell(0, 4, "IMPORTANT NOTICE - PROVISIONAL VISUAL ESTIMATION", 0, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(127, 29, 29)
        pdf.cell(0, 4, "Severity ratings are heuristic visual estimates and do not represent a certified structural safety assessment.", 0, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(8)

        # ── Detailed Detections Table ─────────────────────────────────
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, "Frame-Level Crack Detections", 0, new_x="LMARGIN", new_y="NEXT", align="L")

        # Table header
        pdf.set_fill_color(30, 41, 59)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)
        cols = [("Frame", 15), ("Timestamp", 20), ("Confidence", 22), ("Severity", 22), ("Score", 15), ("Dimensions (px)", 35), ("Reasons / Explanation", 61)]
        for name, width in cols:
            pdf.cell(width, 7, name, 0, new_x="RIGHT", new_y="TOP", align="C", fill=True)
        pdf.ln(7)

        # Table rows
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(51, 65, 85)
        row_alt = False

        detection_rows = []
        for fr in result.frame_results:
            for det in fr.detections:
                detection_rows.append((fr, det))

        if not detection_rows:
            pdf.set_fill_color(248, 250, 252)
            pdf.cell(190, 8, "No crack defects detected in processed frames.", 1, new_x="LMARGIN", new_y="NEXT", align="C", fill=True)
        else:
            for fr, dr in detection_rows[:50]:  # Cap at first 50 to keep reasonable length
                fill_color = (248, 250, 252) if row_alt else (255, 255, 255)
                pdf.set_fill_color(*fill_color)

                bbox = dr.detection.bbox
                w_px = round(bbox.x2 - bbox.x1)
                h_px = round(bbox.y2 - bbox.y1)
                reasons_str = "; ".join(dr.severity.reasons[:2]) if dr.severity.reasons else "Standard detection"
                if len(reasons_str) > 40:
                    reasons_str = reasons_str[:37] + "..."

                pdf.cell(15, 6, str(fr.frame_number), 1, new_x="RIGHT", new_y="TOP", align="C", fill=True)
                pdf.cell(20, 6, f"{fr.timestamp:.2f}s", 1, new_x="RIGHT", new_y="TOP", align="C", fill=True)
                pdf.cell(22, 6, f"{dr.detection.confidence * 100:.1f}%", 1, new_x="RIGHT", new_y="TOP", align="C", fill=True)
                pdf.cell(22, 6, dr.severity.severity.value, 1, new_x="RIGHT", new_y="TOP", align="C", fill=True)
                pdf.cell(15, 6, f"{dr.severity.severity_score:.0f}", 1, new_x="RIGHT", new_y="TOP", align="C", fill=True)
                pdf.cell(35, 6, f"{w_px} x {h_px}", 1, new_x="RIGHT", new_y="TOP", align="C", fill=True)
                pdf.cell(61, 6, reasons_str, 1, new_x="LMARGIN", new_y="NEXT", align="L", fill=True)

                row_alt = not row_alt

        # ── Annotated Keyframes (if provided) ─────────────────────────
        if annotated_images:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 10, "Annotated Defect Keyframes", 0, new_x="LMARGIN", new_y="NEXT", align="L")
            pdf.ln(2)

            for i, img_path in enumerate(annotated_images[:4]):  # Up to 4 images
                if Path(img_path).exists():
                    try:
                        pdf.image(str(img_path), x=15, y=pdf.get_y(), w=180)
                        pdf.ln(105)
                        pdf.set_font("Helvetica", "I", 8)
                        pdf.set_text_color(100, 116, 139)
                        pdf.cell(0, 5, f"Figure {i + 1}: Frame {Path(img_path).stem}", 0, new_x="LMARGIN", new_y="NEXT", align="C")
                        pdf.ln(5)
                        if pdf.get_y() > 240 and i < len(annotated_images) - 1:
                            pdf.add_page()
                    except Exception as e:
                        logger.warning(f"Could not add image {img_path} to PDF: {e}")

        pdf.output(str(pdf_path))
        logger.info(f"PDF report generated: {pdf_path}")
        return pdf_path
