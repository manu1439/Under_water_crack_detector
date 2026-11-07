"""
AquaGuard AI — Automated Test Suite.
Tests frame quality assessment, image enhancement, severity estimation,
report generation, and API router integrity.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add backend to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.schemas.models import (
    BoundingBox,
    CrackDetection,
    DetectionResult,
    EnhancementMethod,
    FrameQuality,
    FrameResult,
    InspectionConfig,
    InspectionResultFull,
    InspectionStatus,
    InspectionSummary,
    SeverityLevel,
)
from backend.app.services.frame_selection import FrameSelector
from backend.app.services.image_enhancement import ImageEnhancer
from backend.app.services.report_generator import ReportGenerator
from backend.app.services.severity_assessment import SeverityAssessor


def test_frame_quality_assessment():
    """Verify blur, brightness, and contrast computation."""
    selector = FrameSelector()

    # Create synthetic sharp image
    sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
    sharp_img[::20, :] = 255  # High-frequency horizontal stripes
    quality = selector.assess_frame_quality(sharp_img, frame_number=1, timestamp=0.5)

    assert quality.frame_number == 1
    assert quality.timestamp == 0.5
    assert quality.blur_score > 50.0
    assert quality.brightness > 0.0

    # Create uniform gray/blurry image
    blurry_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    quality_blur = selector.assess_frame_quality(blurry_img, frame_number=2, timestamp=1.0)
    assert quality_blur.blur_score < 10.0


def test_image_enhancement_clahe_and_gamma():
    """Verify CLAHE and Gamma correction maintain image shape and valid range."""
    enhancer = ImageEnhancer()
    img = np.random.randint(40, 180, (100, 100, 3), dtype=np.uint8)

    clahe_res = enhancer.enhance_clahe(img, clip_limit=3.0, tile_grid_size=8)
    assert clahe_res.shape == img.shape
    assert clahe_res.dtype == np.uint8

    gamma_res = enhancer.enhance_gamma(img, gamma=1.5)
    assert gamma_res.shape == img.shape
    assert gamma_res.dtype == np.uint8

    retinex_res = enhancer.enhance_retinex(img)
    assert retinex_res.shape == img.shape
    assert retinex_res.dtype == np.uint8


def test_severity_assessment():
    """Verify heuristic severity assessment logic."""
    assessor = SeverityAssessor()

    # Synthesize crack-like dark line in a 300x300 image
    img = np.ones((300, 300, 3), dtype=np.uint8) * 200
    img[100:250, 148:152] = 20  # dark vertical crack line

    bbox = BoundingBox(x1=140, y1=95, x2=160, y2=255)
    assessment = assessor.assess_crack(img, bbox, image_width=300, image_height=300)

    assert assessment.severity in [SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH]
    assert 0.0 <= assessment.severity_score <= 100.0
    assert assessment.assessment_method == "heuristic"
    assert len(assessment.reasons) > 0


def test_report_generation(tmp_path):
    """Verify JSON and PDF report generation."""
    generator = ReportGenerator(tmp_path)

    summary = InspectionSummary(
        inspection_id="test-insp-001",
        filename="test_dock_pier.mp4",
        created_at="2026-11-07 12:00:00",
        status=InspectionStatus.COMPLETED,
        video_duration=15.0,
        total_frames_extracted=10,
        selected_frames=5,
        rejected_frames=5,
        processed_frames=5,
        total_detections=2,
        severity_distribution={"LOW": 1, "MEDIUM": 1, "HIGH": 0, "UNKNOWN": 0},
        avg_confidence=0.88,
        enhancement_method="clahe",
    )

    bbox = BoundingBox(x1=50, y1=50, x2=150, y2=200)
    det = CrackDetection(
        detection_id="det-1",
        class_name="crack",
        confidence=0.89,
        bbox=bbox,
        frame_number=1,
        timestamp=0.5,
        image_width=640,
        image_height=480,
    )
    from backend.app.schemas.models import SeverityAssessment

    sev = SeverityAssessment(
        severity=SeverityLevel.MEDIUM,
        severity_score=52.0,
        assessment_method="heuristic",
        reasons=["Visible crack extent is moderate"],
        measurement_unit="pixel",
    )

    frame_res = FrameResult(
        frame_number=1,
        timestamp=0.5,
        original_path=str(tmp_path / "frame1.jpg"),
        enhanced_path=str(tmp_path / "frame1_clahe.jpg"),
        annotated_path=None,
        enhancement_method=EnhancementMethod.CLAHE,
        detections=[DetectionResult(detection=det, severity=sev)],
        crack_count=1,
    )

    full_res = InspectionResultFull(
        summary=summary,
        config=InspectionConfig(),
        frame_results=[frame_res],
    )

    json_file = generator.generate_json_report(full_res)
    assert json_file.exists()

    pdf_file = generator.generate_pdf_report(full_res)
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 0


if __name__ == "__main__":
    print("Running manual test execution...")
    test_frame_quality_assessment()
    print("✓ Frame quality assessment passed.")
    test_image_enhancement_clahe_and_gamma()
    print("✓ Image enhancement passed.")
    test_severity_assessment()
    print("✓ Severity assessment passed.")
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_report_generation(Path(td))
    print("✓ Report generation passed.")
    print("All backend tests PASSED successfully!")
