"""
AquaGuard AI — Pydantic schemas for API request/response models.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────

class SeverityLevel(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class EnhancementMethod(str, enum.Enum):
    ORIGINAL = "original"
    CLAHE = "clahe"
    GAMMA = "gamma"
    RETINEX = "retinex"


class InspectionStatus(str, enum.Enum):
    PENDING = "pending"
    EXTRACTING_FRAMES = "extracting_frames"
    SELECTING_FRAMES = "selecting_frames"
    ENHANCING = "enhancing"
    DETECTING = "detecting"
    ASSESSING_SEVERITY = "assessing_severity"
    GENERATING_VIDEO = "generating_video"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Bounding Box ──────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


# ── Detection ─────────────────────────────────────────────────────────────

class CrackDetection(BaseModel):
    detection_id: str
    class_name: str = "crack"
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    frame_number: int
    timestamp: float = 0.0
    image_width: int
    image_height: int


# ── Severity ──────────────────────────────────────────────────────────────

class SeverityAssessment(BaseModel):
    severity: SeverityLevel = SeverityLevel.UNKNOWN
    severity_score: float = Field(0.0, ge=0.0, le=100.0)
    assessment_method: str = "heuristic"
    reasons: list[str] = Field(default_factory=list)
    measurement_unit: str = "pixel"


# ── Detection + Severity (combined) ──────────────────────────────────────

class DetectionResult(BaseModel):
    detection: CrackDetection
    severity: SeverityAssessment


# ── Frame Quality ─────────────────────────────────────────────────────────

class FrameQuality(BaseModel):
    frame_number: int
    timestamp: float
    blur_score: float
    brightness: float
    contrast: float
    quality_score: float
    is_selected: bool
    rejection_reasons: list[str] = Field(default_factory=list)


# ── Frame Result ──────────────────────────────────────────────────────────

class FrameResult(BaseModel):
    frame_number: int
    timestamp: float
    original_path: str
    enhanced_path: Optional[str] = None
    annotated_path: Optional[str] = None
    enhancement_method: EnhancementMethod = EnhancementMethod.ORIGINAL
    detections: list[DetectionResult] = Field(default_factory=list)
    crack_count: int = 0


# ── Inspection ────────────────────────────────────────────────────────────

class InspectionConfig(BaseModel):
    frame_interval: int = 30
    enhancement_method: EnhancementMethod = EnhancementMethod.CLAHE
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    process_all_frames: bool = False
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: int = 8
    gamma_value: float = 1.5
    inference_img_size: int = 640


class InspectionProgress(BaseModel):
    inspection_id: str
    status: InspectionStatus = InspectionStatus.PENDING
    progress_percent: float = 0.0
    total_frames: int = 0
    extracted_frames: int = 0
    selected_frames: int = 0
    processed_frames: int = 0
    detection_count: int = 0
    current_stage: str = ""
    error_message: Optional[str] = None


class InspectionSummary(BaseModel):
    inspection_id: str
    filename: str
    created_at: str
    status: InspectionStatus
    video_duration: float = 0.0
    total_frames_extracted: int = 0
    selected_frames: int = 0
    rejected_frames: int = 0
    processed_frames: int = 0
    total_detections: int = 0
    severity_distribution: dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0
    enhancement_method: str = "clahe"


class InspectionResultFull(BaseModel):
    summary: InspectionSummary
    config: InspectionConfig
    frame_results: list[FrameResult] = Field(default_factory=list)
    frame_qualities: list[FrameQuality] = Field(default_factory=list)
    annotated_video_path: Optional[str] = None
    report_json_path: Optional[str] = None
    report_pdf_path: Optional[str] = None


# ── Model Status ──────────────────────────────────────────────────────────

class ModelStatus(BaseModel):
    model_loaded: bool = False
    model_path: Optional[str] = None
    model_exists: bool = False
    model_size_mb: float = 0.0
    base_model: str = "yolo26n.pt"
    device: str = "cpu"
    classes: list[str] = Field(default_factory=list)
    inference_img_size: int = 640


class TrainingConfig(BaseModel):
    base_model: str = "yolo26n.pt"
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    workers: int = 4
    device: str = "auto"
    resume: bool = False


class TrainingStatus(BaseModel):
    is_training: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    progress_percent: float = 0.0
    metrics: dict = Field(default_factory=dict)
    error: Optional[str] = None


class DatasetInfo(BaseModel):
    exists: bool = False
    train_images: int = 0
    train_labels: int = 0
    valid_images: int = 0
    valid_labels: int = 0
    test_images: int = 0
    test_labels: int = 0
    classes: list[str] = Field(default_factory=list)
    num_classes: int = 0
    data_yaml_path: Optional[str] = None


class ValidationMetrics(BaseModel):
    precision: float = 0.0
    recall: float = 0.0
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    num_images: int = 0
    num_detections: int = 0


# ── API Responses ─────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    inspection_id: str
    filename: str
    file_type: str
    file_size_mb: float
    message: str


class ProcessResponse(BaseModel):
    inspection_id: str
    message: str
    status: InspectionStatus


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
