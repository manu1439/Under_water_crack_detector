"""
AquaGuard AI — End-to-end video processing pipeline.

Orchestrates: frame extraction → quality filtering → enhancement →
              crack detection → severity assessment → annotated video → report.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from backend.app.config import OUTPUT_DIR
from backend.app.schemas.models import (
    CrackDetection,
    DetectionResult,
    EnhancementMethod,
    FrameQuality,
    FrameResult,
    InspectionConfig,
    InspectionProgress,
    InspectionResultFull,
    InspectionStatus,
    InspectionSummary,
    SeverityAssessment,
    SeverityLevel,
)
from backend.app.services.crack_detection import CrackDetector
from backend.app.services.frame_selection import FrameSelector
from backend.app.services.image_enhancement import ImageEnhancer
from backend.app.services.severity_assessment import SeverityAssessor

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound work
_executor = ThreadPoolExecutor(max_workers=2)


class VideoPipeline:
    """Complete video inspection pipeline."""

    def __init__(self, config: InspectionConfig | None = None):
        self.config = config or InspectionConfig()
        self.frame_selector = FrameSelector()
        self.enhancer = ImageEnhancer()
        self.detector = CrackDetector(
            confidence=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            img_size=self.config.inference_img_size,
        )
        self.severity_assessor = SeverityAssessor()

        # Progress tracking
        self.progress = InspectionProgress(inspection_id="")
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable[[InspectionProgress], None]):
        """Set a callback invoked whenever progress updates."""
        self._progress_callback = callback

    def _update_progress(
        self,
        status: InspectionStatus,
        progress: float = 0.0,
        stage: str = "",
        **kwargs,
    ):
        """Update and broadcast progress."""
        self.progress.status = status
        self.progress.progress_percent = round(progress, 1)
        self.progress.current_stage = stage
        for k, v in kwargs.items():
            if hasattr(self.progress, k):
                setattr(self.progress, k, v)
        if self._progress_callback:
            try:
                self._progress_callback(self.progress)
            except Exception:
                pass

    # ── Main pipeline ─────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str | Path,
        inspection_id: str,
        output_base: str | Path | None = None,
    ) -> InspectionResultFull:
        """
        Run the complete inspection pipeline on a video.
        This is a synchronous, potentially long-running method.
        Call via run_pipeline_async() for non-blocking operation.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_base = Path(output_base) if output_base else OUTPUT_DIR / inspection_id
        output_base.mkdir(parents=True, exist_ok=True)

        self.progress.inspection_id = inspection_id
        start_time = time.time()

        # Directories
        extracted_dir = output_base / "extracted"
        selected_dir = output_base / "selected"
        enhanced_dir = output_base / "enhanced"
        annotated_dir = output_base / "annotated"
        for d in [extracted_dir, selected_dir, enhanced_dir, annotated_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Video validation ──────────────────────────────────
        self._update_progress(InspectionStatus.EXTRACTING_FRAMES, 5, "Validating video")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_video_frames / fps if fps > 0 else 0.0
        cap.release()

        logger.info(
            f"Video: {video_path.name} | FPS={fps:.1f} | "
            f"Frames={total_video_frames} | Duration={duration:.1f}s"
        )

        # ── Step 2: Frame extraction ──────────────────────────────────
        self._update_progress(
            InspectionStatus.EXTRACTING_FRAMES,
            10,
            "Extracting frames",
            total_frames=total_video_frames,
        )
        extracted_info = self.frame_selector.extract_frames(
            video_path, extracted_dir, interval=self.config.frame_interval
        )
        num_extracted = len(extracted_info)
        self._update_progress(
            InspectionStatus.SELECTING_FRAMES,
            20,
            "Selecting best frames",
            extracted_frames=num_extracted,
        )
        logger.info(f"Extracted {num_extracted} frames")

        # ── Step 3: Frame quality assessment + selection ──────────────
        if self.config.process_all_frames:
            # Use all extracted frames
            selected_info = extracted_info
            frame_qualities = []
            rejected_count = 0
        else:
            selection_res = self.frame_selector.select_best_frames(
                extracted_dir, selected_dir
            )
            selected_info = selection_res.get("selected", [])
            rejected_info = selection_res.get("rejected", [])
            frame_qualities = []
            rejected_count = len(rejected_info)

            # Fallback if turbidity caused all frames to be rejected
            if not selected_info and extracted_info:
                logger.info("Quality filter rejected all frames due to high turbidity; falling back to extracted frames.")
                selected_info = extracted_info

        num_selected = len(selected_info)
        self._update_progress(
            InspectionStatus.ENHANCING,
            30,
            "Enhancing frames",
            selected_frames=num_selected,
        )
        logger.info(f"Selected {num_selected} frames, rejected {rejected_count}")

        # Build list of frames to process
        if self.config.process_all_frames:
            frames_to_process = [
                {"path": Path(f["path"]), "frame_number": f["frame_number"], "timestamp": f["timestamp"]}
                for f in extracted_info
            ]
        else:
            frames_to_process = [
                {"path": Path(f["path"]), "frame_number": f["frame_number"], "timestamp": f["timestamp"]}
                for f in selected_info
            ]

        # ── Step 4-6: Enhancement + Detection + Severity ──────────────
        # Load model once
        model_loaded = self.detector.load_model()
        if not model_loaded:
            logger.warning("No trained model available — skipping crack detection")

        frame_results: list[FrameResult] = []
        all_detections: list[DetectionResult] = []
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "UNKNOWN": 0}
        total_confidence = 0.0
        total_det_count = 0

        for idx, frame_info in enumerate(frames_to_process):
            frame_path = frame_info["path"]
            frame_num = frame_info["frame_number"]
            timestamp = frame_info["timestamp"]

            progress_pct = 30 + (idx / max(len(frames_to_process), 1)) * 50
            self._update_progress(
                InspectionStatus.DETECTING,
                progress_pct,
                f"Processing frame {idx + 1}/{len(frames_to_process)}",
                processed_frames=idx,
            )

            # Read image
            image = cv2.imread(str(frame_path))
            if image is None:
                logger.warning(f"Could not read frame: {frame_path}")
                continue

            # Enhance
            method = self.config.enhancement_method.value
            if method != "original":
                enhanced = self.enhancer.enhance(
                    image,
                    method=method,
                    clip_limit=self.config.clahe_clip_limit,
                    tile_grid_size=self.config.clahe_tile_grid,
                    gamma=self.config.gamma_value,
                )
                enhanced_path = enhanced_dir / frame_path.name
                cv2.imwrite(str(enhanced_path), enhanced)
                process_image = enhanced
            else:
                enhanced_path = None
                process_image = image

            # Detect cracks
            frame_detections: list[DetectionResult] = []
            annotated_path = None

            if model_loaded:
                detections = self.detector.detect_image(
                    process_image, frame_number=frame_num, timestamp=timestamp
                )

                # Dual detection: also test raw frame when enhanced is active to capture cracks best visible in raw
                if method != "original":
                    orig_detections = self.detector.detect_image(
                        image, frame_number=frame_num, timestamp=timestamp
                    )
                    detections = self._merge_detections(detections, orig_detections)

                if detections:
                    # Assess severity for each detection
                    severities = self.severity_assessor.assess_multiple(
                        process_image, detections
                    )

                    for det, sev in zip(detections, severities):
                        dr = DetectionResult(detection=det, severity=sev)
                        frame_detections.append(dr)
                        all_detections.append(dr)
                        severity_counts[sev.severity.value] += 1
                        total_confidence += det.confidence
                        total_det_count += 1

                    # Draw annotations
                    annotated = self.detector.draw_detections(
                        process_image, detections, severities
                    )
                    annotated_path = annotated_dir / frame_path.name
                    cv2.imwrite(str(annotated_path), annotated)

            frame_result = FrameResult(
                frame_number=frame_num,
                timestamp=timestamp,
                original_path=str(frame_path),
                enhanced_path=str(enhanced_path) if enhanced_path else None,
                annotated_path=str(annotated_path) if annotated_path else None,
                enhancement_method=self.config.enhancement_method,
                detections=frame_detections,
                crack_count=len(frame_detections),
            )
            frame_results.append(frame_result)

        self._update_progress(
            InspectionStatus.GENERATING_VIDEO,
            82,
            "Generating annotated video",
            processed_frames=len(frames_to_process),
            detection_count=total_det_count,
        )

        # ── Step 7: Generate annotated video ──────────────────────────
        annotated_video_path = None
        annotated_frames = list(annotated_dir.glob("*.jpg")) + list(
            annotated_dir.glob("*.png")
        )
        if annotated_frames:
            annotated_video_path = output_base / "annotated_video.mp4"
            self._generate_annotated_video(
                annotated_frames, annotated_video_path, fps
            )

        # ── Step 8: Build summary ─────────────────────────────────────
        self._update_progress(
            InspectionStatus.GENERATING_REPORT, 90, "Generating report"
        )

        avg_conf = (total_confidence / total_det_count) if total_det_count > 0 else 0.0

        summary = InspectionSummary(
            inspection_id=inspection_id,
            filename=video_path.name,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            status=InspectionStatus.COMPLETED,
            video_duration=duration,
            total_frames_extracted=num_extracted,
            selected_frames=num_selected,
            rejected_frames=rejected_count,
            processed_frames=len(frame_results),
            total_detections=total_det_count,
            severity_distribution=severity_counts,
            avg_confidence=round(avg_conf, 4),
            enhancement_method=self.config.enhancement_method.value,
        )

        result = InspectionResultFull(
            summary=summary,
            config=self.config,
            frame_results=frame_results,
            frame_qualities=frame_qualities if isinstance(frame_qualities, list) else [],
            annotated_video_path=str(annotated_video_path) if annotated_video_path else None,
        )

        # Save JSON results
        result_json_path = output_base / "results.json"
        with open(result_json_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        result.report_json_path = str(result_json_path)

        # Generate PDF report
        try:
            from backend.app.services.report_generator import ReportGenerator
            report_gen = ReportGenerator(output_base)
            pdf_path = report_gen.generate_pdf_report(result, annotated_frames)
            result.report_pdf_path = str(pdf_path)
        except Exception as e:
            logger.warning(f"Could not generate PDF in pipeline: {e}")

        elapsed = time.time() - start_time
        logger.info(
            f"Pipeline complete in {elapsed:.1f}s | "
            f"Detections={total_det_count} | Frames={len(frame_results)}"
        )

        self._update_progress(InspectionStatus.COMPLETED, 100, "Complete")
        return result

    def process_image(
        self,
        image_path: str | Path,
        inspection_id: str,
        output_base: str | Path | None = None,
    ) -> InspectionResultFull:
        """Process a single image (not a video)."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        output_base = Path(output_base) if output_base else OUTPUT_DIR / inspection_id
        output_base.mkdir(parents=True, exist_ok=True)

        enhanced_dir = output_base / "enhanced"
        annotated_dir = output_base / "annotated"
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir.mkdir(parents=True, exist_ok=True)

        self.progress.inspection_id = inspection_id

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Enhance
        method = self.config.enhancement_method.value
        if method != "original":
            enhanced = self.enhancer.enhance(
                image,
                method=method,
                clip_limit=self.config.clahe_clip_limit,
                tile_grid_size=self.config.clahe_tile_grid,
                gamma=self.config.gamma_value,
            )
            enhanced_path = enhanced_dir / image_path.name
            cv2.imwrite(str(enhanced_path), enhanced)
            process_image = enhanced
        else:
            enhanced_path = None
            process_image = image

        # Detect
        model_loaded = self.detector.load_model()
        frame_detections: list[DetectionResult] = []
        annotated_path = None
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "UNKNOWN": 0}

        if model_loaded:
            detections = self.detector.detect_image(process_image, frame_number=0)
            if method != "original":
                orig_detections = self.detector.detect_image(image, frame_number=0)
                detections = self._merge_detections(detections, orig_detections)

            if detections:
                severities = self.severity_assessor.assess_multiple(
                    process_image, detections
                )
                for det, sev in zip(detections, severities):
                    frame_detections.append(DetectionResult(detection=det, severity=sev))
                    severity_counts[sev.severity.value] += 1

                annotated = self.detector.draw_detections(
                    process_image, detections, severities
                )
                annotated_path = annotated_dir / image_path.name
                cv2.imwrite(str(annotated_path), annotated)

        total_det = len(frame_detections)
        avg_conf = (
            sum(d.detection.confidence for d in frame_detections) / total_det
            if total_det > 0
            else 0.0
        )

        frame_result = FrameResult(
            frame_number=0,
            timestamp=0.0,
            original_path=str(image_path),
            enhanced_path=str(enhanced_path) if enhanced_path else None,
            annotated_path=str(annotated_path) if annotated_path else None,
            enhancement_method=self.config.enhancement_method,
            detections=frame_detections,
            crack_count=total_det,
        )

        summary = InspectionSummary(
            inspection_id=inspection_id,
            filename=image_path.name,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            status=InspectionStatus.COMPLETED,
            total_frames_extracted=1,
            selected_frames=1,
            rejected_frames=0,
            processed_frames=1,
            total_detections=total_det,
            severity_distribution=severity_counts,
            avg_confidence=round(avg_conf, 4),
            enhancement_method=self.config.enhancement_method.value,
        )

        result = InspectionResultFull(
            summary=summary,
            config=self.config,
            frame_results=[frame_result],
        )

        result_json_path = output_base / "results.json"
        with open(result_json_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        result.report_json_path = str(result_json_path)

        return result

    # ── Helpers ────────────────────────────────────────────────────────

    def _calculate_iou(self, box1, box2) -> float:
        """Calculate Intersection over Union between two BoundingBoxes."""
        x_left = max(box1.x1, box2.x1)
        y_top = max(box1.y1, box2.y1)
        x_right = min(box1.x2, box2.x2)
        y_bottom = min(box1.y2, box2.y2)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union = area1 + area2 - intersection
        return float(intersection / union) if union > 0 else 0.0

    def _merge_detections(
        self,
        dets_primary: list[CrackDetection],
        dets_secondary: list[CrackDetection],
        iou_thresh: float = 0.40,
    ) -> list[CrackDetection]:
        """Combine detections from two versions (e.g. enhanced + raw), avoiding duplicates."""
        merged = list(dets_primary)
        for d2 in dets_secondary:
            matched_idx = -1
            for idx, d1 in enumerate(merged):
                if self._calculate_iou(d1.bbox, d2.bbox) > iou_thresh:
                    matched_idx = idx
                    break
            if matched_idx >= 0:
                # If secondary detection has higher confidence, replace it
                if d2.confidence > merged[matched_idx].confidence:
                    merged[matched_idx] = d2
            else:
                merged.append(d2)
        return merged

    def _generate_annotated_video(
        self, frame_paths: list[Path], output_path: Path, fps: float
    ):
        """Compile annotated frames into a browser-compatible H.264 video."""
        frame_paths = sorted(frame_paths)
        if not frame_paths:
            return

        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            return

        h, w = first.shape[:2]
        fps_clamped = min(fps, 10.0)
        temp_path = output_path.with_name(f"temp_raw_{output_path.name}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(temp_path), fourcc, fps_clamped, (w, h))

        for fp in frame_paths:
            frame = cv2.imread(str(fp))
            if frame is not None:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                out.write(frame)

        out.release()

        # Convert to web-standard H.264 (AVC1) + YUV420P + faststart
        converted = False
        try:
            import subprocess
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i", str(temp_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            converted = True
            logger.info(f"Web-compatible H.264 annotated video saved: {output_path}")
        except Exception as e:
            logger.warning(f"Could not convert video via ffmpeg: {e}")

        if converted and temp_path.exists():
            temp_path.unlink()
        elif not converted and temp_path.exists():
            if output_path.exists():
                output_path.unlink()
            temp_path.rename(output_path)


# ── Async entry point ─────────────────────────────────────────────────────

async def run_pipeline_async(
    video_path: str | Path,
    inspection_id: str,
    config: InspectionConfig | None = None,
    output_base: str | Path | None = None,
    progress_callback: Callable | None = None,
    is_image: bool = False,
) -> InspectionResultFull:
    """Run the pipeline in a background thread so we don't block the event loop."""
    pipeline = VideoPipeline(config=config)
    if progress_callback:
        pipeline.set_progress_callback(progress_callback)

    loop = asyncio.get_running_loop()
    if is_image:
        result = await loop.run_in_executor(
            _executor, pipeline.process_image, video_path, inspection_id, output_base
        )
    else:
        result = await loop.run_in_executor(
            _executor, pipeline.process_video, video_path, inspection_id, output_base
        )
    return result
