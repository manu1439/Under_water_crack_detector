"""
AquaGuard AI — YOLO-based crack detection service.

Provides the ``CrackDetector`` class that wraps an Ultralytics YOLO model for
underwater crack detection.  The model is loaded lazily on the first inference
call, keeping startup time fast.  Includes helpers for single-image, file-based,
and batch detection as well as annotation drawing for visual output.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from backend.app.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEVICE,
    INFERENCE_IMG_SIZE,
    MODEL_WEIGHTS,
)
from backend.app.schemas.models import (
    BoundingBox,
    CrackDetection,
    ModelStatus,
    SeverityLevel,
)

logger = logging.getLogger(__name__)

# ── Severity-based colour palette (BGR for OpenCV) ────────────────────────
_SEVERITY_COLORS: dict[SeverityLevel, tuple[int, int, int]] = {
    SeverityLevel.LOW: (0, 200, 0),       # green
    SeverityLevel.MEDIUM: (0, 165, 255),   # orange
    SeverityLevel.HIGH: (0, 0, 255),       # red
    SeverityLevel.UNKNOWN: (180, 180, 180),  # gray
}
_DEFAULT_BOX_COLOR: tuple[int, int, int] = (0, 255, 0)  # green fallback
_LABEL_BG_COLOR: tuple[int, int, int] = (0, 0, 0)
_LABEL_TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_BOX_THICKNESS = 2


def _resolve_device(device: str) -> str:
    """Resolve device specifier to a valid torch/ultralytics device string."""
    if device in ("auto", None):
        if torch.cuda.is_available():
            resolved = "cuda:0"
            logger.info("Auto-detected CUDA device: %s", torch.cuda.get_device_name(0))
        else:
            resolved = "cpu"
            logger.info("CUDA not available — using CPU for inference")
        return resolved
    if str(device).isdigit():
        return f"cuda:{device}"
    return str(device)


class CrackDetector:
    """High-level wrapper around an Ultralytics YOLO model for crack detection.

    The model is loaded lazily — construction is cheap and does no I/O.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` weights file.  Defaults to the project-level
        ``config.MODEL_WEIGHTS``.
    device:
        Inference device (``'auto'``, ``'cpu'``, ``'cuda'``, ``'0'``, …).
    confidence:
        Minimum detection confidence threshold.
    iou:
        IoU threshold for Non-Maximum Suppression.
    img_size:
        Inference image size (pixels).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "auto",
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        img_size: int = INFERENCE_IMG_SIZE,
    ) -> None:
        self.model_path: Path = Path(model_path) if model_path else Path(MODEL_WEIGHTS)
        self.device: str = _resolve_device(device if device else DEVICE)
        self.confidence: float = confidence
        self.iou: float = iou
        self.img_size: int = img_size
        self._model: Optional[YOLO] = None

        logger.debug(
            "CrackDetector configured — weights=%s  device=%s  conf=%.2f  iou=%.2f  img=%d",
            self.model_path,
            self.device,
            self.confidence,
            self.iou,
            self.img_size,
        )

    # ── Model lifecycle ───────────────────────────────────────────────────

    def load_model(self) -> bool:
        """Load the YOLO model from disk.

        Returns
        -------
        bool
            ``True`` if the model was loaded successfully, ``False`` if the
            weights file does not exist.
        """
        if not self.model_path.exists():
            logger.warning(
                "Model weights not found at %s — train a model first.",
                self.model_path,
            )
            return False

        try:
            self._model = YOLO(str(self.model_path))
            # Move model to configured device
            self._model.to(self.device)

            class_names = list(self._model.names.values()) if self._model.names else []
            logger.info(
                "YOLO model loaded successfully — path=%s  device=%s  classes=%s",
                self.model_path,
                self.device,
                class_names,
            )
            return True

        except Exception:
            logger.exception("Failed to load YOLO model from %s", self.model_path)
            self._model = None
            return False

    @property
    def is_loaded(self) -> bool:
        """Return whether the model is currently loaded in memory."""
        return self._model is not None

    def ensure_loaded(self) -> None:
        """Load the model if it is not already loaded.

        Raises
        ------
        RuntimeError
            If the model weights file does not exist on disk.
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError(
                    f"Model not found at {self.model_path}. Train a model first."
                )

    # ── Detection ─────────────────────────────────────────────────────────

    def detect_image(
        self,
        image: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> list[CrackDetection]:
        """Run crack detection on a single in-memory image.

        Parameters
        ----------
        image:
            BGR image as a NumPy array (OpenCV convention).
        frame_number:
            Index of the frame inside the source video (0 for standalone
            images).
        timestamp:
            Timestamp of the frame in seconds.

        Returns
        -------
        list[CrackDetection]
            A (possibly empty) list of detections found in the image.
        """
        self.ensure_loaded()

        h, w = image.shape[:2]

        try:
            results = self._model.predict(
                image,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.img_size,
                verbose=False,
            )
        except Exception:
            logger.exception("YOLO prediction failed for frame %d", frame_number)
            return []

        detections: list[CrackDetection] = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                class_name = self._model.names.get(cls_id, f"class_{cls_id}")

                detection = CrackDetection(
                    detection_id=str(uuid.uuid4()),
                    class_name=class_name,
                    confidence=round(conf, 4),
                    bbox=BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    ),
                    frame_number=frame_number,
                    timestamp=timestamp,
                    image_width=w,
                    image_height=h,
                )
                detections.append(detection)

        logger.debug(
            "Frame %d: detected %d crack(s)  (%.3fs)",
            frame_number,
            len(detections),
            timestamp,
        )
        return detections

    def detect_file(
        self,
        image_path: str | Path,
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> list[CrackDetection]:
        """Run crack detection on an image file.

        Parameters
        ----------
        image_path:
            Path to the image file (JPEG, PNG, etc.).
        frame_number:
            Frame index (relevant when the image originates from a video).
        timestamp:
            Timestamp in seconds.

        Returns
        -------
        list[CrackDetection]

        Raises
        ------
        FileNotFoundError
            If *image_path* does not exist.
        ValueError
            If the file cannot be decoded as an image.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to decode image file: {image_path}")

        return self.detect_image(image, frame_number=frame_number, timestamp=timestamp)

    def detect_batch(
        self,
        images: list[np.ndarray],
        frame_numbers: list[int] | None = None,
        timestamps: list[float] | None = None,
    ) -> list[list[CrackDetection]]:
        """Run detection on a batch of images.

        The model is guaranteed to be loaded only once for the entire batch.

        Parameters
        ----------
        images:
            List of BGR images.
        frame_numbers:
            Per-image frame indices.  Defaults to ``[0, 1, 2, …]``.
        timestamps:
            Per-image timestamps.  Defaults to ``[0.0, 0.0, …]``.

        Returns
        -------
        list[list[CrackDetection]]
            Outer list mirrors *images*; inner lists contain detections per
            image.
        """
        self.ensure_loaded()

        n = len(images)
        if frame_numbers is None:
            frame_numbers = list(range(n))
        if timestamps is None:
            timestamps = [0.0] * n

        if len(frame_numbers) != n:
            raise ValueError(
                f"frame_numbers length ({len(frame_numbers)}) != images length ({n})"
            )
        if len(timestamps) != n:
            raise ValueError(
                f"timestamps length ({len(timestamps)}) != images length ({n})"
            )

        logger.info("Starting batch detection on %d images", n)

        all_detections: list[list[CrackDetection]] = []
        for idx, (img, fn, ts) in enumerate(zip(images, frame_numbers, timestamps)):
            dets = self.detect_image(img, frame_number=fn, timestamp=ts)
            all_detections.append(dets)

        total = sum(len(d) for d in all_detections)
        logger.info("Batch detection complete — %d images, %d total detections", n, total)
        return all_detections

    # ── Annotation ────────────────────────────────────────────────────────

    def draw_detections(
        self,
        image: np.ndarray,
        detections: list[CrackDetection],
        severity_results: list | None = None,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on an image.

        Parameters
        ----------
        image:
            BGR image (will **not** be modified in-place).
        detections:
            Detections to draw.
        severity_results:
            Optional parallel list of ``SeverityAssessment`` objects.  When
            provided, box colour and label text reflect severity level.

        Returns
        -------
        np.ndarray
            Annotated copy of the image.
        """
        annotated = image.copy()

        for idx, det in enumerate(detections):
            # Determine colour & severity label
            severity_tag: str = ""
            color = _DEFAULT_BOX_COLOR

            if severity_results is not None and idx < len(severity_results):
                sev = severity_results[idx]
                # Accept both SeverityAssessment objects and raw SeverityLevel enums
                level: SeverityLevel | None = None
                if hasattr(sev, "severity"):
                    level = sev.severity
                elif isinstance(sev, SeverityLevel):
                    level = sev

                if level is not None:
                    color = _SEVERITY_COLORS.get(level, _DEFAULT_BOX_COLOR)
                    severity_tag = f" [{level.value}]"

            # Coordinates (clamp to image bounds)
            x1 = max(0, int(det.bbox.x1))
            y1 = max(0, int(det.bbox.y1))
            x2 = min(annotated.shape[1], int(det.bbox.x2))
            y2 = min(annotated.shape[0], int(det.bbox.y2))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, _BOX_THICKNESS)

            # Build label text
            label = f"{det.class_name} {det.confidence * 100:.1f}%{severity_tag}"

            # Calculate text size for background rectangle
            (text_w, text_h), baseline = cv2.getTextSize(
                label, _FONT, _FONT_SCALE, _FONT_THICKNESS
            )
            text_x = x1
            text_y = y1 - 6  # small gap above the box

            # Ensure the label does not go above the image
            if text_y - text_h - baseline < 0:
                text_y = y2 + text_h + 6  # place below the box instead

            # Filled rectangle behind text for readability
            bg_top_left = (text_x, text_y - text_h - baseline)
            bg_bottom_right = (text_x + text_w, text_y + baseline)
            cv2.rectangle(annotated, bg_top_left, bg_bottom_right, _LABEL_BG_COLOR, cv2.FILLED)

            # Put the label text
            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                _FONT,
                _FONT_SCALE,
                _LABEL_TEXT_COLOR,
                _FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated

    # ── Model info ────────────────────────────────────────────────────────

    def get_model_status(self) -> ModelStatus:
        """Return current model status as a ``ModelStatus`` schema.

        Returns
        -------
        ModelStatus
        """
        model_exists = self.model_path.exists()
        model_size_mb = 0.0
        if model_exists:
            try:
                model_size_mb = round(self.model_path.stat().st_size / (1024 * 1024), 2)
            except OSError:
                model_size_mb = 0.0

        classes: list[str] = []
        base_model = ""
        if self.is_loaded and self._model is not None:
            classes = list(self._model.names.values()) if self._model.names else []
            # Try to extract the base model architecture name
            if hasattr(self._model, "ckpt") and self._model.ckpt:
                train_args = self._model.ckpt.get("train_args", {})
                if isinstance(train_args, dict):
                    base_model = str(train_args.get("model", ""))

        return ModelStatus(
            model_loaded=self.is_loaded,
            model_path=str(self.model_path),
            model_exists=model_exists,
            model_size_mb=model_size_mb,
            base_model=base_model,
            device=self.device,
            classes=classes,
            inference_img_size=self.img_size,
        )


# ── Module-level singleton ────────────────────────────────────────────────

_default_detector: CrackDetector | None = None


def get_detector(**kwargs) -> CrackDetector:
    """Return (and optionally create) the module-level ``CrackDetector`` singleton.

    Keyword arguments are forwarded to the constructor only when creating a
    new instance.

    Returns
    -------
    CrackDetector
    """
    global _default_detector
    if _default_detector is None:
        _default_detector = CrackDetector(**kwargs)
    return _default_detector


# ── Async wrapper ─────────────────────────────────────────────────────────

async def detect_cracks(
    image_path: str | Path,
    detector: CrackDetector | None = None,
    **kwargs,
) -> list[CrackDetection]:
    """Async convenience wrapper for running crack detection on an image file.

    Offloads the synchronous YOLO inference to a thread-pool executor so that
    the event loop is never blocked.

    Parameters
    ----------
    image_path:
        Path to an image file.
    detector:
        An existing ``CrackDetector`` instance.  If ``None`` the module-level
        singleton is used (created with *kwargs* if it does not yet exist).
    **kwargs:
        Forwarded to ``CrackDetector.__init__`` when constructing a new
        singleton or to ``detect_file`` when matching its signature.

    Returns
    -------
    list[CrackDetection]
    """
    if detector is None:
        # Separate constructor kwargs from detect_file kwargs
        init_keys = {"model_path", "device", "confidence", "iou", "img_size"}
        init_kwargs = {k: v for k, v in kwargs.items() if k in init_keys}
        detector = get_detector(**init_kwargs)

    detect_kwargs: dict = {}
    if "frame_number" in kwargs:
        detect_kwargs["frame_number"] = kwargs["frame_number"]
    if "timestamp" in kwargs:
        detect_kwargs["timestamp"] = kwargs["timestamp"]

    loop = asyncio.get_running_loop()
    detections = await loop.run_in_executor(
        None,
        partial(detector.detect_file, image_path, **detect_kwargs),
    )
    return detections
