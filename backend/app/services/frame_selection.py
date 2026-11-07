"""
AquaGuard AI — Video frame extraction and intelligent frame selection.

This module provides the :class:`FrameSelector` class which handles:

1. **Frame Extraction** — Reads a video file with OpenCV and extracts every
   *N*-th frame as a JPEG image.
2. **Frame Quality Assessment** — Scores each frame for blur (Laplacian
   variance), brightness, and contrast, producing a normalised 0–100 quality
   score.
3. **Duplicate Detection** — Uses perceptual hashing (pHash) to identify
   near-duplicate frames and keep only unique viewpoints.
4. **Best-Frame Selection** — Combines all quality and duplicate checks to
   automatically select the best frames for downstream crack detection.

A high-level async convenience function :func:`extract_and_select_frames`
wraps the CPU-bound pipeline so it can be ``await``-ed from FastAPI route
handlers.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import imagehash
import numpy as np
from PIL import Image

from backend.app.config import (
    BLUR_THRESHOLD,
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    CONTRAST_MIN,
    DEFAULT_FRAME_INTERVAL,
    DUPLICATE_HASH_THRESHOLD,
)
from backend.app.schemas.models import FrameQuality

logger = logging.getLogger(__name__)

# ── Internal type aliases ────────────────────────────────────────────────

FrameInfo = dict[str, Any]
"""Dict with keys: frame_number, timestamp, path."""


# ── FrameSelector ────────────────────────────────────────────────────────


class FrameSelector:
    """Extracts, assesses and selects the best frames from an underwater video.

    Parameters
    ----------
    blur_threshold:
        Laplacian-variance threshold.  Frames with variance **below** this
        value are considered blurry and rejected.
    brightness_min / brightness_max:
        Acceptable range for mean grayscale brightness (0–255).
    contrast_min:
        Minimum standard-deviation of grayscale pixel values.
    duplicate_hash_threshold:
        Maximum Hamming distance between two perceptual hashes for frames to
        be considered duplicates.  Lower = stricter.
    """

    def __init__(
        self,
        blur_threshold: float = BLUR_THRESHOLD,
        brightness_min: float = BRIGHTNESS_MIN,
        brightness_max: float = BRIGHTNESS_MAX,
        contrast_min: float = CONTRAST_MIN,
        duplicate_hash_threshold: int = DUPLICATE_HASH_THRESHOLD,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.contrast_min = contrast_min
        self.duplicate_hash_threshold = duplicate_hash_threshold

        logger.info(
            "FrameSelector initialised — blur_thresh=%.1f, brightness=[%.1f, %.1f], "
            "contrast_min=%.1f, dup_hash_thresh=%d",
            self.blur_threshold,
            self.brightness_min,
            self.brightness_max,
            self.contrast_min,
            self.duplicate_hash_threshold,
        )

    # ── A. Frame extraction ──────────────────────────────────────────────

    def extract_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        interval: int = DEFAULT_FRAME_INTERVAL,
    ) -> list[FrameInfo]:
        """Extract every *interval*-th frame from a video and save as JPEG.

        Parameters
        ----------
        video_path:
            Path to the source video file.
        output_dir:
            Directory in which extracted frame images are saved.
        interval:
            Extract one frame every *interval* frames.  For example,
            ``interval=30`` with a 30-fps video extracts one frame per second.

        Returns
        -------
        list[FrameInfo]
            A list of dicts, each containing ``frame_number`` (``int``),
            ``timestamp`` (``float``, seconds) and ``path`` (``str``).

        Raises
        ------
        FileNotFoundError
            If *video_path* does not exist.
        ValueError
            If the video cannot be opened or contains no frames.
        """

        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration: float = total_frames / fps if fps > 0 else 0.0

        logger.info(
            "Video opened — %s | FPS=%.2f | total_frames=%d | duration=%.2fs",
            video_path.name,
            fps,
            total_frames,
            duration,
        )

        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video file contains no frames: {video_path}")

        extracted: list[FrameInfo] = []
        frame_idx: int = 0
        read_failures: int = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                timestamp = frame_idx / fps if fps > 0 else 0.0
                filename = f"frame_{frame_idx:06d}.jpg"
                frame_path = output_dir / filename

                success = cv2.imwrite(str(frame_path), frame)
                if success:
                    extracted.append(
                        {
                            "frame_number": frame_idx,
                            "timestamp": round(timestamp, 3),
                            "path": str(frame_path),
                        }
                    )
                else:
                    read_failures += 1
                    logger.warning(
                        "Failed to write frame %d to %s", frame_idx, frame_path
                    )

            frame_idx += 1

        cap.release()

        if read_failures:
            logger.warning(
                "Encountered %d write failures during extraction.", read_failures
            )

        logger.info(
            "Extracted %d frames from %s (interval=%d).",
            len(extracted),
            video_path.name,
            interval,
        )

        return extracted

    # ── B. Frame quality assessment ──────────────────────────────────────

    def assess_frame_quality(
        self,
        image_path_or_array: Union[str, Path, np.ndarray],
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> FrameQuality:
        """Compute quality metrics for a single frame.

        Parameters
        ----------
        image_path_or_array:
            Either a filesystem path to an image, or a BGR ``numpy`` array
            as returned by ``cv2.imread``.
        frame_number:
            Frame index within the source video (used for book-keeping).
        timestamp:
            Timestamp (seconds) within the source video.

        Returns
        -------
        FrameQuality
            Pydantic model populated with blur, brightness, contrast and
            overall quality scores, plus a boolean ``is_selected`` flag and
            a ``rejection_reasons`` list.
        """

        # Load image ---------------------------------------------------------
        if isinstance(image_path_or_array, (str, Path)):
            image = cv2.imread(str(image_path_or_array))
            if image is None:
                logger.warning(
                    "Unable to read image: %s — returning zero-quality.",
                    image_path_or_array,
                )
                return FrameQuality(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    blur_score=0.0,
                    brightness=0.0,
                    contrast=0.0,
                    quality_score=0.0,
                    is_selected=False,
                    rejection_reasons=["unreadable_image"],
                )
        else:
            image = image_path_or_array

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Metrics ------------------------------------------------------------
        blur_score: float = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness: float = float(np.mean(gray))
        contrast: float = float(np.std(gray))

        # Quality score (weighted, normalised to 0–100) ----------------------
        #   blur      → 40 %   — clamp to [0, 1000], scale to 0–1
        #   brightness → 30 %  — distance from ideal centre 128, scale 0–1
        #   contrast   → 30 %  — clamp to [0, 128], scale to 0–1
        blur_norm = min(blur_score / 1000.0, 1.0)
        brightness_norm = 1.0 - abs(brightness - 128.0) / 128.0
        brightness_norm = max(brightness_norm, 0.0)
        contrast_norm = min(contrast / 128.0, 1.0)

        quality_score = (
            0.40 * blur_norm + 0.30 * brightness_norm + 0.30 * contrast_norm
        ) * 100.0
        quality_score = round(max(0.0, min(quality_score, 100.0)), 2)

        # Selection decision --------------------------------------------------
        rejection_reasons: list[str] = []

        if blur_score < self.blur_threshold:
            rejection_reasons.append(
                f"blurry (laplacian_var={blur_score:.1f} < {self.blur_threshold})"
            )

        if brightness < self.brightness_min:
            rejection_reasons.append(
                f"too_dark (brightness={brightness:.1f} < {self.brightness_min})"
            )
        elif brightness > self.brightness_max:
            rejection_reasons.append(
                f"too_bright (brightness={brightness:.1f} > {self.brightness_max})"
            )

        if contrast < self.contrast_min:
            rejection_reasons.append(
                f"low_contrast (contrast={contrast:.1f} < {self.contrast_min})"
            )

        is_selected = len(rejection_reasons) == 0

        return FrameQuality(
            frame_number=frame_number,
            timestamp=timestamp,
            blur_score=round(blur_score, 2),
            brightness=round(brightness, 2),
            contrast=round(contrast, 2),
            quality_score=quality_score,
            is_selected=is_selected,
            rejection_reasons=rejection_reasons,
        )

    @staticmethod
    def compute_phash(image: Union[str, Path, np.ndarray]) -> imagehash.ImageHash:
        """Compute a perceptual hash (pHash) for an image.

        Parameters
        ----------
        image:
            Filesystem path *or* a BGR ``numpy`` array.

        Returns
        -------
        imagehash.ImageHash
            64-bit perceptual hash suitable for Hamming-distance comparison.
        """

        if isinstance(image, (str, Path)):
            pil_image = Image.open(str(image))
        else:
            # cv2 BGR → RGB → PIL
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

        return imagehash.phash(pil_image)

    @staticmethod
    def is_duplicate(
        phash: imagehash.ImageHash,
        existing_hashes: list[imagehash.ImageHash],
        threshold: int = DUPLICATE_HASH_THRESHOLD,
    ) -> bool:
        """Check whether *phash* is a near-duplicate of any hash in *existing_hashes*.

        Parameters
        ----------
        phash:
            Hash of the candidate frame.
        existing_hashes:
            Hashes of frames already accepted.
        threshold:
            Maximum Hamming distance to consider two hashes as duplicates.

        Returns
        -------
        bool
            ``True`` if the candidate is a duplicate of at least one existing
            hash; ``False`` otherwise.
        """

        for existing in existing_hashes:
            distance = phash - existing  # Hamming distance via __sub__
            if distance <= threshold:
                return True
        return False

    # ── C. Best-frame selection ──────────────────────────────────────────

    def select_best_frames(
        self,
        extracted_frames_dir: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> dict[str, Any]:
        """Assess all extracted frames, reject bad ones, and copy the best.

        The method scans *extracted_frames_dir* for ``*.jpg`` images, runs
        quality assessment and duplicate detection, and copies the selected
        frames into *output_dir*.

        Parameters
        ----------
        extracted_frames_dir:
            Directory containing extracted JPEG frames (e.g. from
            :meth:`extract_frames`).
        output_dir:
            Destination directory for the selected "best" frames.

        Returns
        -------
        dict
            ``selected``  — list of dicts with ``frame_number``, ``timestamp``,
            ``path``, ``quality_score``.
            ``rejected``  — list of dicts with ``frame_number``, ``timestamp``,
            ``path``, ``reasons``.
            ``statistics`` — summary dict with counts and averages.
        """

        extracted_frames_dir = Path(extracted_frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = sorted(extracted_frames_dir.glob("*.jpg"))

        if not frame_paths:
            logger.warning(
                "No JPEG frames found in %s — nothing to select.",
                extracted_frames_dir,
            )
            return {
                "selected": [],
                "rejected": [],
                "statistics": {
                    "total_extracted": 0,
                    "total_selected": 0,
                    "total_rejected": 0,
                    "avg_quality_score": 0.0,
                    "rejection_breakdown": {},
                },
            }

        selected: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        accepted_hashes: list[imagehash.ImageHash] = []

        # Rejection-reason counters
        rejection_counts: dict[str, int] = {
            "blurry": 0,
            "too_dark": 0,
            "too_bright": 0,
            "low_contrast": 0,
            "duplicate": 0,
            "unreadable": 0,
        }

        quality_scores: list[float] = []
        start_time = time.perf_counter()

        for frame_path in frame_paths:
            # Parse frame_number from filename (frame_000030.jpg → 30)
            stem = frame_path.stem  # e.g. "frame_000030"
            try:
                frame_number = int(stem.split("_", 1)[1])
            except (IndexError, ValueError):
                frame_number = 0

            # Rough timestamp placeholder (will be 0 if unknown)
            timestamp = 0.0

            # Quality assessment
            quality = self.assess_frame_quality(
                frame_path,
                frame_number=frame_number,
                timestamp=timestamp,
            )
            quality_scores.append(quality.quality_score)

            if not quality.is_selected:
                # Update rejection counters
                for reason in quality.rejection_reasons:
                    if "blurry" in reason:
                        rejection_counts["blurry"] += 1
                    elif "too_dark" in reason:
                        rejection_counts["too_dark"] += 1
                    elif "too_bright" in reason:
                        rejection_counts["too_bright"] += 1
                    elif "low_contrast" in reason:
                        rejection_counts["low_contrast"] += 1
                    elif "unreadable" in reason:
                        rejection_counts["unreadable"] += 1

                rejected.append(
                    {
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "path": str(frame_path),
                        "reasons": quality.rejection_reasons,
                    }
                )
                continue

            # Duplicate detection
            try:
                phash = self.compute_phash(frame_path)
            except Exception:
                logger.warning(
                    "Failed to compute pHash for %s — skipping duplicate check.",
                    frame_path.name,
                    exc_info=True,
                )
                phash = None

            if phash is not None and self.is_duplicate(
                phash, accepted_hashes, self.duplicate_hash_threshold
            ):
                rejection_counts["duplicate"] += 1
                rejected.append(
                    {
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "path": str(frame_path),
                        "reasons": ["duplicate"],
                    }
                )
                continue

            # Frame accepted — copy to output
            dest_path = output_dir / frame_path.name
            shutil.copy2(str(frame_path), str(dest_path))

            if phash is not None:
                accepted_hashes.append(phash)

            selected.append(
                {
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "path": str(dest_path),
                    "quality_score": quality.quality_score,
                }
            )

        elapsed = time.perf_counter() - start_time

        avg_quality = (
            round(sum(quality_scores) / len(quality_scores), 2)
            if quality_scores
            else 0.0
        )

        statistics: dict[str, Any] = {
            "total_extracted": len(frame_paths),
            "total_selected": len(selected),
            "total_rejected": len(rejected),
            "selection_rate_pct": round(
                len(selected) / len(frame_paths) * 100.0, 1
            )
            if frame_paths
            else 0.0,
            "avg_quality_score": avg_quality,
            "processing_time_sec": round(elapsed, 2),
            "rejection_breakdown": {
                k: v for k, v in rejection_counts.items() if v > 0
            },
        }

        logger.info(
            "Frame selection complete — %d/%d selected (%.1f%%) in %.2fs.",
            len(selected),
            len(frame_paths),
            statistics["selection_rate_pct"],
            elapsed,
        )

        return {
            "selected": selected,
            "rejected": rejected,
            "statistics": statistics,
        }


# ── Async convenience wrapper ───────────────────────────────────────────


_executor = ThreadPoolExecutor(max_workers=2)


async def extract_and_select_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    interval: int = DEFAULT_FRAME_INTERVAL,
    **kwargs: Any,
) -> dict[str, Any]:
    """High-level async wrapper that runs frame extraction + selection.

    The heavy CPU-bound OpenCV / pHash work is dispatched to a thread-pool
    executor so the event loop stays responsive.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    output_dir:
        Root output directory.  Sub-directories ``extracted`` and ``selected``
        will be created beneath it.
    interval:
        Frame extraction interval (every *N*-th frame).
    **kwargs:
        Forwarded to :class:`FrameSelector` ``__init__`` as threshold
        overrides (e.g. ``blur_threshold``, ``brightness_min``).

    Returns
    -------
    dict
        ``extracted_frames``  — list returned by :meth:`FrameSelector.extract_frames`.
        ``selected``          — list of selected frame dicts.
        ``rejected``          — list of rejected frame dicts.
        ``statistics``        — summary statistics dict.
    """

    output_dir = Path(output_dir)
    extracted_dir = output_dir / "extracted"
    selected_dir = output_dir / "selected"

    def _run() -> dict[str, Any]:
        selector = FrameSelector(**kwargs)

        logger.info("Starting frame extraction from %s …", video_path)
        extracted_frames = selector.extract_frames(
            video_path, extracted_dir, interval=interval
        )

        logger.info("Starting best-frame selection …")
        selection_result = selector.select_best_frames(extracted_dir, selected_dir)

        return {
            "extracted_frames": extracted_frames,
            "selected": selection_result["selected"],
            "rejected": selection_result["rejected"],
            "statistics": selection_result["statistics"],
        }

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _run)
    return result
