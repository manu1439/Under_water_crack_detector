"""
AquaGuard AI — Crack Severity Assessment Module.

Performs **heuristic** visual analysis of detected crack regions to estimate
severity.  All spatial measurements are expressed in **pixel** units.

IMPORTANT DISTINCTION:
    • Detection confidence (from YOLO) = how certain the model is that the
      region contains a crack.
    • Severity = how serious / dangerous the crack appears based on visual
      characteristics such as size, edge complexity, and darkness.

The two metrics are independent.  A model may be 99 % confident it found a
hairline crack (LOW severity), or 30 % confident about a region that, if it
really is a crack, looks severe (HIGH severity).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from backend.app.schemas.models import (
    BoundingBox,
    CrackDetection,
    SeverityAssessment,
    SeverityLevel,
)

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────

_MIN_CROP_PIXELS: int = 5
"""Minimum width *and* height (in pixels) for a crop to be analysable."""

_DARKNESS_THRESHOLD: int = 70
"""Grayscale intensity below which a pixel is considered "dark"."""

_CANNY_LOW: int = 50
"""Lower hysteresis threshold for Canny edge detection."""

_CANNY_HIGH: int = 150
"""Upper hysteresis threshold for Canny edge detection."""

# Adaptive thresholding for crack pixel segmentation
_ADAPTIVE_BLOCK_SIZE: int = 15
"""Block size for local adaptive thresholding (must be odd)."""

_ADAPTIVE_C: int = 3
"""Constant subtracted from mean in adaptive thresholding."""

# Severity score band boundaries
_LOW_UPPER: float = 30.0
_MEDIUM_UPPER: float = 55.0

# Weight vector (must sum to 1.0)
_W_CRACK_PIXELS: float = 0.45
_W_DARK: float = 0.25
_W_ASPECT: float = 0.15
_W_EDGE: float = 0.15


class SeverityAssessor:
    """Heuristic crack severity assessor.

    Analyses the visual characteristics of a crack region (crop) extracted
    from a detection bounding box and produces a composite severity score
    in the range [0, 100].

    The score is a weighted combination of:
        • **crack_pixel_count** (40 %) — segmented crack pixel count and density.
        • **edge_density** (25 %) — ratio of Canny edge pixels in the crop.
        • **darkness_ratio** (20 %) — fraction of dark pixels (< threshold).
        • **aspect_ratio** (15 %) — elongation of the bounding box.

    Severity categories:
        • 0–33   → LOW
        • 34–60  → MEDIUM
        • 61–100 → HIGH
    """

    # ── Public API ────────────────────────────────────────────────────────

    def assess_crack(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        image_width: int,
        image_height: int,
    ) -> SeverityAssessment:
        """Assess the severity of a single crack detection.

        Parameters
        ----------
        image:
            Source image as a NumPy array (BGR or grayscale).
        bbox:
            Bounding box of the detected crack region.
        image_width:
            Full image width in pixels.
        image_height:
            Full image height in pixels.

        Returns
        -------
        SeverityAssessment
            Populated assessment with severity level, score (0–100),
            assessment_method ``"heuristic"``, measurement_unit ``"pixel"``,
            and human-readable reasons.
        """
        crop = self._extract_crop(image, bbox)

        if crop is None:
            logger.warning(
                "Crop too small or invalid for bbox (%s); returning UNKNOWN.",
                bbox,
            )
            return SeverityAssessment(
                severity=SeverityLevel.UNKNOWN,
                severity_score=0.0,
                assessment_method="heuristic",
                measurement_unit="pixel",
                reasons=["Crack region too small for reliable analysis."],
            )

        # ── Visual feature extraction ────────────────────────────────────
        area_ratio = self._calculate_area_ratio(bbox, image_width, image_height)
        aspect_ratio = self._calculate_aspect_ratio(bbox)
        edge_density = self._analyze_edge_density(crop)
        darkness_ratio = self._analyze_darkness(crop, threshold=_DARKNESS_THRESHOLD)
        crack_pixel_count, crack_pixel_ratio, estimated_width = self._segment_crack_pixels(crop)

        bbox_w = abs(bbox.x2 - bbox.x1)
        bbox_h = abs(bbox.y2 - bbox.y1)
        width_px = int(round(bbox_w))
        height_px = int(round(bbox_h))
        length_estimate = max(width_px, height_px)

        # ── Sub-scores (each normalised to 0-100) ───────────────────────
        # 1. Crack pixel & width score
        #    Combines absolute crack pixel count (cap at 1500 px) and estimated width (cap at 3.5 px)
        count_score = min(crack_pixel_count / 1500.0, 1.0) * 100.0
        width_score = min(estimated_width / 3.5, 1.0) * 100.0
        pixel_score = 0.50 * count_score + 0.50 * width_score

        # 2. Edge density — scaled to typical underwater range
        edge_score = min(edge_density / 0.40, 1.0) * 100.0

        # 3. Darkness ratio — scaled to typical underwater contrast
        dark_score = min(darkness_ratio / 0.70, 1.0) * 100.0

        # 4. Aspect ratio — 1 = square, higher = more elongated.
        aspect_score = min((aspect_ratio - 1.0) / 6.0, 1.0) * 100.0
        aspect_score = max(aspect_score, 0.0)

        # ── Weighted composite ───────────────────────────────────────────
        severity_score = (
            _W_CRACK_PIXELS * pixel_score
            + _W_DARK * dark_score
            + _W_ASPECT * aspect_score
            + _W_EDGE * edge_score
        )
        severity_score = float(np.clip(severity_score, 0.0, 100.0))

        severity_level = self._score_to_severity(severity_score)
        reasons = self._generate_reasons(
            score=severity_score,
            area_ratio=area_ratio,
            edge_density=edge_density,
            darkness_ratio=darkness_ratio,
            aspect_ratio=aspect_ratio,
            width_px=width_px,
            height_px=height_px,
            crack_pixels=crack_pixel_count,
            estimated_width=estimated_width,
        )

        logger.debug(
            "Severity assessment: score=%.1f level=%s  "
            "(crack_px=%d px_ratio=%.3f edge=%.3f dark=%.3f ar=%.2f est_w=%.1f)",
            severity_score,
            severity_level.value,
            crack_pixel_count,
            crack_pixel_ratio,
            edge_density,
            darkness_ratio,
            aspect_ratio,
            estimated_width,
        )

        return SeverityAssessment(
            severity=severity_level,
            severity_score=round(severity_score, 2),
            assessment_method="heuristic",
            measurement_unit="pixel",
            reasons=reasons,
        )

    def assess_multiple(
        self,
        image: np.ndarray,
        detections: list[CrackDetection],
    ) -> list[SeverityAssessment]:
        """Assess severity for every detection in *detections*.

        Parameters
        ----------
        image:
            Source image (BGR or grayscale) corresponding to the detections.
        detections:
            List of :class:`CrackDetection` objects from the detector.

        Returns
        -------
        list[SeverityAssessment]
            One :class:`SeverityAssessment` per detection, in the same order.
        """
        assessments: list[SeverityAssessment] = []
        for det in detections:
            assessment = self.assess_crack(
                image=image,
                bbox=det.bbox,
                image_width=det.image_width,
                image_height=det.image_height,
            )
            assessments.append(assessment)
            logger.info(
                "Detection %s — severity=%s score=%.1f",
                det.detection_id,
                assessment.severity.value,
                assessment.severity_score,
            )
        return assessments

    # ── Private helpers ───────────────────────────────────────────────────

    def _extract_crop(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
    ) -> np.ndarray | None:
        """Extract the bounding-box region from *image*.

        Coordinates are clamped to image bounds.  Returns ``None`` when the
        resulting crop is smaller than :data:`_MIN_CROP_PIXELS` in either
        dimension.
        """
        img_h, img_w = image.shape[:2]

        x1 = int(max(0, round(bbox.x1)))
        y1 = int(max(0, round(bbox.y1)))
        x2 = int(min(img_w, round(bbox.x2)))
        y2 = int(min(img_h, round(bbox.y2)))

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < _MIN_CROP_PIXELS or crop_h < _MIN_CROP_PIXELS:
            return None

        return image[y1:y2, x1:x2].copy()

    def _segment_crack_pixels(self, crop: np.ndarray) -> tuple[int, float, float]:
        """Segment and count the actual foreground crack pixels inside *crop*.

        Uses local adaptive thresholding and morphological filtering to isolate
        the dark crack fissure from concrete background texture.

        Returns
        -------
        tuple[int, float, float]
            - crack_pixel_count: total active crack pixels
            - crack_pixel_ratio: crack pixels / total crop pixels
            - estimated_width: estimated average crack thickness (px)
        """
        if crop.size == 0:
            return 0, 0.0, 0.0

        gray = (
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if len(crop.shape) == 3
            else crop
        )
        h, w = gray.shape[:2]
        total_pixels = h * w
        if total_pixels == 0:
            return 0, 0.0, 0.0

        # Gaussian blur to suppress camera noise and fine water particulates
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Dynamic block size for adaptive thresholding (must be odd and <= min dimension)
        min_dim = min(h, w)
        block_size = _ADAPTIVE_BLOCK_SIZE
        if block_size >= min_dim:
            block_size = max(3, (min_dim // 2) * 2 - 1)

        # Adaptive thresholding (extracts darker crack lines as foreground 255)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            _ADAPTIVE_C,
        )

        # Morphological opening to eliminate isolated single-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Filter out tiny disconnected specks (< 8 pixels)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
        filtered_mask = np.zeros_like(clean_mask)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= 8:
                filtered_mask[labels == label] = 255

        # If connected component filter removed everything (e.g. faint hairline), use clean_mask
        final_mask = filtered_mask if np.count_nonzero(filtered_mask) > 0 else clean_mask

        crack_pixels = int(np.count_nonzero(final_mask))
        crack_pixel_ratio = float(crack_pixels / total_pixels)

        # Estimated average crack width along the longer axis
        length_estimate = max(h, w)
        estimated_width = float(round(crack_pixels / max(length_estimate, 1), 1))

        return crack_pixels, crack_pixel_ratio, estimated_width

    def _analyze_edge_density(self, crop: np.ndarray) -> float:
        """Return the fraction of edge pixels in *crop* via Canny detection.

        Parameters
        ----------
        crop:
            Image region (BGR or grayscale).

        Returns
        -------
        float
            Value in [0.0, 1.0].  Higher values indicate a complex crack
            pattern with branching or fragmentation.
        """
        if crop.size == 0:
            return 0.0

        gray = (
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if len(crop.shape) == 3
            else crop
        )
        edges = cv2.Canny(gray, _CANNY_LOW, _CANNY_HIGH)
        total_pixels = edges.shape[0] * edges.shape[1]
        if total_pixels == 0:
            return 0.0

        edge_pixels = int(np.count_nonzero(edges))
        return float(edge_pixels / total_pixels)

    def _analyze_darkness(
        self,
        crop: np.ndarray,
        threshold: int = _DARKNESS_THRESHOLD,
    ) -> float:
        """Return the fraction of pixels darker than *threshold*.

        Cracks typically appear as dark lines/regions; a high darkness ratio
        suggests a deeper or wider crack.

        Parameters
        ----------
        crop:
            Image region (BGR or grayscale).
        threshold:
            Intensity cutoff (0–255).  Pixels strictly below this value are
            counted as "dark".

        Returns
        -------
        float
            Value in [0.0, 1.0].
        """
        if crop.size == 0:
            return 0.0

        gray = (
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if len(crop.shape) == 3
            else crop
        )
        total_pixels = gray.shape[0] * gray.shape[1]
        if total_pixels == 0:
            return 0.0

        dark_pixels = int(np.count_nonzero(gray < threshold))
        return float(dark_pixels / total_pixels)

    def _calculate_area_ratio(
        self,
        bbox: BoundingBox,
        img_w: int,
        img_h: int,
    ) -> float:
        """Return the fraction of the image area occupied by *bbox*.

        Parameters
        ----------
        bbox:
            Bounding box of the crack region.
        img_w:
            Image width in pixels.
        img_h:
            Image height in pixels.

        Returns
        -------
        float
            Value in [0.0, 1.0].
        """
        image_area = img_w * img_h
        if image_area == 0:
            return 0.0

        bbox_w = abs(bbox.x2 - bbox.x1)
        bbox_h = abs(bbox.y2 - bbox.y1)
        bbox_area = bbox_w * bbox_h

        return float(min(bbox_area / image_area, 1.0))

    def _calculate_aspect_ratio(self, bbox: BoundingBox) -> float:
        """Return the aspect ratio ``max(w, h) / min(w, h)`` of *bbox*.

        A higher ratio indicates a more elongated region, which is typical
        of crack lines.

        Returns
        -------
        float
            Always ≥ 1.0.  Returns 1.0 when the box is degenerate.
        """
        bbox_w = abs(bbox.x2 - bbox.x1)
        bbox_h = abs(bbox.y2 - bbox.y1)

        long_side = max(bbox_w, bbox_h)
        short_side = min(bbox_w, bbox_h)

        if short_side < 1.0:
            return 1.0

        return float(long_side / short_side)

    def _score_to_severity(self, score: float) -> SeverityLevel:
        """Map a numeric severity score to a :class:`SeverityLevel` enum.

        Bands:
            • 0–33   → LOW
            • 34–60  → MEDIUM
            • 61–100 → HIGH

        Parameters
        ----------
        score:
            Composite severity score in [0, 100].

        Returns
        -------
        SeverityLevel
        """
        if score <= _LOW_UPPER:
            return SeverityLevel.LOW
        if score <= _MEDIUM_UPPER:
            return SeverityLevel.MEDIUM
        return SeverityLevel.HIGH

    def _generate_reasons(
        self,
        score: float,
        area_ratio: float,
        edge_density: float,
        darkness_ratio: float,
        aspect_ratio: float,
        width_px: int,
        height_px: int,
        crack_pixels: int = 0,
        estimated_width: float = 0.0,
    ) -> list[str]:
        """Generate human-readable sentences explaining the severity rating.

        Each sentence describes one visual characteristic and its qualitative
        interpretation.

        Parameters
        ----------
        score:
            Overall severity score (0–100).
        area_ratio:
            Fraction of image area occupied by the crack bbox.
        edge_density:
            Fraction of edge pixels in the crop (0–1).
        darkness_ratio:
            Fraction of dark pixels in the crop (0–1).
        aspect_ratio:
            Elongation ratio (≥ 1.0).
        width_px:
            Bounding-box width in pixels.
        height_px:
            Bounding-box height in pixels.
        crack_pixels:
            Segmented crack foreground pixel count.
        estimated_width:
            Estimated average crack width in pixels.

        Returns
        -------
        list[str]
            Ordered list of descriptive reason strings.
        """
        reasons: list[str] = []

        # ── Crack pixel count & width ────────────────────────────────────
        if crack_pixels > 0:
            reasons.append(
                f"Segmented crack area contains {crack_pixels:,} active fracture pixels "
                f"with an estimated average width of {estimated_width:.1f} px."
            )
        else:
            reasons.append("Segmented crack area shows a hairline fracture pattern.")

        # ── Area ratio ───────────────────────────────────────────────────
        area_pct = area_ratio * 100.0
        if area_pct < 1.0:
            area_desc = "minor"
        elif area_pct < 5.0:
            area_desc = "moderate"
        else:
            area_desc = "significant"
        reasons.append(
            f"Crack region occupies {area_pct:.1f}% of the image area, "
            f"indicating {area_desc} extent."
        )

        # ── Edge density ─────────────────────────────────────────────────
        if edge_density < 0.10:
            edge_desc = "a simple, uniform crack surface"
        elif edge_density < 0.30:
            edge_desc = "moderate surface texture or minor branching"
        else:
            edge_desc = "complex crack pattern with branching"
        reasons.append(
            f"{'High' if edge_density >= 0.30 else 'Moderate' if edge_density >= 0.10 else 'Low'} "
            f"edge density ({edge_density:.2f}) suggests {edge_desc}."
        )

        # ── Darkness ratio ───────────────────────────────────────────────
        if darkness_ratio < 0.15:
            dark_desc = "a shallow or narrow surface crack"
        elif darkness_ratio < 0.40:
            dark_desc = "a moderately deep crack"
        else:
            dark_desc = "a deep or wide crack"
        reasons.append(
            f"Dark pixel ratio of {darkness_ratio:.2f} indicates {dark_desc}."
        )

        # ── Aspect ratio ─────────────────────────────────────────────────
        if aspect_ratio >= 3.0:
            reasons.append(
                f"High elongation ratio ({aspect_ratio:.1f}:1) suggests a "
                f"linear crack extending across the surface."
            )
        elif aspect_ratio >= 1.5:
            reasons.append(
                f"Moderate elongation ratio ({aspect_ratio:.1f}:1) indicates "
                f"a somewhat directional crack."
            )
        else:
            reasons.append(
                f"Low elongation ratio ({aspect_ratio:.1f}:1) suggests a "
                f"localised or roughly square damage area."
            )

        # ── Absolute dimensions ──────────────────────────────────────────
        length_estimate = max(width_px, height_px)
        reasons.append(
            f"Crack spans approximately {length_estimate} pixels in length "
            f"and {min(width_px, height_px)} pixels in width."
        )

        # ── Overall summary ──────────────────────────────────────────────
        level = self._score_to_severity(score)
        level_desc_map = {
            SeverityLevel.LOW: "Low severity — minor surface crack, monitor over time.",
            SeverityLevel.MEDIUM: "Medium severity — notable crack requiring closer inspection.",
            SeverityLevel.HIGH: "High severity — significant damage requiring prompt attention.",
            SeverityLevel.UNKNOWN: "Severity could not be determined.",
        }
        reasons.append(
            f"Overall heuristic severity score: {score:.1f}/100. "
            f"{level_desc_map[level]}"
        )

        return reasons
