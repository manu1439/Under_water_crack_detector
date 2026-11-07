"""
AquaGuard AI — Underwater image enhancement service.

Provides multiple enhancement algorithms optimised for murky, low-contrast
underwater footage:

* **CLAHE** — Contrast-Limited Adaptive Histogram Equalisation on the L channel
  of the CIE-LAB colour space.
* **Gamma correction** — simple power-law transform via a lookup table.
* **Single-Scale Retinex (SSR)** — logarithmic ratio of the image and its
  Gaussian-blurred version, normalised back to [0, 255].

The module exposes both synchronous helpers (for direct use inside worker
threads) and a thin ``async`` wrapper (``enhance_frames``) that offloads the
CPU-bound work to an ``asyncio`` thread-pool executor.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    DEFAULT_GAMMA,
)
from backend.app.schemas.models import EnhancementMethod

logger = logging.getLogger(__name__)

# Thread-pool shared across async calls inside this module.
_ENHANCEMENT_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="enhance")


class ImageEnhancer:
    """Stateless service that applies underwater image enhancement algorithms.

    Each public ``enhance_*`` method accepts a BGR ``numpy`` array (as returned
    by ``cv2.imread``) and returns an enhanced BGR array of the same shape and
    dtype.
    """

    # ── Core enhancement algorithms ──────────────────────────────────────

    def enhance_clahe(
        self,
        image: np.ndarray,
        clip_limit: float = CLAHE_CLIP_LIMIT,
        tile_grid_size: int = CLAHE_TILE_GRID,
    ) -> np.ndarray:
        """Apply Contrast-Limited Adaptive Histogram Equalisation (CLAHE).

        The image is converted from BGR to the CIE-LAB colour space so that
        only the luminance channel (L) is equalised, preserving the original
        colour information.

        Args:
            image: Input BGR image (``uint8``).
            clip_limit: CLAHE clip limit – controls contrast amplification.
            tile_grid_size: Size of each tile in the grid used by CLAHE.

        Returns:
            Enhanced BGR image (``uint8``).
        """
        logger.debug(
            "CLAHE enhancement: clip_limit=%.2f, tile_grid=%d",
            clip_limit,
            tile_grid_size,
        )
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size),
        )
        l_enhanced = clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        result: np.ndarray = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return result

    def enhance_gamma(
        self,
        image: np.ndarray,
        gamma: float = DEFAULT_GAMMA,
    ) -> np.ndarray:
        """Apply gamma (power-law) correction via a lookup table.

        Values of ``gamma > 1`` brighten the image (useful for dark underwater
        scenes), while ``gamma < 1`` would darken it.

        Args:
            image: Input BGR image (``uint8``).
            gamma: Gamma value.  Effective exponent is ``1/gamma``.

        Returns:
            Enhanced BGR image (``uint8``).
        """
        logger.debug("Gamma correction: gamma=%.2f", gamma)
        inverse_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inverse_gamma) * 255 for i in range(256)]
        ).astype("uint8")
        result: np.ndarray = cv2.LUT(image, table)
        return result

    def enhance_retinex(
        self,
        image: np.ndarray,
        sigma: float = 300.0,
    ) -> np.ndarray:
        """Apply Single-Scale Retinex (SSR) enhancement.

        For each colour channel the method computes::

            R(x,y) = log(I(x,y)) - log(G(x,y) * I(x,y))

        where *G* is a Gaussian kernel of width ``sigma``.  The result is
        linearly stretched back to the ``[0, 255]`` range.

        Args:
            image: Input BGR image (``uint8``).
            sigma: Standard deviation of the Gaussian blur kernel.

        Returns:
            Enhanced BGR image (``uint8``).
        """
        logger.debug("Retinex enhancement: sigma=%.1f", sigma)
        img_float = image.astype(np.float64) + 1.0  # +1 to avoid log(0)

        # Compute a suitable odd kernel size from sigma.
        ksize = int(sigma * 6) | 1  # must be odd
        if ksize < 3:
            ksize = 3

        retinex = np.zeros_like(img_float)
        for ch in range(img_float.shape[2]):
            channel = img_float[:, :, ch]
            blurred = cv2.GaussianBlur(channel, (ksize, ksize), sigma)
            blurred = np.maximum(blurred, 1.0)  # avoid log(0)
            retinex[:, :, ch] = np.log10(channel) - np.log10(blurred)

        # Normalise each channel independently to [0, 255].
        for ch in range(retinex.shape[2]):
            c_min = retinex[:, :, ch].min()
            c_max = retinex[:, :, ch].max()
            denom = c_max - c_min
            if denom == 0:
                retinex[:, :, ch] = 0.0
            else:
                retinex[:, :, ch] = (retinex[:, :, ch] - c_min) / denom * 255.0

        result = retinex.astype(np.uint8)
        return result

    # ── Dispatcher ───────────────────────────────────────────────────────

    def enhance(
        self,
        image: np.ndarray,
        method: str = "clahe",
        **kwargs: Any,
    ) -> np.ndarray:
        """Dispatch to the enhancement method identified by *method*.

        Accepted method strings correspond to
        :class:`~backend.app.schemas.models.EnhancementMethod` values:
        ``"original"``, ``"clahe"``, ``"gamma"``, ``"retinex"``.

        Args:
            image: Input BGR image (``uint8``).
            method: Enhancement algorithm name.
            **kwargs: Forwarded to the chosen algorithm.

        Returns:
            Enhanced (or original) BGR image.

        Raises:
            ValueError: If *method* is not recognised.
        """
        method_lower = method.strip().lower()

        if method_lower == EnhancementMethod.ORIGINAL.value:
            logger.debug("Enhancement method 'original' — returning image unchanged.")
            return image.copy()

        dispatch: dict[str, Any] = {
            EnhancementMethod.CLAHE.value: self.enhance_clahe,
            EnhancementMethod.GAMMA.value: self.enhance_gamma,
            EnhancementMethod.RETINEX.value: self.enhance_retinex,
        }

        handler = dispatch.get(method_lower)
        if handler is None:
            valid = ", ".join(sorted(dispatch.keys()) + [EnhancementMethod.ORIGINAL.value])
            raise ValueError(
                f"Unknown enhancement method '{method}'. Choose from: {valid}"
            )

        import inspect
        sig = inspect.signature(handler)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        logger.info("Enhancing image with method '%s'", method_lower)
        enhanced: np.ndarray = handler(image, **filtered_kwargs)
        return enhanced

    # ── File-level helpers ───────────────────────────────────────────────

    def enhance_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        method: str = "clahe",
        **kwargs: Any,
    ) -> Path:
        """Read an image file, enhance it and write the result.

        Args:
            input_path: Source image path.
            output_path: Destination path (parent dirs are created as needed).
            method: Enhancement algorithm name.
            **kwargs: Forwarded to :meth:`enhance`.

        Returns:
            The resolved *output_path*.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            ValueError: If the image cannot be decoded by OpenCV.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Failed to read image: {input_path}")

        enhanced = self.enhance(image, method=method, **kwargs)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), enhanced)
        logger.info("Enhanced image saved: %s", output_path)
        return output_path

    def enhance_batch(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        method: str = "clahe",
        **kwargs: Any,
    ) -> list[Path]:
        """Enhance every image in *input_dir* and save to *output_dir*.

        Only files whose suffix is in
        :data:`~backend.app.config.ALLOWED_IMAGE_EXTENSIONS` are processed.

        Args:
            input_dir: Directory containing source images.
            output_dir: Directory to write enhanced images into.
            method: Enhancement algorithm name.
            **kwargs: Forwarded to :meth:`enhance`.

        Returns:
            List of output paths for successfully enhanced images.

        Raises:
            FileNotFoundError: If *input_dir* does not exist.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
        )

        if not image_files:
            logger.warning("No images found in %s", input_dir)
            return []

        logger.info(
            "Batch enhancement: %d images, method='%s'",
            len(image_files),
            method,
        )

        results: list[Path] = []
        for idx, img_path in enumerate(image_files, 1):
            try:
                out_path = output_dir / img_path.name
                self.enhance_file(img_path, out_path, method=method, **kwargs)
                results.append(out_path)
                if idx % 50 == 0 or idx == len(image_files):
                    logger.info("Batch progress: %d / %d", idx, len(image_files))
            except Exception:
                logger.exception("Failed to enhance %s", img_path.name)

        logger.info(
            "Batch enhancement complete: %d / %d succeeded",
            len(results),
            len(image_files),
        )
        return results

    # ── Comparison strip ─────────────────────────────────────────────────

    def create_comparison(
        self,
        image: np.ndarray,
        methods: list[str] | None = None,
    ) -> np.ndarray:
        """Create a side-by-side comparison strip for visual inspection.

        Each panel is labelled at the top with the method name using
        ``cv2.putText``.

        Args:
            image: Input BGR image (``uint8``).
            methods: Enhancement method names to include.  Defaults to all
                available methods (``original``, ``clahe``, ``gamma``,
                ``retinex``).

        Returns:
            A single horizontally-concatenated BGR image containing all
            panels.
        """
        if methods is None:
            methods = [m.value for m in EnhancementMethod]

        panels: list[np.ndarray] = []
        for method_name in methods:
            enhanced = self.enhance(image, method=method_name)

            # Draw the method label on a copy so the original array is untouched.
            labelled = enhanced.copy()
            height, width = labelled.shape[:2]

            # Adaptive font scale based on image width.
            font_scale = max(0.5, width / 800.0)
            thickness = max(1, int(font_scale * 2))

            label = method_name.upper()
            text_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            text_w, text_h = text_size

            # Background rectangle for readability.
            margin = int(text_h * 0.5)
            cv2.rectangle(
                labelled,
                (0, 0),
                (text_w + margin * 2, text_h + margin * 2),
                (0, 0, 0),
                cv2.FILLED,
            )
            cv2.putText(
                labelled,
                label,
                (margin, text_h + margin),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            panels.append(labelled)

        comparison: np.ndarray = cv2.hconcat(panels)
        logger.info(
            "Comparison strip created: %d panels, shape %s",
            len(panels),
            comparison.shape,
        )
        return comparison


# ── Module-level convenience instance ────────────────────────────────────

_default_enhancer = ImageEnhancer()


async def enhance_frames(
    input_dir: str | Path,
    output_dir: str | Path,
    method: str = "clahe",
    **kwargs: Any,
) -> list[Path]:
    """Async wrapper that runs batch enhancement in a thread-pool executor.

    This is the primary entry-point used by the inspection pipeline so that
    CPU-bound image processing does not block the ``asyncio`` event loop.

    Args:
        input_dir: Directory containing extracted frames.
        output_dir: Directory to write enhanced frames into.
        method: Enhancement algorithm name.
        **kwargs: Forwarded to :meth:`ImageEnhancer.enhance_batch`.

    Returns:
        List of paths to successfully enhanced images.
    """
    loop = asyncio.get_running_loop()
    result: list[Path] = await loop.run_in_executor(
        _ENHANCEMENT_POOL,
        lambda: _default_enhancer.enhance_batch(
            input_dir, output_dir, method=method, **kwargs
        ),
    )
    return result
