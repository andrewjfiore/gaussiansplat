"""Quick Preview: generate an instant rough 3D preview from a single video frame.

Uses the portrait pipeline's depth-to-Gaussian conversion with speed-optimized
settings to produce a preview PLY in seconds while the full pipeline runs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuickPreviewResult:
    ply_path: Path
    num_gaussians: int
    source_frame: str
    processing_time_seconds: float


def _compute_sharpness(image_gray: np.ndarray) -> float:
    """Compute sharpness using Laplacian variance (numpy-based).

    Higher values indicate sharper images.
    """
    # Laplacian kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)

    # Manual 2D convolution via sliding window (fast enough for single images)
    from scipy.signal import convolve2d
    laplacian = convolve2d(image_gray.astype(np.float32), kernel, mode="valid")
    return float(np.var(laplacian))


def _select_best_frame(frames_dir: Path, max_evaluate: int = 20) -> Path:
    """Select the sharpest frame from extracted frames.

    Evaluates a sample of frames (every Nth) for speed, returns the
    path to the sharpest one.
    """
    import cv2

    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No .jpg frames found in {frames_dir}")

    if len(frames) == 1:
        return frames[0]

    # Sample every Nth frame if there are many
    step = max(1, len(frames) // max_evaluate)
    candidates = frames[::step]

    best_path = candidates[0]
    best_score = -1.0

    for frame_path in candidates:
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Resize for faster evaluation
        h, w = img.shape[:2]
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        try:
            score = _compute_sharpness(img)
        except Exception:
            # Fallback if scipy unavailable: use cv2 Laplacian
            lap = cv2.Laplacian(img, cv2.CV_64F)
            score = float(np.var(lap))

        if score > best_score:
            best_score = score
            best_path = frame_path

    logger.info(
        "Selected sharpest frame: %s (score=%.1f, evaluated %d/%d)",
        best_path.name, best_score, len(candidates), len(frames),
    )
    return best_path


def generate_quick_preview(
    project_dir: Path,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> QuickPreviewResult:
    """Generate a quick 3D preview from the best extracted frame.

    This is a thin wrapper around the portrait pipeline optimized for speed:
    - Selects the sharpest frame automatically
    - Uses stride=3 for fewer Gaussians
    - Resizes to max 768px for faster depth estimation
    - Skips novel view generation
    - Includes background for spatial context

    Parameters
    ----------
    project_dir : Path to the project directory (contains frames/ subdirectory)
    on_progress : optional callback(message, percent)

    Returns
    -------
    QuickPreviewResult with ply_path, num_gaussians, source_frame, processing_time
    """
    import cv2
    from PIL import Image

    from . import portrait as portrait_svc

    start_time = time.time()

    def _progress(msg: str, pct: float):
        logger.info("[quick-preview %.0f%%] %s", pct, msg)
        if on_progress:
            on_progress(msg, pct)

    frames_dir = project_dir / "frames"
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Select best frame (0-10%) ---
    _progress("Selecting sharpest frame...", 0)
    best_frame = _select_best_frame(frames_dir)
    _progress(f"Selected frame: {best_frame.name}", 10)

    # --- 2. Load and resize image (10-15%) ---
    _progress("Loading image...", 10)
    pil_image = Image.open(best_frame).convert("RGB")
    image = np.array(pil_image)
    h, w = image.shape[:2]

    max_dim = 768
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
        logger.info("Resized to %dx%d for quick preview", w, h)

    _progress("Image loaded", 15)

    # --- 3. Estimate depth (15-55%) ---
    _progress("Estimating depth...", 15)

    # Save resized image for depth estimation
    preview_input = output_dir / "preview_input.jpg"
    cv2.imwrite(str(preview_input), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    depth = portrait_svc.estimate_depth(preview_input, model_size="small")

    # Resize depth to match image if needed
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    _progress("Depth estimation complete", 55)

    # --- 4. Segment subject (55-65%) ---
    _progress("Segmenting...", 55)
    mask = portrait_svc.segment_subject(image, depth)
    _progress("Segmentation complete", 65)

    # --- 5. Convert to Gaussians (65-80%) ---
    _progress("Generating Gaussians...", 65)
    gaussians = portrait_svc.depth_to_gaussians(
        image, depth, mask,
        stride=3,
        focal_mult=0.8,
        include_background=True,
    )
    num_gaussians = len(gaussians["means"])
    _progress(f"Generated {num_gaussians} Gaussians", 80)

    # --- 6. Write preview PLY (80-100%) ---
    _progress("Writing preview PLY...", 80)
    ply_path = output_dir / "preview_point_cloud.ply"
    portrait_svc.write_gaussians_ply(gaussians, ply_path)

    elapsed = time.time() - start_time
    _progress(f"Quick preview complete! ({elapsed:.1f}s, {num_gaussians} Gaussians)", 100)

    # Clean up temp file
    try:
        preview_input.unlink(missing_ok=True)
    except Exception:
        pass

    return QuickPreviewResult(
        ply_path=ply_path,
        num_gaussians=num_gaussians,
        source_frame=best_frame.name,
        processing_time_seconds=round(elapsed, 1),
    )
