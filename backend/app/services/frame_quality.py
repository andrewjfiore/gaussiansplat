"""
frame_quality.py — Frame quality filtering using blur detection.

Computes Laplacian variance for each frame and moves blurry frames
to a rejected directory. Uses OpenCV (already a dependency via equirect.py).
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    total_frames: int
    kept_frames: int
    rejected_frames: int
    min_score: float
    max_score: float
    mean_score: float


def compute_blur_score(image_path: Path) -> float:
    """Return Laplacian variance -- higher means sharper."""
    import cv2
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def filter_frames(
    frames_dir: Path,
    min_blur_score: float = 50.0,
    keep_ratio: float = 0.7,
) -> FilterResult:
    """
    Score all frames by sharpness. Move blurry frames to frames_rejected/.

    Args:
        frames_dir: Directory containing .jpg frames.
        min_blur_score: Minimum Laplacian variance to keep a frame.
        keep_ratio: Never remove more than (1 - keep_ratio) of frames.

    Returns:
        FilterResult with statistics.
    """
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return FilterResult(0, 0, 0, 0.0, 0.0, 0.0)

    scores = [(f, compute_blur_score(f)) for f in frames]
    all_scores = [s for _, s in scores]
    total = len(scores)
    min_keep = max(int(total * keep_ratio), 3)  # always keep at least 3

    # Sort by score ascending (worst first)
    scores.sort(key=lambda x: x[1])

    # Determine which frames to reject
    to_reject = []
    for frame_path, score in scores:
        if score < min_blur_score and len(scores) - len(to_reject) > min_keep:
            to_reject.append(frame_path)
        else:
            break  # scores are sorted, rest are above threshold

    # Move rejected frames
    if to_reject:
        reject_dir = frames_dir.parent / "frames_rejected"
        reject_dir.mkdir(exist_ok=True)
        for frame_path in to_reject:
            dest = reject_dir / frame_path.name
            shutil.move(str(frame_path), str(dest))
        logger.info(
            "Blur filter: rejected %d/%d frames (threshold=%.1f)",
            len(to_reject), total, min_blur_score,
        )

    kept = total - len(to_reject)
    return FilterResult(
        total_frames=total,
        kept_frames=kept,
        rejected_frames=len(to_reject),
        min_score=min(all_scores),
        max_score=max(all_scores),
        mean_score=sum(all_scores) / len(all_scores),
    )
