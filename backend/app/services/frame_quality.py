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


@dataclass
class SharpSelectResult:
    total_frames: int
    selected_frames: int
    rejected_frames: int
    bucket_size: int


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


def select_sharpest_per_bucket(frames_dir: Path, bucket_size: int = 11) -> SharpSelectResult:
    """
    Keep the sharpest frame in each fixed-size bucket, reject the rest.

    This is used for "sharp frame extraction": frames are extracted densely
    (higher FPS), then down-selected to one sharp frame per target interval.
    """
    if bucket_size <= 1:
        frames = list(frames_dir.glob("*.jpg"))
        return SharpSelectResult(len(frames), len(frames), 0, bucket_size)

    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return SharpSelectResult(0, 0, 0, bucket_size)

    # Group by optional multi-video prefix (e.g., v0_0001.jpg)
    groups: dict[str, list[Path]] = {}
    for f in frames:
        prefix = ""
        if "_" in f.stem and f.stem.startswith("v"):
            maybe_prefix = f.stem.split("_", 1)[0]
            if maybe_prefix[1:].isdigit():
                prefix = maybe_prefix + "_"
        groups.setdefault(prefix, []).append(f)

    reject_dir = frames_dir.parent / "frames_rejected_sharp"
    reject_dir.mkdir(exist_ok=True)

    total = 0
    kept = 0
    rejected = 0

    for prefix, group in groups.items():
        # Score once
        scored = [(p, compute_blur_score(p)) for p in group]
        total += len(scored)

        keep_set: set[Path] = set()
        for i in range(0, len(scored), bucket_size):
            bucket = scored[i:i + bucket_size]
            if not bucket:
                continue
            best = max(bucket, key=lambda x: x[1])[0]
            keep_set.add(best)

        # Move non-kept frames
        for path, _ in scored:
            if path not in keep_set:
                shutil.move(str(path), str(reject_dir / path.name))
                rejected += 1
            else:
                kept += 1

        # Re-number kept frames for each group to keep contiguous sequence
        kept_group = sorted(
            (p for p, _ in scored if p in keep_set),
            key=lambda p: p.name,
        )
        for idx, path in enumerate(kept_group, start=1):
            new_name = f"{prefix}{idx:04d}.jpg"
            target = path.with_name(new_name)
            if path != target:
                path.rename(target)

    logger.info(
        "Sharp-select: kept %d/%d (bucket_size=%d, rejected=%d)",
        kept, total, bucket_size, rejected,
    )
    return SharpSelectResult(
        total_frames=total,
        selected_frames=kept,
        rejected_frames=rejected,
        bucket_size=bucket_size,
    )
