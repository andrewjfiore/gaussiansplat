"""
sfm_quality.py — Evaluate COLMAP SfM reconstruction quality.

Reads binary model files to compute registration ratio, point count,
and mean reprojection error, then applies quality thresholds.
"""

import logging
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SfmQuality:
    registered_images: int
    total_frames: int
    registered_ratio: float
    num_3d_points: int
    mean_reprojection_error: float
    passed: bool
    failure_reason: str | None = None
    warnings: list[str] | None = None


def _count_registered_images_bin(images_bin: Path) -> int:
    """Count registered images in a COLMAP images.bin file."""
    with open(images_bin, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
    return n


def _read_points3d_stats(points3d_bin: Path) -> tuple[int, float]:
    """Return (num_points, mean_reprojection_error) from points3D.bin."""
    errors = []
    with open(points3d_bin, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(8)  # point3D_id
            f.read(24)  # xyz (3 doubles)
            f.read(3)  # rgb (3 bytes)
            (error,) = struct.unpack("<d", f.read(8))
            errors.append(error)
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.read(track_len * 8)  # track entries
    mean_err = sum(errors) / len(errors) if errors else 0.0
    return n, mean_err


def evaluate_sfm_quality(
    sparse_model_dir: Path,
    frames_dir: Path,
    min_registered_ratio: float = 0.3,
    min_3d_points: int = 100,
    max_reprojection_error: float = 4.0,
) -> SfmQuality:
    """Evaluate SfM quality and return pass/fail with metrics."""
    warnings = []

    # Count total input frames
    total_frames = sum(1 for _ in frames_dir.glob("*.jpg"))

    # Read registered images
    images_bin = sparse_model_dir / "images.bin"
    if images_bin.exists():
        registered = _count_registered_images_bin(images_bin)
    else:
        # Text fallback: count non-comment, non-empty lines, divide by 2
        images_txt = sparse_model_dir / "images.txt"
        if images_txt.exists():
            with open(images_txt) as f:
                lines = [l for l in f if not l.startswith("#") and l.strip()]
            registered = len(lines) // 2
        else:
            return SfmQuality(
                registered_images=0, total_frames=total_frames,
                registered_ratio=0.0, num_3d_points=0,
                mean_reprojection_error=0.0, passed=False,
                failure_reason="No images.bin or images.txt found",
            )

    # Read 3D point stats
    points3d_bin = sparse_model_dir / "points3D.bin"
    if points3d_bin.exists():
        num_points, mean_error = _read_points3d_stats(points3d_bin)
    else:
        num_points, mean_error = 0, 0.0

    ratio = registered / total_frames if total_frames > 0 else 0.0

    # Check thresholds
    failure_reason = None
    if ratio < min_registered_ratio:
        failure_reason = (
            f"Only {registered}/{total_frames} images registered "
            f"({ratio:.0%}), minimum is {min_registered_ratio:.0%}"
        )
    elif num_points < min_3d_points:
        failure_reason = (
            f"Only {num_points} 3D points reconstructed, minimum is {min_3d_points}"
        )

    if mean_error > max_reprojection_error:
        warnings.append(
            f"High mean reprojection error: {mean_error:.2f}px "
            f"(threshold: {max_reprojection_error:.1f}px)"
        )

    passed = failure_reason is None

    return SfmQuality(
        registered_images=registered,
        total_frames=total_frames,
        registered_ratio=ratio,
        num_3d_points=num_points,
        mean_reprojection_error=mean_error,
        passed=passed,
        failure_reason=failure_reason,
        warnings=warnings or None,
    )


def export_sparse_ply(sparse_model_dir: Path, output_path: Path) -> int:
    """Export COLMAP sparse points as a binary PLY file for preview rendering.

    Returns the number of points written.
    """
    # Import the shared COLMAP reader
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from colmap_io import read_points3d_bin, read_points3d_txt

    points3d_bin = sparse_model_dir / "points3D.bin"
    points3d_txt = sparse_model_dir / "points3D.txt"

    if points3d_bin.exists():
        xyz, rgb_f = read_points3d_bin(points3d_bin)
    elif points3d_txt.exists():
        xyz, rgb_f = read_points3d_txt(points3d_txt)
    else:
        logger.warning("No points3D file found in %s", sparse_model_dir)
        return 0

    n = len(xyz)
    if n == 0:
        return 0

    rgb_u8 = (np.clip(rgb_f, 0, 1) * 255).astype(np.uint8)

    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property uchar red\n"
        f"property uchar green\n"
        f"property uchar blue\n"
        f"end_header\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fff", xyz[i, 0], xyz[i, 1], xyz[i, 2]))
            f.write(struct.pack("<BBB", rgb_u8[i, 0], rgb_u8[i, 1], rgb_u8[i, 2]))

    logger.info("Exported sparse PLY: %d points to %s", n, output_path)
    return n
