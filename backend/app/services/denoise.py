"""
denoise.py — Statistical outlier removal for COLMAP point clouds.

Applies open3d statistical outlier removal to clean up noise before training.
Strength levels:
  off        → no-op, returns input path
  light      → std_ratio=3.0 (mild removal)
  medium     → std_ratio=2.0 (recommended)
  aggressive → std_ratio=1.5 (strong removal)
"""

import logging
import struct
from pathlib import Path
from typing import Tuple

import numpy as np

log = logging.getLogger(__name__)

_STD_RATIO = {
    "off":        None,
    "light":      3.0,
    "medium":     2.0,
    "aggressive": 1.5,
}


def _read_points3d_bin(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        xyz = np.zeros((n, 3), dtype=np.float64)
        rgb = np.zeros((n, 3), dtype=np.uint8)
        for i in range(n):
            f.read(8)  # point3d_id
            xyz[i] = struct.unpack("<3d", f.read(24))
            rgb[i] = struct.unpack("<3B", f.read(3))
            (n_err,) = struct.unpack("<Q", f.read(8))
            f.read(8 + 8 * n_err)  # error + track
    return xyz, rgb


def _write_points3d_bin(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    n = len(xyz)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n))
        for i in range(n):
            f.write(struct.pack("<Q", i + 1))          # point3d_id (1-indexed)
            f.write(struct.pack("<3d", *xyz[i]))
            f.write(struct.pack("<3B", *rgb[i]))
            f.write(struct.pack("<d", 0.0))            # error
            f.write(struct.pack("<Q", 0))              # empty track


def denoise_point_cloud(
    points3d_path: Path,
    output_path: Path,
    strength: str = "medium",
) -> Path:
    """
    Apply statistical outlier removal to a COLMAP points3D.bin file.
    Returns the output path (which equals points3d_path when strength="off").
    """
    std_ratio = _STD_RATIO.get(strength)
    if std_ratio is None:
        log.info("Denoising: strength=off, skipping")
        return points3d_path

    try:
        import open3d as o3d
    except ImportError:
        log.warning("open3d not installed; skipping point cloud denoising")
        return points3d_path

    xyz, rgb = _read_points3d_bin(points3d_path)
    n_before = len(xyz)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

    pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
    n_after = len(pcd_clean.points)
    removed = n_before - n_after

    log.info(
        "Denoising (%s, std_ratio=%.1f): removed %d / %d points (%.1f%%)",
        strength, std_ratio, removed, n_before,
        100 * removed / n_before if n_before else 0,
    )

    clean_xyz = np.asarray(pcd_clean.points)
    clean_rgb = (np.asarray(pcd_clean.colors) * 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_points3d_bin(output_path, clean_xyz, clean_rgb)
    return output_path
