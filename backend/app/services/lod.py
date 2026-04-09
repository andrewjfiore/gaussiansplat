"""
Level-of-Detail (LOD) generation for Gaussian splat point clouds.

After training completes, generates downsampled versions of the PLY file
so the viewer can load a coarse preview instantly and progressively refine.

LOD 0 (Preview): ~10% of Gaussians via coarse voxel grid
LOD 1 (Medium):  ~40% of Gaussians via finer voxel grid
LOD 2 (Full):    100% — the original file (symlinked/copied)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOD_LEVELS = {
    0: {"name": "preview", "target_fraction": 0.10},
    1: {"name": "medium", "target_fraction": 0.40},
    2: {"name": "full", "target_fraction": 1.0},
}

# Minimum number of Gaussians below which LOD generation is skipped
MIN_GAUSSIANS_FOR_LOD = 5000


# ---------------------------------------------------------------------------
# PLY reading / writing with plyfile
# ---------------------------------------------------------------------------

def _find_ply(output_dir: Path) -> Optional[Path]:
    """Locate the primary PLY file in the output directory."""
    # Check common locations
    candidates = [
        output_dir / "ply" / "point_cloud.ply",
        output_dir / "point_cloud.ply",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: search recursively
    plys = list(output_dir.rglob("*.ply"))
    # Filter out any existing LOD files
    plys = [p for p in plys if "_lod" not in p.stem]
    if plys:
        return plys[0]
    return None


def _read_ply_data(ply_path: Path):
    """Read a PLY file and return (PlyData, vertex element)."""
    from plyfile import PlyData
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    return plydata, vertex


def _get_positions(vertex) -> np.ndarray:
    """Extract XYZ positions from a vertex element."""
    x = np.array(vertex["x"], dtype=np.float32)
    y = np.array(vertex["y"], dtype=np.float32)
    z = np.array(vertex["z"], dtype=np.float32)
    return np.column_stack([x, y, z])


def _get_opacity(vertex) -> np.ndarray:
    """Extract opacity values. Returns ones if not present."""
    try:
        return np.array(vertex["opacity"], dtype=np.float32)
    except (ValueError, KeyError):
        return np.ones(len(vertex.data), dtype=np.float32)


def _voxel_downsample(positions: np.ndarray, opacity: np.ndarray,
                       target_fraction: float) -> np.ndarray:
    """
    Voxel-grid downsample: hash positions into voxels, keep the highest-opacity
    Gaussian per voxel. Returns indices into the original array.

    Iteratively adjusts voxel size to approximate the target fraction.
    """
    n = len(positions)
    target_count = max(1, int(n * target_fraction))

    if target_count >= n:
        return np.arange(n)

    # Compute bounding box
    pmin = positions.min(axis=0)
    pmax = positions.max(axis=0)
    extent = pmax - pmin
    max_extent = extent.max()
    if max_extent < 1e-8:
        return np.arange(min(target_count, n))

    # Binary search for the right voxel size
    lo, hi = max_extent / 1000.0, max_extent / 2.0

    best_indices = None
    for _ in range(20):
        voxel_size = (lo + hi) / 2.0
        indices = _downsample_with_voxel_size(positions, opacity, voxel_size)
        count = len(indices)

        if best_indices is None or abs(count - target_count) < abs(len(best_indices) - target_count):
            best_indices = indices

        if count < target_count:
            hi = voxel_size  # voxels too big, shrink
        else:
            lo = voxel_size  # voxels too small, enlarge

        # Close enough
        if abs(count - target_count) / target_count < 0.1:
            break

    return best_indices  # type: ignore[return-value]


def _downsample_with_voxel_size(positions: np.ndarray, opacity: np.ndarray,
                                  voxel_size: float) -> np.ndarray:
    """For a given voxel size, return indices of the highest-opacity Gaussian per voxel."""
    # Quantize to voxel grid
    voxel_indices = np.floor(positions / voxel_size).astype(np.int32)

    # Hash voxel indices for grouping
    # Use a large prime hash to reduce collisions
    h = (voxel_indices[:, 0].astype(np.int64) * 73856093
         ^ voxel_indices[:, 1].astype(np.int64) * 19349669
         ^ voxel_indices[:, 2].astype(np.int64) * 83492791)

    # Group by voxel hash, keep highest opacity per group
    unique_hashes, inverse = np.unique(h, return_inverse=True)
    n_voxels = len(unique_hashes)

    # For each voxel, find the gaussian with highest opacity
    best_idx = np.full(n_voxels, -1, dtype=np.int64)
    best_opa = np.full(n_voxels, -np.inf, dtype=np.float32)

    for i in range(len(opacity)):
        v = inverse[i]
        if opacity[i] > best_opa[v]:
            best_opa[v] = opacity[i]
            best_idx[v] = i

    return best_idx[best_idx >= 0]


def _write_ply_subset(ply_path: Path, vertex, indices: np.ndarray,
                       output_path: Path):
    """Write a subset of vertices to a new PLY file, sorted by opacity descending."""
    from plyfile import PlyData, PlyElement

    # Sort indices by opacity descending for optimal visual impact during progressive load
    opacity = _get_opacity(vertex)
    sub_opacity = opacity[indices]
    sort_order = np.argsort(-sub_opacity)
    sorted_indices = indices[sort_order]

    # Extract subset
    subset_data = vertex.data[sorted_indices]

    new_element = PlyElement.describe(subset_data, "vertex")
    PlyData([new_element], text=False).write(str(output_path))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_lod(output_dir: Path) -> dict:
    """
    Generate LOD versions of the point cloud in output_dir.

    Returns a dict with LOD info: {level: {path, count, size_bytes}}.
    """
    ply_path = _find_ply(output_dir)
    if ply_path is None:
        logger.warning("No PLY file found in %s, skipping LOD generation", output_dir)
        return {}

    logger.info("Generating LOD versions from %s", ply_path)

    plydata, vertex = _read_ply_data(ply_path)
    n_total = len(vertex.data)

    if n_total < MIN_GAUSSIANS_FOR_LOD:
        logger.info("Only %d Gaussians (< %d), skipping LOD generation",
                     n_total, MIN_GAUSSIANS_FOR_LOD)
        return {}

    positions = _get_positions(vertex)
    opacity = _get_opacity(vertex)

    # Determine output directory for LOD files (same dir as the source PLY)
    lod_dir = ply_path.parent
    results = {}

    for level, cfg in LOD_LEVELS.items():
        if level == 2:
            # Full quality - just reference the original
            results[level] = {
                "path": str(ply_path),
                "filename": ply_path.name,
                "count": n_total,
                "size_bytes": ply_path.stat().st_size,
            }
            continue

        lod_filename = f"point_cloud_lod{level}.ply"
        lod_path = lod_dir / lod_filename

        target_frac = cfg["target_fraction"]
        logger.info("Generating LOD %d (%s, %.0f%%) ...",
                     level, cfg["name"], target_frac * 100)

        indices = _voxel_downsample(positions, opacity, target_frac)
        _write_ply_subset(ply_path, vertex, indices, lod_path)

        file_size = lod_path.stat().st_size
        results[level] = {
            "path": str(lod_path),
            "filename": lod_filename,
            "count": len(indices),
            "size_bytes": file_size,
        }
        logger.info("LOD %d: %d Gaussians, %.1f MB",
                     level, len(indices), file_size / 1024 / 1024)

    logger.info("LOD generation complete")
    return results


def get_lod_info(output_dir: Path) -> dict:
    """
    Check which LOD files exist and return info about them.
    Returns {levels: [{level, name, filename, count_estimate, size_bytes, available}]}
    """
    ply_path = _find_ply(output_dir)
    if ply_path is None:
        return {"levels": [], "has_lod": False}

    lod_dir = ply_path.parent
    levels = []

    for level, cfg in LOD_LEVELS.items():
        if level == 2:
            # Full is the original file
            levels.append({
                "level": level,
                "name": cfg["name"],
                "filename": ply_path.name,
                "size_bytes": ply_path.stat().st_size,
                "available": True,
            })
        else:
            lod_filename = f"point_cloud_lod{level}.ply"
            lod_path = lod_dir / lod_filename
            if lod_path.exists():
                levels.append({
                    "level": level,
                    "name": cfg["name"],
                    "filename": lod_filename,
                    "size_bytes": lod_path.stat().st_size,
                    "available": True,
                })
            else:
                levels.append({
                    "level": level,
                    "name": cfg["name"],
                    "filename": lod_filename,
                    "size_bytes": 0,
                    "available": False,
                })

    has_lod = any(l["available"] and l["level"] < 2 for l in levels)
    return {"levels": levels, "has_lod": has_lod}
