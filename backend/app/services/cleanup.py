"""Post-training Gaussian splat cleanup service.

Removes common artifacts (floaters, large blobs, transparent splats, etc.)
from trained .ply files using a multi-stage filtering pipeline inspired by
SPAG-4D's scene filtering approach.
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FilterStats:
    """Statistics for a single cleanup filter pass."""

    name: str
    removed: int = 0


@dataclass
class CleanupStats:
    """Aggregate statistics for the full cleanup pipeline."""

    original_count: int = 0
    final_count: int = 0
    filters: list[FilterStats] = field(default_factory=list)

    @property
    def total_removed(self) -> int:
        return self.original_count - self.final_count

    @property
    def removal_pct(self) -> float:
        if self.original_count == 0:
            return 0.0
        return (self.total_removed / self.original_count) * 100

    def to_dict(self) -> dict:
        return {
            "original_count": self.original_count,
            "final_count": self.final_count,
            "total_removed": self.total_removed,
            "removal_pct": round(self.removal_pct, 2),
            "filters": [
                {"name": f.name, "removed": f.removed} for f in self.filters
            ],
        }


# ---------------------------------------------------------------------------
# PLY I/O helpers
# ---------------------------------------------------------------------------

def _read_ply(path: Path) -> tuple[np.ndarray, list[str], list[np.dtype]]:
    """Read a .ply file and return (structured array, property names, dtypes).

    Returns the vertex data as a structured numpy array along with metadata
    needed to write it back.
    """
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = [p.name for p in vertex.properties]
    dtypes = [vertex[n].dtype for n in names]
    count = len(vertex.data)

    # Build a plain structured array so downstream code can slice freely
    dt = np.dtype([(n, d) for n, d in zip(names, dtypes)])
    arr = np.empty(count, dtype=dt)
    for n in names:
        arr[n] = vertex[n]

    return arr, names, dtypes


def _write_ply(path: Path, data: np.ndarray) -> None:
    """Write a structured numpy array back to a binary .ply file."""
    from plyfile import PlyData, PlyElement

    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=False).write(str(path))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _get_positions(data: np.ndarray) -> np.ndarray:
    """Extract Nx3 position array from structured vertex data."""
    return np.column_stack([data["x"], data["y"], data["z"]])


def _get_scales(data: np.ndarray) -> np.ndarray | None:
    """Extract per-Gaussian scale if the PLY has scale_0/1/2 properties."""
    scale_names = ["scale_0", "scale_1", "scale_2"]
    if all(n in data.dtype.names for n in scale_names):
        return np.column_stack([data[n] for n in scale_names])
    return None


def _get_opacity(data: np.ndarray) -> np.ndarray | None:
    """Extract raw opacity values (pre-sigmoid) if available."""
    if "opacity" in data.dtype.names:
        return data["opacity"].astype(np.float64)
    return None


# ---------------------------------------------------------------------------
# Individual filters
# ---------------------------------------------------------------------------

def _filter_statistical_outlier(
    positions: np.ndarray,
    k: int = 50,
    std_multiplier: float = 2.0,
) -> np.ndarray:
    """Statistical Outlier Removal (SOR).

    For each point, compute the mean distance to its *k* nearest neighbours.
    Keep only points whose mean distance is within
    ``global_mean + std_multiplier * global_std``.

    Returns a boolean mask (True = keep).
    """
    from scipy.spatial import KDTree

    tree = KDTree(positions)
    # k+1 because the query includes the point itself
    dists, _ = tree.query(positions, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)  # skip self-distance

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_multiplier * global_std

    return mean_dists <= threshold


def _filter_sparse_regions(
    positions: np.ndarray,
    min_neighbors: int = 3,
    radius_multiplier: float = 1.5,
) -> np.ndarray:
    """Sparse Region Pruning.

    Remove isolated Gaussians in low-density areas.  The adaptive radius is
    computed as ``median_nn_dist * radius_multiplier`` where ``median_nn_dist``
    is the median nearest-neighbour distance across the whole point cloud.

    Returns a boolean mask (True = keep).
    """
    from scipy.spatial import KDTree

    tree = KDTree(positions)

    # Compute adaptive radius from median nearest-neighbour distance
    dists_nn, _ = tree.query(positions, k=2)  # [self, nearest]
    nn_dists = dists_nn[:, 1]
    adaptive_radius = float(np.median(nn_dists) * radius_multiplier)

    # Count neighbours within radius (subtract 1 to exclude self)
    counts = tree.query_ball_point(positions, adaptive_radius, return_length=True)
    neighbour_counts = np.asarray(counts) - 1  # exclude self

    return neighbour_counts >= min_neighbors


def _filter_large_splats(
    scales: np.ndarray,
    percentile: float = 99.0,
) -> np.ndarray:
    """Large Splat Pruning.

    Remove Gaussians whose maximum scale component exceeds the given
    percentile of the scale distribution.

    Returns a boolean mask (True = keep).
    """
    max_scales = np.max(scales, axis=1)
    threshold = np.percentile(max_scales, percentile)
    return max_scales <= threshold


def _filter_opacity(
    raw_opacity: np.ndarray,
    min_opacity: float = 0.05,
) -> np.ndarray:
    """Opacity Pruning.

    Remove near-transparent Gaussians.  Opacity values stored in the PLY are
    pre-sigmoid, so we apply sigmoid before thresholding.

    Returns a boolean mask (True = keep).
    """
    sigmoid_opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
    return sigmoid_opacity >= min_opacity


def _filter_background(
    positions: np.ndarray,
    std_multiplier: float = 3.0,
) -> np.ndarray:
    """Sky / Background Removal.

    Remove Gaussians that are very far from the scene centroid (distance >
    mean + ``std_multiplier`` * std of all distances from centroid).

    Returns a boolean mask (True = keep).
    """
    centroid = positions.mean(axis=0)
    dists = np.linalg.norm(positions - centroid, axis=1)
    threshold = dists.mean() + std_multiplier * dists.std()
    return dists <= threshold


# ---------------------------------------------------------------------------
# Main cleanup pipeline
# ---------------------------------------------------------------------------

def run_cleanup(
    ply_path: Path,
    *,
    sor_k: int = 50,
    sor_std: float = 2.0,
    sparse_min_neighbors: int = 3,
    large_splat_percentile: float = 99.0,
    opacity_threshold: float = 0.05,
    bg_std_multiplier: float = 3.0,
    on_progress: Callable[[str, float], None] | None = None,
) -> CleanupStats:
    """Run the full cleanup pipeline on a .ply file.

    The original file is backed up as ``point_cloud_original.ply`` in the same
    directory before the cleaned version is written.

    Parameters
    ----------
    ply_path:
        Path to the input .ply file.
    on_progress:
        Optional callback ``(message, percent)`` for progress reporting.

    Returns
    -------
    CleanupStats with per-filter breakdown.
    """

    def _progress(msg: str, pct: float) -> None:
        if on_progress:
            on_progress(msg, pct)

    _progress("Reading PLY file...", 0)
    data, names, dtypes = _read_ply(ply_path)
    stats = CleanupStats(original_count=len(data))
    logger.info("Cleanup: loaded %d Gaussians from %s", len(data), ply_path)

    mask = np.ones(len(data), dtype=bool)

    # --- 1. Statistical Outlier Removal ---
    _progress("Running statistical outlier removal...", 10)
    positions = _get_positions(data)
    sor_mask = _filter_statistical_outlier(positions[mask], k=sor_k, std_multiplier=sor_std)
    # Map sub-mask back to global mask
    idx = np.where(mask)[0]
    removed_before = mask.sum()
    mask[idx[~sor_mask]] = False
    removed = int(removed_before - mask.sum())
    stats.filters.append(FilterStats(name="Statistical Outlier Removal", removed=removed))
    logger.info("  SOR: removed %d", removed)

    # --- 2. Sparse Region Pruning ---
    _progress("Pruning sparse regions...", 30)
    positions = _get_positions(data)
    sparse_mask = _filter_sparse_regions(positions[mask], min_neighbors=sparse_min_neighbors)
    idx = np.where(mask)[0]
    removed_before = mask.sum()
    mask[idx[~sparse_mask]] = False
    removed = int(removed_before - mask.sum())
    stats.filters.append(FilterStats(name="Sparse Region Pruning", removed=removed))
    logger.info("  Sparse: removed %d", removed)

    # --- 3. Large Splat Pruning ---
    _progress("Pruning large splats...", 50)
    scales = _get_scales(data)
    if scales is not None:
        large_mask = _filter_large_splats(scales[mask], percentile=large_splat_percentile)
        idx = np.where(mask)[0]
        removed_before = mask.sum()
        mask[idx[~large_mask]] = False
        removed = int(removed_before - mask.sum())
    else:
        removed = 0
        logger.info("  Large splat filter skipped — no scale properties found")
    stats.filters.append(FilterStats(name="Large Splat Pruning", removed=removed))
    logger.info("  Large splats: removed %d", removed)

    # --- 4. Opacity Pruning ---
    _progress("Pruning transparent splats...", 70)
    raw_opacity = _get_opacity(data)
    if raw_opacity is not None:
        opacity_mask = _filter_opacity(raw_opacity[mask], min_opacity=opacity_threshold)
        idx = np.where(mask)[0]
        removed_before = mask.sum()
        mask[idx[~opacity_mask]] = False
        removed = int(removed_before - mask.sum())
    else:
        removed = 0
        logger.info("  Opacity filter skipped — no opacity property found")
    stats.filters.append(FilterStats(name="Opacity Pruning", removed=removed))
    logger.info("  Opacity: removed %d", removed)

    # --- 5. Sky / Background Removal ---
    _progress("Removing background floaters...", 85)
    positions = _get_positions(data)
    bg_mask = _filter_background(positions[mask], std_multiplier=bg_std_multiplier)
    idx = np.where(mask)[0]
    removed_before = mask.sum()
    mask[idx[~bg_mask]] = False
    removed = int(removed_before - mask.sum())
    stats.filters.append(FilterStats(name="Sky/Background Removal", removed=removed))
    logger.info("  Background: removed %d", removed)

    # --- Write results ---
    _progress("Writing cleaned PLY...", 95)
    cleaned = data[mask]
    stats.final_count = len(cleaned)

    # Backup original
    backup_path = ply_path.parent / "point_cloud_original.ply"
    if not backup_path.exists():
        shutil.copy2(ply_path, backup_path)
        logger.info("  Backed up original to %s", backup_path)

    _write_ply(ply_path, cleaned)
    logger.info(
        "Cleanup complete: %d -> %d (removed %d, %.1f%%)",
        stats.original_count, stats.final_count,
        stats.total_removed, stats.removal_pct,
    )

    _progress("Cleanup complete", 100)
    return stats
