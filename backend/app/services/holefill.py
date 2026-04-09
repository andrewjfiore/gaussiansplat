"""Hole detection and filling for Gaussian splat point clouds.

Detects gaps/holes in the Gaussian splat by analyzing spatial occupancy on a
voxel grid, then fills them by interpolating properties from nearby Gaussians.
Inspired by SPAG-4D's GSFix3D and Free-Range Gaussians' generative
hole-filling, implemented as a lightweight geometric approach.
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HoleCluster:
    """A contiguous group of empty voxels adjacent to occupied ones."""
    center: np.ndarray          # world-space center [x, y, z]
    voxel_count: int            # number of empty voxels in this cluster
    neighbor_indices: list[int] # indices of nearby Gaussians


@dataclass
class HolefillResult:
    """Aggregate statistics for a hole-fill run."""
    holes_detected: int = 0
    holes_filled: int = 0
    gaussians_added: int = 0
    original_count: int = 0
    final_count: int = 0
    hole_locations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "holes_detected": self.holes_detected,
            "holes_filled": self.holes_filled,
            "gaussians_added": self.gaussians_added,
            "original_count": self.original_count,
            "final_count": self.final_count,
            "hole_locations": self.hole_locations,
        }


# ---------------------------------------------------------------------------
# PLY I/O helpers  (mirrors cleanup.py patterns)
# ---------------------------------------------------------------------------

def _read_ply(path: Path) -> tuple[np.ndarray, list[str], list[np.dtype]]:
    """Read a .ply file and return (structured array, property names, dtypes)."""
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = [p.name for p in vertex.properties]
    dtypes = [vertex[n].dtype for n in names]
    count = len(vertex.data)

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


def _get_colors(data: np.ndarray) -> np.ndarray | None:
    """Extract per-Gaussian SH DC color coefficients (f_dc_0/1/2)."""
    dc_names = ["f_dc_0", "f_dc_1", "f_dc_2"]
    if all(n in data.dtype.names for n in dc_names):
        return np.column_stack([data[n] for n in dc_names])
    return None


def _get_scales(data: np.ndarray) -> np.ndarray | None:
    """Extract per-Gaussian scale (scale_0/1/2)."""
    scale_names = ["scale_0", "scale_1", "scale_2"]
    if all(n in data.dtype.names for n in scale_names):
        return np.column_stack([data[n] for n in scale_names])
    return None


def _get_rotations(data: np.ndarray) -> np.ndarray | None:
    """Extract per-Gaussian rotation quaternion (rot_0/1/2/3)."""
    rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
    if all(n in data.dtype.names for n in rot_names):
        return np.column_stack([data[n] for n in rot_names])
    return None


def _get_opacity(data: np.ndarray) -> np.ndarray | None:
    """Extract raw opacity (pre-sigmoid)."""
    if "opacity" in data.dtype.names:
        return data["opacity"].astype(np.float64)
    return None


# ---------------------------------------------------------------------------
# Hole detection
# ---------------------------------------------------------------------------

def _build_occupancy_grid(
    positions: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a 3D occupancy grid from Gaussian positions.

    Returns (grid [res, res, res] bool, bbox_min [3], voxel_size).
    """
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)

    # Add small padding to avoid edge issues
    extent = bbox_max - bbox_min
    padding = extent * 0.01
    bbox_min = bbox_min - padding
    extent = extent + 2 * padding

    voxel_size = float(extent.max() / resolution)
    if voxel_size < 1e-10:
        voxel_size = 1.0

    # Quantize positions to voxel coordinates
    voxel_coords = np.floor((positions - bbox_min) / voxel_size).astype(np.int32)
    voxel_coords = np.clip(voxel_coords, 0, resolution - 1)

    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = True

    return grid, bbox_min, voxel_size


def _find_boundary_holes(
    grid: np.ndarray,
    min_occupied_neighbors: int = 2,
) -> np.ndarray:
    """Find empty voxels adjacent to occupied voxels (6-connectivity).

    Returns Nx3 array of hole voxel coordinates.
    """
    res = grid.shape[0]

    # 6-connected neighbor offsets
    offsets = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=np.int32)

    # Count occupied neighbors for each voxel
    neighbor_count = np.zeros_like(grid, dtype=np.int32)
    for off in offsets:
        # Shift grid and add
        shifted = np.zeros_like(grid, dtype=np.int32)
        sx = slice(max(0, off[0]), min(res, res + off[0]))
        sy = slice(max(0, off[1]), min(res, res + off[1]))
        sz = slice(max(0, off[2]), min(res, res + off[2]))
        dx = slice(max(0, -off[0]), min(res, res - off[0]))
        dy = slice(max(0, -off[1]), min(res, res - off[1]))
        dz = slice(max(0, -off[2]), min(res, res - off[2]))
        shifted[dx, dy, dz] = grid[sx, sy, sz].astype(np.int32)
        neighbor_count += shifted

    # Holes: empty cells with >= min_occupied_neighbors occupied neighbors
    hole_mask = (~grid) & (neighbor_count >= min_occupied_neighbors)

    hole_coords = np.argwhere(hole_mask)
    return hole_coords


def _cluster_holes(
    hole_coords: np.ndarray,
    min_size: int = 2,
    max_size: int = 500,
) -> list[np.ndarray]:
    """Cluster connected hole voxels using connected-component labeling.

    Returns list of Nx3 arrays, one per cluster, filtered by size.
    """
    if len(hole_coords) == 0:
        return []

    from scipy import ndimage

    # Reconstruct a boolean grid of holes
    if len(hole_coords) == 0:
        return []

    max_coords = hole_coords.max(axis=0)
    shape = tuple(max_coords + 1)

    hole_grid = np.zeros(shape, dtype=bool)
    hole_grid[hole_coords[:, 0], hole_coords[:, 1], hole_coords[:, 2]] = True

    # Label connected components (6-connectivity)
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connected
    labeled, num_features = ndimage.label(hole_grid, structure=struct)

    clusters = []
    for label_id in range(1, num_features + 1):
        coords = np.argwhere(labeled == label_id)
        if min_size <= len(coords) <= max_size:
            clusters.append(coords)

    return clusters


def detect_holes(
    positions: np.ndarray,
    grid_resolution: int = 64,
    min_hole_size: int = 2,
    max_hole_size: int = 500,
) -> tuple[list[HoleCluster], np.ndarray, float]:
    """Detect holes in a Gaussian splat point cloud.

    Returns (list of HoleCluster, bbox_min, voxel_size).
    """
    from scipy.spatial import KDTree

    grid, bbox_min, voxel_size = _build_occupancy_grid(positions, grid_resolution)
    logger.info("Occupancy grid: %d^3, voxel_size=%.4f, occupied=%d/%d",
                grid_resolution, voxel_size,
                int(grid.sum()), grid_resolution ** 3)

    hole_coords = _find_boundary_holes(grid, min_occupied_neighbors=2)
    logger.info("Boundary hole voxels: %d", len(hole_coords))

    if len(hole_coords) == 0:
        return [], bbox_min, voxel_size

    clusters = _cluster_holes(hole_coords, min_size=min_hole_size,
                              max_size=max_hole_size)
    logger.info("Hole clusters after size filtering: %d", len(clusters))

    # Build KDTree for neighbor lookups
    tree = KDTree(positions)

    hole_clusters = []
    for cluster_coords in clusters:
        # Convert voxel coordinates to world space
        world_coords = cluster_coords * voxel_size + bbox_min + voxel_size / 2
        center = world_coords.mean(axis=0)

        # Find nearest Gaussians to the cluster center
        search_radius = voxel_size * max(3, len(cluster_coords) ** (1.0 / 3.0))
        neighbor_idx = tree.query_ball_point(center, search_radius)

        # If ball query finds too few, use k-nearest
        if len(neighbor_idx) < 5:
            _, neighbor_idx = tree.query(center, k=min(20, len(positions)))
            neighbor_idx = list(neighbor_idx) if np.ndim(neighbor_idx) > 0 else [neighbor_idx]

        hole_clusters.append(HoleCluster(
            center=center,
            voxel_count=len(cluster_coords),
            neighbor_indices=neighbor_idx[:50],  # cap to avoid memory issues
        ))

    return hole_clusters, bbox_min, voxel_size


# ---------------------------------------------------------------------------
# Hole filling via interpolation
# ---------------------------------------------------------------------------

def _interpolate_gaussians(
    hole: HoleCluster,
    positions: np.ndarray,
    colors: np.ndarray | None,
    scales: np.ndarray | None,
    rotations: np.ndarray | None,
    opacity: np.ndarray | None,
    voxel_size: float,
    fill_density: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate fill Gaussians for a single hole cluster.

    Returns dict mapping property name to arrays for the new Gaussians.
    """
    neighbor_idx = np.array(hole.neighbor_indices, dtype=int)
    if len(neighbor_idx) == 0:
        return {}

    # Clamp indices
    neighbor_idx = neighbor_idx[neighbor_idx < len(positions)]
    if len(neighbor_idx) == 0:
        return {}

    neighbor_pos = positions[neighbor_idx]
    center = hole.center

    # Calculate target number of fill Gaussians based on hole size and density
    # Use surrounding density as reference
    surrounding_volume = (voxel_size * 3) ** 3
    surrounding_density = len(neighbor_idx) / max(surrounding_volume, 1e-10)
    hole_volume = hole.voxel_count * (voxel_size ** 3)
    target_count = max(1, int(surrounding_density * hole_volume * fill_density))
    target_count = min(target_count, hole.voxel_count * 4)  # cap overfill

    if target_count == 0:
        return {}

    # Generate fill positions: interpolate between neighbors and hole center
    rng = np.random.default_rng(42)

    # Strategy: random points within the hole volume, biased toward center
    fill_positions = np.empty((target_count, 3), dtype=np.float32)
    hole_radius = (hole.voxel_count ** (1.0 / 3.0)) * voxel_size / 2
    for i in range(target_count):
        # Random offset from center within hole volume
        offset = rng.normal(0, hole_radius * 0.5, size=3)
        # Bias toward nearest neighbor direction
        if len(neighbor_pos) > 0:
            nearest_idx = rng.integers(0, len(neighbor_pos))
            direction = neighbor_pos[nearest_idx] - center
            blend = rng.uniform(0.1, 0.6)
            fill_positions[i] = center + offset * (1 - blend) + direction * blend
        else:
            fill_positions[i] = center + offset

    # Compute inverse-distance weights for property interpolation
    dists = np.linalg.norm(neighbor_pos - center, axis=1)
    dists = np.maximum(dists, 1e-8)
    weights = 1.0 / dists
    weights /= weights.sum()

    result = {
        "x": fill_positions[:, 0],
        "y": fill_positions[:, 1],
        "z": fill_positions[:, 2],
    }

    # Color: weighted average of neighbor colors
    if colors is not None:
        neighbor_colors = colors[neighbor_idx]
        avg_color = (weights[:, None] * neighbor_colors).sum(axis=0)
        result["f_dc_0"] = np.full(target_count, avg_color[0], dtype=np.float32)
        result["f_dc_1"] = np.full(target_count, avg_color[1], dtype=np.float32)
        result["f_dc_2"] = np.full(target_count, avg_color[2], dtype=np.float32)

    # Scale: median of neighbors, slightly larger for coverage
    if scales is not None:
        neighbor_scales = scales[neighbor_idx]
        median_scale = np.median(neighbor_scales, axis=0)
        fill_scale = median_scale * 1.2  # slightly larger for coverage
        result["scale_0"] = np.full(target_count, fill_scale[0], dtype=np.float32)
        result["scale_1"] = np.full(target_count, fill_scale[1], dtype=np.float32)
        result["scale_2"] = np.full(target_count, fill_scale[2], dtype=np.float32)

    # Opacity: 0.7 sigmoid => inverse sigmoid = log(0.7/0.3) ~ 0.847
    if opacity is not None:
        raw_opacity_val = float(np.log(0.7 / 0.3))
        result["opacity"] = np.full(target_count, raw_opacity_val, dtype=np.float32)

    # Rotation: mean quaternion of neighbors, normalized
    if rotations is not None:
        neighbor_rots = rotations[neighbor_idx]
        # Ensure quaternions are in the same hemisphere before averaging
        signs = np.sign(np.sum(neighbor_rots * neighbor_rots[0:1], axis=1))
        signs[signs == 0] = 1
        aligned_rots = neighbor_rots * signs[:, None]
        avg_rot = (weights[:, None] * aligned_rots).sum(axis=0)
        norm = np.linalg.norm(avg_rot)
        if norm > 1e-8:
            avg_rot /= norm
        else:
            avg_rot = np.array([1.0, 0.0, 0.0, 0.0])
        result["rot_0"] = np.full(target_count, avg_rot[0], dtype=np.float32)
        result["rot_1"] = np.full(target_count, avg_rot[1], dtype=np.float32)
        result["rot_2"] = np.full(target_count, avg_rot[2], dtype=np.float32)
        result["rot_3"] = np.full(target_count, avg_rot[3], dtype=np.float32)

    return result


def fill_holes(
    data: np.ndarray,
    holes: list[HoleCluster],
    voxel_size: float,
    fill_density: float = 1.0,
    on_progress: Callable[[str, float], None] | None = None,
) -> np.ndarray:
    """Generate fill Gaussians for all detected holes and append to data.

    Returns the combined structured array (original + fill).
    """
    positions = _get_positions(data)
    colors = _get_colors(data)
    scales = _get_scales(data)
    rotations = _get_rotations(data)
    opacity = _get_opacity(data)

    all_fill_props: dict[str, list[np.ndarray]] = {}
    total_added = 0

    for i, hole in enumerate(holes):
        if on_progress and i % max(1, len(holes) // 10) == 0:
            pct = 50 + (i / len(holes)) * 40
            on_progress(f"Filling hole {i + 1}/{len(holes)}...", pct)

        fill = _interpolate_gaussians(
            hole, positions, colors, scales, rotations, opacity,
            voxel_size, fill_density,
        )
        if not fill:
            continue

        count = len(fill["x"])
        total_added += count

        for key, arr in fill.items():
            if key not in all_fill_props:
                all_fill_props[key] = []
            all_fill_props[key].append(arr)

    if total_added == 0:
        return data

    # Concatenate all fill properties
    fill_concat: dict[str, np.ndarray] = {}
    for key, arrs in all_fill_props.items():
        fill_concat[key] = np.concatenate(arrs)

    # Build a structured array matching the original dtype
    fill_arr = np.zeros(total_added, dtype=data.dtype)
    for name in data.dtype.names:
        if name in fill_concat:
            fill_arr[name] = fill_concat[name].astype(data.dtype[name])
        # else: leave as zero (SH higher-order terms, etc.)

    # Append to original
    combined = np.concatenate([data, fill_arr])
    logger.info("Hole fill: added %d Gaussians (%d -> %d)",
                total_added, len(data), len(combined))

    return combined


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_holefill(
    ply_path: Path,
    grid_resolution: int = 64,
    min_hole_size: int = 2,
    max_hole_size: int = 500,
    fill_density: float = 1.0,
    on_progress: Callable[[str, float], None] | None = None,
) -> HolefillResult:
    """Run hole detection and filling on a .ply file.

    The original file is backed up as ``point_cloud_pre_holefill.ply`` in the
    same directory before the filled version is written.

    Parameters
    ----------
    ply_path:
        Path to the input .ply file.
    grid_resolution:
        Resolution of the occupancy grid (e.g. 32, 64, 128).
    min_hole_size:
        Minimum number of voxels for a hole cluster to be considered.
    max_hole_size:
        Maximum number of voxels for a hole cluster (filters out huge voids).
    fill_density:
        Multiplier for fill density (1.0 = match surrounding density).
    on_progress:
        Optional callback ``(message, percent)`` for progress reporting.

    Returns
    -------
    HolefillResult with detection and fill statistics.
    """
    def _progress(msg: str, pct: float) -> None:
        if on_progress:
            on_progress(msg, pct)

    _progress("Reading PLY file...", 0)
    data, names, dtypes = _read_ply(ply_path)
    original_count = len(data)
    logger.info("Holefill: loaded %d Gaussians from %s", original_count, ply_path)

    result = HolefillResult(original_count=original_count)

    # Step 1: Detect holes
    _progress("Analyzing spatial coverage...", 10)
    positions = _get_positions(data)
    holes, bbox_min, voxel_size = detect_holes(
        positions,
        grid_resolution=grid_resolution,
        min_hole_size=min_hole_size,
        max_hole_size=max_hole_size,
    )

    result.holes_detected = len(holes)
    result.hole_locations = [
        {"center": h.center.tolist(), "size": h.voxel_count}
        for h in holes
    ]
    logger.info("Detected %d holes", len(holes))

    if len(holes) == 0:
        _progress("No holes detected", 100)
        result.final_count = original_count
        return result

    # Step 2: Fill holes
    _progress(f"Filling {len(holes)} holes...", 50)
    combined = fill_holes(data, holes, voxel_size, fill_density, on_progress)

    result.gaussians_added = len(combined) - original_count
    result.holes_filled = sum(1 for h in holes if h.voxel_count > 0)
    result.final_count = len(combined)

    if result.gaussians_added > 0:
        # Backup original
        _progress("Backing up original PLY...", 92)
        backup_path = ply_path.parent / "point_cloud_pre_holefill.ply"
        if not backup_path.exists():
            shutil.copy2(ply_path, backup_path)
            logger.info("Backed up original to %s", backup_path)

        # Write combined result
        _progress("Writing filled PLY...", 95)
        _write_ply(ply_path, combined)

    logger.info(
        "Holefill complete: %d holes, +%d Gaussians (%d -> %d)",
        result.holes_detected, result.gaussians_added,
        result.original_count, result.final_count,
    )

    _progress("Hole filling complete", 100)
    return result
