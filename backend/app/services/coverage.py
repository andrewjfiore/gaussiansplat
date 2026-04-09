"""Coverage analysis for Gaussian splat point clouds.

Evaluates how well a scene is captured from different viewpoints by
projecting Gaussians into virtual camera frustums arranged around the
scene and measuring spatial density in an 8x8 grid per viewpoint.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CoverageGap:
    direction: str
    score: float
    recommendation: str


@dataclass
class CoverageResult:
    overall_score: float
    direction_scores: dict[str, float]
    gaps: list[CoverageGap]
    grid_data: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------------

def _load_ply(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Gaussian positions and opacities from a .ply file.

    Returns (positions [N,3], opacities [N]).
    """
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]

    x = np.asarray(vertex["x"], dtype=np.float32)
    y = np.asarray(vertex["y"], dtype=np.float32)
    z = np.asarray(vertex["z"], dtype=np.float32)
    positions = np.stack([x, y, z], axis=-1)

    # Opacity may be stored as "opacity" or "alpha"; fall back to 1.0
    if "opacity" in vertex.data.dtype.names:
        opacities = np.asarray(vertex["opacity"], dtype=np.float32)
    elif "alpha" in vertex.data.dtype.names:
        opacities = np.asarray(vertex["alpha"], dtype=np.float32)
    else:
        opacities = np.ones(len(x), dtype=np.float32)

    return positions, opacities


# ---------------------------------------------------------------------------
# Virtual camera helpers
# ---------------------------------------------------------------------------

def _build_camera_rig(
    center: np.ndarray,
    radius: float,
) -> list[dict[str, Any]]:
    """Create 36 virtual cameras (12 azimuth x 3 elevation)."""
    azimuths = np.arange(0, 360, 30, dtype=np.float64)  # 12 steps
    elevations = np.array([-15.0, 0.0, 30.0], dtype=np.float64)

    cameras: list[dict[str, Any]] = []
    cam_dist = radius * 1.5

    for elev_deg in elevations:
        elev = math.radians(elev_deg)
        for az_deg in azimuths:
            az = math.radians(az_deg)
            # Camera position on a sphere around center
            cx = center[0] + cam_dist * math.cos(elev) * math.sin(az)
            cy = center[1] + cam_dist * math.sin(elev)
            cz = center[2] + cam_dist * math.cos(elev) * math.cos(az)

            cam_pos = np.array([cx, cy, cz], dtype=np.float64)
            # Forward direction points toward center
            forward = center - cam_pos
            forward /= np.linalg.norm(forward) + 1e-12

            cameras.append({
                "position": cam_pos,
                "forward": forward,
                "azimuth": float(az_deg),
                "elevation": float(elev_deg),
            })

    return cameras


def _camera_axes(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute right/up/forward orthonormal basis from a forward vector."""
    world_up = np.array([0.0, 1.0, 0.0])
    # If forward is nearly parallel to world_up, use a different reference
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-12
    return right, up, forward


# ---------------------------------------------------------------------------
# Per-viewpoint coverage scoring
# ---------------------------------------------------------------------------

GRID_SIZE = 8
FOV_DEG = 60.0
DENSITY_THRESHOLD = 3  # minimum Gaussians per cell to count as "covered"


def _score_viewpoint(
    cam: dict[str, Any],
    positions: np.ndarray,
    opacities: np.ndarray,
    radius: float,
) -> tuple[float, list[list[int]]]:
    """Score a single virtual camera viewpoint.

    Returns (coverage_fraction, 8x8 grid of counts).
    """
    cam_pos = cam["position"]
    forward = cam["forward"]
    right, up, fwd = _camera_axes(forward)

    # Vector from camera to each Gaussian
    delta = positions - cam_pos  # (N, 3)

    # Depth along forward axis
    depths = delta @ fwd  # (N,)

    # Only consider points in front of camera and within a reasonable distance
    max_depth = radius * 4.0
    in_front = (depths > 0.01) & (depths < max_depth)

    if not np.any(in_front):
        grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        return 0.0, grid

    delta_f = delta[in_front]
    depths_f = depths[in_front]
    opacities_f = opacities[in_front]

    # Filter by opacity (ignore very transparent Gaussians)
    visible = opacities_f > 0.1
    if not np.any(visible):
        grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        return 0.0, grid

    delta_v = delta_f[visible]
    depths_v = depths_f[visible]

    # Project onto the image plane (normalized coordinates in [-1, 1])
    half_fov = math.tan(math.radians(FOV_DEG / 2.0))
    proj_x = (delta_v @ right) / (depths_v * half_fov + 1e-12)
    proj_y = (delta_v @ up) / (depths_v * half_fov + 1e-12)

    # Keep only points inside the frustum [-1, 1]
    in_frustum = (np.abs(proj_x) <= 1.0) & (np.abs(proj_y) <= 1.0)
    proj_x = proj_x[in_frustum]
    proj_y = proj_y[in_frustum]

    if len(proj_x) == 0:
        grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        return 0.0, grid

    # Map to grid cells [0, GRID_SIZE)
    gx = np.clip(((proj_x + 1.0) / 2.0 * GRID_SIZE).astype(int), 0, GRID_SIZE - 1)
    gy = np.clip(((proj_y + 1.0) / 2.0 * GRID_SIZE).astype(int), 0, GRID_SIZE - 1)

    # Count Gaussians per cell
    grid_arr = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    np.add.at(grid_arr, (gy, gx), 1)

    # Coverage = fraction of cells with enough density
    covered_cells = int(np.sum(grid_arr >= DENSITY_THRESHOLD))
    total_cells = GRID_SIZE * GRID_SIZE
    coverage = covered_cells / total_cells

    grid = grid_arr.tolist()
    return coverage, grid


# ---------------------------------------------------------------------------
# Cardinal direction aggregation
# ---------------------------------------------------------------------------

_DIRECTION_RANGES: dict[str, tuple[float, float] | None] = {
    "Forward": (315.0, 45.0),   # wraps around 0
    "Right": (45.0, 135.0),
    "Backward": (135.0, 225.0),
    "Left": (225.0, 315.0),
}


def _azimuth_in_range(az: float, lo: float, hi: float) -> bool:
    """Check if an azimuth falls in [lo, hi), handling wrap-around."""
    if lo > hi:
        return az >= lo or az < hi
    return lo <= az < hi


def _aggregate_directions(
    cameras: list[dict[str, Any]],
    scores: list[float],
) -> dict[str, float]:
    """Average per-camera scores into 6 cardinal directions."""
    direction_scores: dict[str, list[float]] = {d: [] for d in
        ["Forward", "Right", "Backward", "Left", "Top", "Bottom"]}

    for cam, score in zip(cameras, scores):
        az = cam["azimuth"]
        elev = cam["elevation"]

        # Horizontal direction
        for name, (lo, hi) in _DIRECTION_RANGES.items():
            if _azimuth_in_range(az, lo, hi):
                direction_scores[name].append(score)
                break

        # Vertical direction
        if elev > 15.0:
            direction_scores["Top"].append(score)
        elif elev < -5.0:
            direction_scores["Bottom"].append(score)

    result: dict[str, float] = {}
    for name, vals in direction_scores.items():
        if vals:
            result[name] = round(float(np.mean(vals)) * 100, 1)
        else:
            result[name] = 0.0
    return result


# ---------------------------------------------------------------------------
# Gap identification
# ---------------------------------------------------------------------------

_RECOMMENDATIONS = {
    "Forward": "Capture more video from the front of the scene",
    "Right": "Capture more video from the right side",
    "Backward": "Capture more video from behind the scene",
    "Left": "Capture more video from the left side",
    "Top": "Capture more video from above the scene",
    "Bottom": "Capture more video from below / lower angles",
}

GAP_THRESHOLD = 60.0  # directions below this % are flagged


def _identify_gaps(direction_scores: dict[str, float]) -> list[CoverageGap]:
    gaps: list[CoverageGap] = []
    for direction, score in direction_scores.items():
        if score < GAP_THRESHOLD:
            rec = f"{_RECOMMENDATIONS.get(direction, 'Capture more from this angle')} ({score:.0f}% coverage)"
            gaps.append(CoverageGap(direction=direction, score=score, recommendation=rec))
    # Sort by worst coverage first
    gaps.sort(key=lambda g: g.score)
    return gaps


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _find_ply(project_dir: Path) -> Path | None:
    """Locate the .ply output file for a project."""
    output_dir = project_dir / "output"
    if not output_dir.exists():
        return None

    # Common locations produced by gsplat trainer
    candidates = [
        output_dir / "point_cloud.ply",
        output_dir / "ply" / "point_cloud.ply",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback: search for any .ply
    plys = list(output_dir.rglob("*.ply"))
    return plys[0] if plys else None


def analyze_coverage(project_dir: Path) -> CoverageResult:
    """Run full coverage analysis on a project's splat output.

    Raises FileNotFoundError if no .ply file is found.
    """
    ply_path = _find_ply(project_dir)
    if ply_path is None:
        raise FileNotFoundError("No .ply file found in project output")

    logger.info("Running coverage analysis on %s", ply_path)
    positions, opacities = _load_ply(ply_path)
    logger.info("Loaded %d Gaussians from PLY", len(positions))

    # Scene bounds
    center = positions.mean(axis=0)
    extents = positions.max(axis=0) - positions.min(axis=0)
    bounding_radius = float(np.linalg.norm(extents) / 2.0)

    if bounding_radius < 1e-6:
        # Degenerate point cloud
        return CoverageResult(
            overall_score=0.0,
            direction_scores={d: 0.0 for d in
                ["Forward", "Right", "Backward", "Left", "Top", "Bottom"]},
            gaps=[CoverageGap(d, 0.0, _RECOMMENDATIONS[d])
                  for d in _RECOMMENDATIONS],
            grid_data=[],
        )

    cameras = _build_camera_rig(center, bounding_radius)

    # Score each viewpoint
    cam_scores: list[float] = []
    grid_data: list[dict[str, Any]] = []
    for cam in cameras:
        score, grid = _score_viewpoint(cam, positions, opacities, bounding_radius)
        cam_scores.append(score)
        grid_data.append({
            "azimuth": cam["azimuth"],
            "elevation": cam["elevation"],
            "score": round(score * 100, 1),
            "grid": grid,
        })

    # Aggregate into cardinal directions
    direction_scores = _aggregate_directions(cameras, cam_scores)

    # Overall score: weighted average of all directions
    if direction_scores:
        overall = float(np.mean(list(direction_scores.values())))
    else:
        overall = 0.0
    overall = round(overall, 1)

    # Identify gaps
    gaps = _identify_gaps(direction_scores)

    logger.info("Coverage analysis complete: overall=%.1f%%", overall)
    return CoverageResult(
        overall_score=overall,
        direction_scores=direction_scores,
        gaps=gaps,
        grid_data=grid_data,
    )
