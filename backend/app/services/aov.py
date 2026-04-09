"""AOV (Arbitrary Output Variable) multi-output rendering for Gaussian splats.

Generates depth, scale, opacity, and density visualizations from PLY files
using simple point-based projection with Z-buffering. No matplotlib dependency --
colormaps are implemented as numpy lookup tables.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colormap lookup tables (256 entries each, RGB)
# ---------------------------------------------------------------------------

def _make_viridis_lut() -> np.ndarray:
    """Approximate viridis colormap as a 256x3 uint8 array."""
    # Key stops: (position, R, G, B)
    stops = [
        (0.0, 68, 1, 84),
        (0.13, 71, 44, 122),
        (0.25, 59, 81, 139),
        (0.38, 44, 113, 142),
        (0.50, 33, 144, 140),
        (0.63, 39, 173, 129),
        (0.75, 92, 200, 99),
        (0.88, 170, 220, 50),
        (1.0, 253, 231, 37),
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        # Find bounding stops
        for j in range(len(stops) - 1):
            if stops[j][0] <= t <= stops[j + 1][0]:
                frac = (t - stops[j][0]) / (stops[j + 1][0] - stops[j][0] + 1e-12)
                r = int(stops[j][1] + frac * (stops[j + 1][1] - stops[j][1]))
                g = int(stops[j][2] + frac * (stops[j + 1][2] - stops[j][2]))
                b = int(stops[j][3] + frac * (stops[j + 1][3] - stops[j][3]))
                lut[i] = [r, g, b]
                break
    return lut


def _make_inferno_lut() -> np.ndarray:
    """Approximate inferno colormap (cool=small, warm=large)."""
    stops = [
        (0.0, 0, 0, 4),
        (0.14, 40, 11, 84),
        (0.29, 101, 21, 110),
        (0.43, 159, 42, 99),
        (0.57, 212, 72, 66),
        (0.71, 245, 125, 21),
        (0.86, 250, 193, 39),
        (1.0, 252, 255, 164),
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for j in range(len(stops) - 1):
            if stops[j][0] <= t <= stops[j + 1][0]:
                frac = (t - stops[j][0]) / (stops[j + 1][0] - stops[j][0] + 1e-12)
                r = int(stops[j][1] + frac * (stops[j + 1][1] - stops[j][1]))
                g = int(stops[j][2] + frac * (stops[j + 1][2] - stops[j][2]))
                b = int(stops[j][3] + frac * (stops[j + 1][3] - stops[j][3]))
                lut[i] = [r, g, b]
                break
    return lut


def _make_hot_lut() -> np.ndarray:
    """Approximate hot colormap for density heatmaps."""
    stops = [
        (0.0, 0, 0, 0),
        (0.33, 200, 0, 0),
        (0.66, 255, 165, 0),
        (1.0, 255, 255, 255),
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for j in range(len(stops) - 1):
            if stops[j][0] <= t <= stops[j + 1][0]:
                frac = (t - stops[j][0]) / (stops[j + 1][0] - stops[j][0] + 1e-12)
                r = int(stops[j][1] + frac * (stops[j + 1][1] - stops[j][1]))
                g = int(stops[j][2] + frac * (stops[j + 1][2] - stops[j][2]))
                b = int(stops[j][3] + frac * (stops[j + 1][3] - stops[j][3]))
                lut[i] = [r, g, b]
                break
    return lut


def _make_gray_lut() -> np.ndarray:
    """Grayscale with slight warm tint for opacity visualization."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        lut[i] = [i, int(i * 0.95), int(i * 0.85)]
    return lut


VIRIDIS_LUT = _make_viridis_lut()
INFERNO_LUT = _make_inferno_lut()
HOT_LUT = _make_hot_lut()
GRAY_LUT = _make_gray_lut()

# ---------------------------------------------------------------------------
# PLY reading
# ---------------------------------------------------------------------------

def _load_ply(ply_path: Path) -> dict[str, np.ndarray]:
    """Load Gaussian properties from a PLY file.

    Returns a dict with keys: positions, scales (optional), opacity (optional).
    """
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names)

    x = np.asarray(vertex["x"], dtype=np.float32)
    y = np.asarray(vertex["y"], dtype=np.float32)
    z = np.asarray(vertex["z"], dtype=np.float32)

    result: dict[str, np.ndarray] = {
        "positions": np.stack([x, y, z], axis=-1),
    }

    # Scales (stored in log-space)
    if all(f"scale_{i}" in names for i in range(3)):
        result["scales"] = np.stack(
            [np.asarray(vertex[f"scale_{i}"], dtype=np.float32) for i in range(3)],
            axis=-1,
        )

    # Opacity (stored in logit-space)
    if "opacity" in names:
        result["opacity"] = np.asarray(vertex["opacity"], dtype=np.float32)

    return result


# ---------------------------------------------------------------------------
# Camera & projection helpers
# ---------------------------------------------------------------------------

IMG_SIZE = 512
FOV_DEG = 60.0

# 6 canonical viewpoints: front, back, left, right, top, bottom
VIEWPOINTS = {
    "front":  {"position_dir": np.array([0, 0, 1], dtype=np.float64),  "up": np.array([0, -1, 0], dtype=np.float64)},
    "back":   {"position_dir": np.array([0, 0, -1], dtype=np.float64), "up": np.array([0, -1, 0], dtype=np.float64)},
    "left":   {"position_dir": np.array([-1, 0, 0], dtype=np.float64), "up": np.array([0, -1, 0], dtype=np.float64)},
    "right":  {"position_dir": np.array([1, 0, 0], dtype=np.float64),  "up": np.array([0, -1, 0], dtype=np.float64)},
    "top":    {"position_dir": np.array([0, -1, 0], dtype=np.float64), "up": np.array([0, 0, 1], dtype=np.float64)},
    "bottom": {"position_dir": np.array([0, 1, 0], dtype=np.float64),  "up": np.array([0, 0, -1], dtype=np.float64)},
}


def _build_view_matrix(
    cam_pos: np.ndarray, center: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """Build a 4x4 view matrix (world -> camera)."""
    forward = center - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    true_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ cam_pos
    return view


def _project_points(
    positions: np.ndarray,
    cam_pos: np.ndarray,
    center: np.ndarray,
    up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points to 2D pixel coordinates.

    Returns (pixel_coords [N,2], depths [N], mask [N] for valid points).
    """
    view = _build_view_matrix(cam_pos, center, up)
    n = len(positions)

    # Transform to camera space
    ones = np.ones((n, 1), dtype=np.float64)
    pts_h = np.hstack([positions.astype(np.float64), ones])  # Nx4
    cam_pts = (view @ pts_h.T).T  # Nx4

    # Camera space: x-right, y-up, z points behind camera (negative forward)
    depths = -cam_pts[:, 2]  # positive depths are in front

    # Perspective projection
    half_fov = math.tan(math.radians(FOV_DEG / 2.0))
    proj_x = cam_pts[:, 0] / (depths * half_fov + 1e-12)
    proj_y = cam_pts[:, 1] / (depths * half_fov + 1e-12)

    # Valid: in front of camera and within frustum
    valid = (depths > 0.01) & (np.abs(proj_x) <= 1.0) & (np.abs(proj_y) <= 1.0)

    # Map NDC [-1,1] to pixel [0, IMG_SIZE)
    px = ((proj_x + 1.0) / 2.0 * (IMG_SIZE - 1)).astype(np.int32)
    py = ((1.0 - (proj_y + 1.0) / 2.0) * (IMG_SIZE - 1)).astype(np.int32)
    pixel_coords = np.stack([px, py], axis=-1)

    return pixel_coords, depths, valid


def _render_aov_image(
    positions: np.ndarray,
    values: np.ndarray,
    cam_pos: np.ndarray,
    center: np.ndarray,
    up: np.ndarray,
    lut: np.ndarray,
    point_size: int = 2,
) -> Image.Image:
    """Render an AOV visualization from a single viewpoint.

    positions: Nx3 world positions
    values: N float values normalized to [0, 1]
    lut: 256x3 uint8 colormap lookup table
    """
    pixel_coords, depths, valid = _project_points(positions, cam_pos, center, up)

    # Filter to valid points
    px = pixel_coords[valid, 0]
    py = pixel_coords[valid, 1]
    d = depths[valid]
    v = values[valid]

    if len(px) == 0:
        # Empty view
        return Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

    # Sort by depth (far to near) so near points overwrite far ones (painter's algo)
    order = np.argsort(-d)
    px = px[order]
    py = py[order]
    v = v[order]

    # Create image
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # Map values to colormap indices
    indices = np.clip((v * 255).astype(np.int32), 0, 255)
    colors = lut[indices]

    # Render points with size > 1 for better visibility
    half = point_size // 2
    for i in range(len(px)):
        x, y = int(px[i]), int(py[i])
        x0 = max(0, x - half)
        x1 = min(IMG_SIZE, x + half + 1)
        y0 = max(0, y - half)
        y1 = min(IMG_SIZE, y + half + 1)
        img[y0:y1, x0:x1] = colors[i]

    return Image.fromarray(img)


# ---------------------------------------------------------------------------
# AOV channel generators
# ---------------------------------------------------------------------------

def _compute_depth_values(positions: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Compute normalized depth (distance from centroid) for each Gaussian."""
    dists = np.linalg.norm(positions - center, axis=1).astype(np.float32)
    dmin, dmax = dists.min(), dists.max()
    if dmax - dmin < 1e-6:
        return np.zeros(len(dists), dtype=np.float32)
    return (dists - dmin) / (dmax - dmin)


def _compute_scale_values(scales_log: np.ndarray) -> np.ndarray:
    """Compute normalized max scale per Gaussian (from log-space scales)."""
    scales = np.exp(scales_log)
    max_scale = np.max(scales, axis=1)
    smin, smax = max_scale.min(), max_scale.max()
    if smax - smin < 1e-12:
        return np.zeros(len(max_scale), dtype=np.float32)
    return ((max_scale - smin) / (smax - smin)).astype(np.float32)


def _compute_opacity_values(raw_opacity: np.ndarray) -> np.ndarray:
    """Compute normalized opacity (sigmoid of logit-space values)."""
    sigmoid = (1.0 / (1.0 + np.exp(-raw_opacity.astype(np.float64)))).astype(np.float32)
    return sigmoid  # already in [0, 1]


# ---------------------------------------------------------------------------
# Density heatmap
# ---------------------------------------------------------------------------

def _render_density_heatmap(
    positions: np.ndarray,
    plane: str,
    grid_res: int = 128,
) -> Image.Image:
    """Render a density heatmap for a 2D projection plane (XY, XZ, YZ)."""
    axis_map = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
    ax0, ax1 = axis_map[plane]

    p0 = positions[:, ax0]
    p1 = positions[:, ax1]

    # Compute bounds with small padding
    pad = 0.05
    range0 = (p0.min() - pad * (p0.max() - p0.min() + 1e-6),
              p0.max() + pad * (p0.max() - p0.min() + 1e-6))
    range1 = (p1.min() - pad * (p1.max() - p1.min() + 1e-6),
              p1.max() + pad * (p1.max() - p1.min() + 1e-6))

    # 2D histogram
    hist, _, _ = np.histogram2d(
        p0, p1,
        bins=grid_res,
        range=[range0, range1],
    )

    # Normalize with log scale for better visibility
    hist = np.log1p(hist)
    hmax = hist.max()
    if hmax > 0:
        hist = hist / hmax
    else:
        hist = np.zeros_like(hist)

    # Apply hot colormap
    indices = np.clip((hist * 255).astype(np.int32), 0, 255)
    img_data = HOT_LUT[indices]  # (grid_res, grid_res, 3)

    # Transpose so axis 1 is horizontal (and flip vertically for image convention)
    img_data = np.flipud(img_data.transpose(1, 0, 2))

    # Resize to IMG_SIZE
    img = Image.fromarray(img_data.astype(np.uint8))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return img


# ---------------------------------------------------------------------------
# Scene statistics
# ---------------------------------------------------------------------------

@dataclass
class SceneStatistics:
    gaussian_count: int = 0
    scale_mean: float = 0.0
    scale_median: float = 0.0
    scale_std: float = 0.0
    opacity_mean: float = 0.0
    opacity_median: float = 0.0
    opacity_std: float = 0.0
    bbox_min: list[float] = field(default_factory=list)
    bbox_max: list[float] = field(default_factory=list)
    bbox_extent: list[float] = field(default_factory=list)
    density_mean: float = 0.0
    density_max: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "gaussian_count": self.gaussian_count,
            "scale_mean": round(self.scale_mean, 6),
            "scale_median": round(self.scale_median, 6),
            "scale_std": round(self.scale_std, 6),
            "opacity_mean": round(self.opacity_mean, 4),
            "opacity_median": round(self.opacity_median, 4),
            "opacity_std": round(self.opacity_std, 4),
            "bbox_min": [round(v, 4) for v in self.bbox_min],
            "bbox_max": [round(v, 4) for v in self.bbox_max],
            "bbox_extent": [round(v, 4) for v in self.bbox_extent],
            "density_mean": round(self.density_mean, 2),
            "density_max": round(self.density_max, 2),
        }


def _compute_stats(data: dict[str, np.ndarray]) -> SceneStatistics:
    """Compute scene statistics from loaded PLY data."""
    positions = data["positions"]
    stats = SceneStatistics(gaussian_count=len(positions))

    # Bounding box
    bbox_min = positions.min(axis=0).tolist()
    bbox_max = positions.max(axis=0).tolist()
    stats.bbox_min = bbox_min
    stats.bbox_max = bbox_max
    stats.bbox_extent = [mx - mn for mn, mx in zip(bbox_min, bbox_max)]

    # Scale statistics
    if "scales" in data:
        scales = np.exp(data["scales"])
        max_scales = np.max(scales, axis=1)
        stats.scale_mean = float(np.mean(max_scales))
        stats.scale_median = float(np.median(max_scales))
        stats.scale_std = float(np.std(max_scales))

    # Opacity statistics
    if "opacity" in data:
        sigmoid_op = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))
        stats.opacity_mean = float(np.mean(sigmoid_op))
        stats.opacity_median = float(np.median(sigmoid_op))
        stats.opacity_std = float(np.std(sigmoid_op))

    # Density (using a coarse grid)
    grid_res = 32
    extent = positions.max(axis=0) - positions.min(axis=0)
    cell_size = extent / grid_res
    cell_size = np.where(cell_size < 1e-6, 1.0, cell_size)
    grid_idx = ((positions - positions.min(axis=0)) / cell_size).astype(np.int32)
    grid_idx = np.clip(grid_idx, 0, grid_res - 1)

    # Count per cell using linear indexing
    linear = grid_idx[:, 0] * grid_res * grid_res + grid_idx[:, 1] * grid_res + grid_idx[:, 2]
    counts = np.bincount(linear, minlength=grid_res ** 3)
    # Only consider occupied cells
    occupied = counts[counts > 0]
    if len(occupied) > 0:
        stats.density_mean = float(np.mean(occupied))
        stats.density_max = float(np.max(occupied))

    return stats


# ---------------------------------------------------------------------------
# Find PLY file
# ---------------------------------------------------------------------------

def _find_ply(project_dir: Path) -> Path | None:
    """Locate the .ply output file for a project."""
    output_dir = project_dir / "output"
    if not output_dir.exists():
        return None
    candidates = [
        output_dir / "point_cloud.ply",
        output_dir / "ply" / "point_cloud.ply",
    ]
    for c in candidates:
        if c.exists():
            return c
    plys = list(output_dir.rglob("*.ply"))
    return plys[0] if plys else None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_aov(
    project_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate all AOV visualizations for a project.

    Parameters
    ----------
    project_dir : Path
        The project root directory.
    output_dir : Path, optional
        Where to save images. Defaults to project_dir/output/aov/.

    Returns
    -------
    dict with keys: images (list of image metadata), stats (scene statistics).
    """
    ply_path = _find_ply(project_dir)
    if ply_path is None:
        raise FileNotFoundError("No .ply file found in project output")

    if output_dir is None:
        output_dir = project_dir / "output" / "aov"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating AOV visualizations from %s", ply_path)
    data = _load_ply(ply_path)
    positions = data["positions"]
    logger.info("Loaded %d Gaussians", len(positions))

    # Scene geometry
    center = positions.mean(axis=0).astype(np.float64)
    extents = positions.max(axis=0) - positions.min(axis=0)
    radius = float(np.linalg.norm(extents) / 2.0)
    cam_dist = radius * 2.0

    # Adaptive point size based on Gaussian count
    if len(positions) > 500_000:
        point_size = 1
    elif len(positions) > 100_000:
        point_size = 2
    else:
        point_size = 3

    images: list[dict[str, str]] = []

    # ---- Depth visualization ----
    depth_values = _compute_depth_values(positions, center)
    for direction, vp in VIEWPOINTS.items():
        cam_pos = center + vp["position_dir"] * cam_dist
        img = _render_aov_image(
            positions, depth_values, cam_pos, center, vp["up"],
            VIRIDIS_LUT, point_size=point_size,
        )
        fname = f"aov_depth_{direction}.png"
        img.save(output_dir / fname)
        images.append({"filename": fname, "channel": "depth", "direction": direction})
    logger.info("  Depth: 6 views rendered")

    # ---- Scale visualization ----
    if "scales" in data:
        scale_values = _compute_scale_values(data["scales"])
        for direction, vp in VIEWPOINTS.items():
            cam_pos = center + vp["position_dir"] * cam_dist
            img = _render_aov_image(
                positions, scale_values, cam_pos, center, vp["up"],
                INFERNO_LUT, point_size=point_size,
            )
            fname = f"aov_scale_{direction}.png"
            img.save(output_dir / fname)
            images.append({"filename": fname, "channel": "scale", "direction": direction})
        logger.info("  Scale: 6 views rendered")
    else:
        logger.info("  Scale: skipped (no scale properties)")

    # ---- Opacity visualization ----
    if "opacity" in data:
        opacity_values = _compute_opacity_values(data["opacity"])
        for direction, vp in VIEWPOINTS.items():
            cam_pos = center + vp["position_dir"] * cam_dist
            img = _render_aov_image(
                positions, opacity_values, cam_pos, center, vp["up"],
                GRAY_LUT, point_size=point_size,
            )
            fname = f"aov_opacity_{direction}.png"
            img.save(output_dir / fname)
            images.append({"filename": fname, "channel": "opacity", "direction": direction})
        logger.info("  Opacity: 6 views rendered")
    else:
        logger.info("  Opacity: skipped (no opacity property)")

    # ---- Density heatmaps ----
    for plane in ["XY", "XZ", "YZ"]:
        img = _render_density_heatmap(positions, plane)
        fname = f"aov_density_{plane}.png"
        img.save(output_dir / fname)
        images.append({"filename": fname, "channel": "density", "direction": plane})
    logger.info("  Density: 3 planes rendered")

    # ---- Statistics ----
    stats = _compute_stats(data)

    logger.info("AOV generation complete: %d images", len(images))
    return {
        "images": images,
        "stats": stats.to_dict(),
    }
