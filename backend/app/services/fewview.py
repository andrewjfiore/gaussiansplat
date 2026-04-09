"""Few-View Generative Reconstruction pipeline.

Reconstructs a 3D Gaussian splat from 2-8 sparse images without requiring
video input or COLMAP.  Uses monocular depth estimation (Depth Anything V2)
plus multi-view consistency to merge per-image point clouds into a single
coherent scene.

Inspired by Free-Range Gaussians and similar sparse-view reconstruction
approaches.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .portrait import estimate_depth, write_gaussians_ply

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FewViewResult:
    ply_path: Path
    num_gaussians: int
    num_images_used: int
    num_merged_points: int
    num_fill_points: int


# ---------------------------------------------------------------------------
# 2a. Per-image depth estimation & back-projection
# ---------------------------------------------------------------------------

def _backproject_image(
    image: np.ndarray,
    depth: np.ndarray,
    stride: int = 2,
    focal_mult: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project an RGBD image to a 3D point cloud.

    Parameters
    ----------
    image : (H, W, 3) uint8
    depth : (H, W) float32
    stride : pixel stride for downsampling
    focal_mult : focal_length = image_width * focal_mult

    Returns
    -------
    points : (N, 3) float32
    colors : (N, 3) float32 in [0, 1]
    """
    h, w = depth.shape[:2]
    fx = fy = w * focal_mult
    cx, cy = w / 2.0, h / 2.0

    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    vs = vs.ravel()
    us = us.ravel()

    d = depth[vs, us]
    valid = d > 0.01

    vs = vs[valid]
    us = us[valid]
    d = d[valid]

    x = (us.astype(np.float32) - cx) * d / fx
    y = (vs.astype(np.float32) - cy) * d / fy
    z = d

    points = np.column_stack([x, y, z]).astype(np.float32)
    colors = image[vs, us].astype(np.float32) / 255.0

    return points, colors


# ---------------------------------------------------------------------------
# 2b. Camera pose estimation
# ---------------------------------------------------------------------------

def _rotation_matrix_y(angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix around Y axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float32)


def _rotation_matrix_x(angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix around X axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float32)


def _compute_camera_poses(
    num_images: int,
    arrangement: str,
    scene_radius: float = 2.0,
) -> list[np.ndarray]:
    """Compute camera extrinsic matrices (4x4) for each image.

    Parameters
    ----------
    num_images : number of input images
    arrangement : "turntable" | "forward" | "free"
    scene_radius : approximate scene radius for camera placement

    Returns
    -------
    List of 4x4 transformation matrices (world-from-camera).
    """
    poses: list[np.ndarray] = []

    if arrangement == "turntable":
        # Cameras evenly spaced around a circle looking at center
        for i in range(num_images):
            angle = 2.0 * math.pi * i / num_images
            rot = _rotation_matrix_y(angle)
            t = np.array([
                scene_radius * math.sin(angle),
                0.0,
                scene_radius * math.cos(angle),
            ], dtype=np.float32)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot
            pose[:3, 3] = t
            poses.append(pose)

    elif arrangement == "forward":
        # Cameras in a slight arc, all looking roughly the same direction
        spread = min(30.0, 60.0 / max(num_images - 1, 1))
        for i in range(num_images):
            if num_images > 1:
                frac = i / (num_images - 1)
            else:
                frac = 0.5
            angle = math.radians(-spread + 2 * spread * frac)
            rot = _rotation_matrix_y(angle)
            t = np.array([
                scene_radius * 0.3 * math.sin(angle),
                0.0,
                0.0,
            ], dtype=np.float32)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot
            pose[:3, 3] = t
            poses.append(pose)

    else:  # "free" — hemisphere distribution
        for i in range(num_images):
            if num_images > 1:
                frac = i / (num_images - 1)
            else:
                frac = 0.5
            # Fibonacci-like hemisphere placement
            phi = math.acos(1 - frac)
            theta = math.pi * (1 + math.sqrt(5)) * i
            rot = _rotation_matrix_y(theta) @ _rotation_matrix_x(phi * 0.3)
            t = np.array([
                scene_radius * math.sin(phi) * math.cos(theta),
                scene_radius * math.cos(phi) * 0.3,
                scene_radius * math.sin(phi) * math.sin(theta),
            ], dtype=np.float32)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot
            pose[:3, 3] = t
            poses.append(pose)

    return poses


# ---------------------------------------------------------------------------
# 2c. Point cloud merging with voxel-grid downsampling
# ---------------------------------------------------------------------------

def _voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-grid downsampling: keep one point per voxel cell.

    For each occupied voxel, keeps the point closest to the cell center
    and averages colors from all contributing points.

    Parameters
    ----------
    points : (N, 3) float32
    colors : (N, 3) float32
    voxel_size : size of each voxel cell

    Returns
    -------
    downsampled_points : (M, 3) float32
    downsampled_colors : (M, 3) float32
    """
    if len(points) == 0:
        return points, colors

    # Quantize to voxel grid
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # Use a dictionary to accumulate points per voxel
    voxel_map: dict[tuple[int, int, int], list[int]] = {}
    for idx in range(len(voxel_indices)):
        key = (int(voxel_indices[idx, 0]),
               int(voxel_indices[idx, 1]),
               int(voxel_indices[idx, 2]))
        if key not in voxel_map:
            voxel_map[key] = []
        voxel_map[key].append(idx)

    # For each voxel, compute mean position and mean color
    out_points = np.empty((len(voxel_map), 3), dtype=np.float32)
    out_colors = np.empty((len(voxel_map), 3), dtype=np.float32)

    for i, indices in enumerate(voxel_map.values()):
        idx_arr = np.array(indices)
        out_points[i] = points[idx_arr].mean(axis=0)
        out_colors[i] = colors[idx_arr].mean(axis=0)

    return out_points, out_colors


def _merge_point_clouds(
    per_image_points: list[np.ndarray],
    per_image_colors: list[np.ndarray],
    poses: list[np.ndarray],
    merge_resolution: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge point clouds from multiple views into a single cloud.

    1. Transform each view's points by its camera pose
    2. Concatenate all
    3. Voxel-grid downsample to merge near-duplicates

    Returns
    -------
    merged_points : (M, 3) float32
    merged_colors : (M, 3) float32
    """
    all_points = []
    all_colors = []

    for pts, cols, pose in zip(per_image_points, per_image_colors, poses):
        if len(pts) == 0:
            continue
        # Transform points by camera pose
        rot = pose[:3, :3]
        t = pose[:3, 3]
        transformed = (pts @ rot.T) + t
        all_points.append(transformed)
        all_colors.append(cols)

    if not all_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    concat_points = np.concatenate(all_points, axis=0)
    concat_colors = np.concatenate(all_colors, axis=0)

    logger.info("Concatenated %d points from %d views", len(concat_points), len(per_image_points))

    # Voxel-grid downsample to merge near-duplicates
    merged_pts, merged_cols = _voxel_downsample(concat_points, concat_colors, merge_resolution)
    logger.info("After voxel downsampling: %d points (resolution=%.4f)", len(merged_pts), merge_resolution)

    return merged_pts, merged_cols


# ---------------------------------------------------------------------------
# 2d. Gap-aware densification
# ---------------------------------------------------------------------------

def _densify_gaps(
    points: np.ndarray,
    colors: np.ndarray,
    grid_resolution: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill gaps in the merged point cloud using occupancy-grid interpolation.

    1. Build an occupancy grid
    2. Identify empty cells adjacent to occupied cells (boundary holes)
    3. Fill with interpolated position and nearest-neighbor color

    Returns
    -------
    fill_points : (K, 3) float32 — new points to add
    fill_colors : (K, 3) float32 — colors for fill points
    fill_opacities : (K,) float32 — opacities for fill points (lower than observed)
    """
    if len(points) < 10:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty(0, dtype=np.float32))

    # Compute bounding box
    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
    extent = p_max - p_min
    # Add small padding
    padding = extent * 0.05
    p_min -= padding
    p_max += padding
    extent = p_max - p_min

    # Avoid division by zero
    extent = np.maximum(extent, 1e-6)

    voxel_size = extent / grid_resolution

    # Quantize points to grid
    grid_coords = np.floor((points - p_min) / voxel_size).astype(np.int32)
    grid_coords = np.clip(grid_coords, 0, grid_resolution - 1)

    # Build occupancy grid
    occupancy = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=bool)
    # Assign a representative color to each voxel
    color_grid = np.zeros((grid_resolution, grid_resolution, grid_resolution, 3), dtype=np.float32)
    count_grid = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=np.int32)

    for i in range(len(grid_coords)):
        gx, gy, gz = grid_coords[i]
        occupancy[gx, gy, gz] = True
        color_grid[gx, gy, gz] += colors[i]
        count_grid[gx, gy, gz] += 1

    # Average colors in occupied cells
    occupied_mask = count_grid > 0
    color_grid[occupied_mask] /= count_grid[occupied_mask, np.newaxis]

    # Find boundary holes: empty cells with at least 2 occupied 6-neighbors
    fill_cells = []
    for x in range(1, grid_resolution - 1):
        for y in range(1, grid_resolution - 1):
            for z in range(1, grid_resolution - 1):
                if occupancy[x, y, z]:
                    continue
                # Count occupied neighbors
                neighbors = [
                    (x - 1, y, z), (x + 1, y, z),
                    (x, y - 1, z), (x, y + 1, z),
                    (x, y, z - 1), (x, y, z + 1),
                ]
                occ_neighbors = []
                for nx, ny, nz in neighbors:
                    if occupancy[nx, ny, nz]:
                        occ_neighbors.append((nx, ny, nz))
                if len(occ_neighbors) >= 2:
                    # Interpolate color from occupied neighbors
                    avg_color = np.mean(
                        [color_grid[nx, ny, nz] for nx, ny, nz in occ_neighbors],
                        axis=0,
                    )
                    fill_cells.append((x, y, z, avg_color))

    if not fill_cells:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty(0, dtype=np.float32))

    # Convert fill cells back to world coordinates
    fill_points = np.array(
        [[(c[0] + 0.5) * voxel_size[0] + p_min[0],
          (c[1] + 0.5) * voxel_size[1] + p_min[1],
          (c[2] + 0.5) * voxel_size[2] + p_min[2]] for c in fill_cells],
        dtype=np.float32,
    )
    fill_colors = np.array([c[3] for c in fill_cells], dtype=np.float32)
    fill_opacities = np.full(len(fill_cells), 0.7, dtype=np.float32)

    logger.info("Gap densification: added %d fill points", len(fill_points))
    return fill_points, fill_colors, fill_opacities


# ---------------------------------------------------------------------------
# 2e. Convert to Gaussians
# ---------------------------------------------------------------------------

def _points_to_gaussians(
    points: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
) -> dict:
    """Convert a point cloud to Gaussian splat parameters.

    Parameters
    ----------
    points : (N, 3) float32
    colors : (N, 3) float32 in [0, 1]
    opacities : (N,) float32

    Returns
    -------
    dict with keys: means, scales, rotations, colors, opacities
    """
    n = len(points)
    if n == 0:
        return {
            "means": np.empty((0, 3), dtype=np.float32),
            "scales": np.empty((0, 3), dtype=np.float32),
            "rotations": np.empty((0, 4), dtype=np.float32),
            "colors": np.empty((0, 3), dtype=np.float32),
            "opacities": np.empty(0, dtype=np.float32),
        }

    # Estimate scale from local point density (median of k nearest-neighbor distances)
    from scipy.spatial import KDTree
    k = min(8, n - 1)
    if k > 0:
        tree = KDTree(points)
        dists, _ = tree.query(points, k=k + 1)
        median_nn = np.median(dists[:, 1:], axis=1).astype(np.float32)
    else:
        median_nn = np.full(n, 0.01, dtype=np.float32)

    # Scale: proportional to local spacing
    base_scale = np.clip(median_nn * 0.5, 1e-5, 0.05)
    scales = np.column_stack([base_scale, base_scale, base_scale * 0.5]).astype(np.float32)

    # Rotation: identity quaternion (w, x, y, z)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 0] = 1.0

    return {
        "means": points,
        "scales": scales,
        "rotations": rotations,
        "colors": colors,
        "opacities": opacities,
    }


# ---------------------------------------------------------------------------
# 2g. Main pipeline orchestrator
# ---------------------------------------------------------------------------

def run_fewview_pipeline(
    image_paths: list[Path],
    output_dir: Path,
    arrangement: str = "turntable",
    merge_resolution: float = 0.01,
    fill_gaps: bool = True,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> FewViewResult:
    """Run the full few-view reconstruction pipeline.

    Steps:
    1. Load images (0-5%)
    2. Estimate depth for each image (5-40%)
    3. Back-project each to 3D (40-50%)
    4. Estimate/assign camera poses (50-55%)
    5. Merge point clouds (55-70%)
    6. Densify gaps (70-80%)
    7. Convert to Gaussians (80-90%)
    8. Write PLY (90-100%)

    Parameters
    ----------
    image_paths : list of paths to input images (2-8)
    output_dir : directory for all outputs
    arrangement : camera arrangement type ("turntable", "forward", "free")
    merge_resolution : voxel size for merging duplicate points
    fill_gaps : whether to fill gaps between views
    on_progress : callback(message, percent)

    Returns
    -------
    FewViewResult
    """
    import cv2
    from PIL import Image

    def _progress(msg: str, pct: float):
        logger.info("[fewview %.0f%%] %s", pct, msg)
        if on_progress:
            on_progress(msg, pct)

    num_images = len(image_paths)
    if num_images < 2:
        raise ValueError("Few-view pipeline requires at least 2 images")
    if num_images > 8:
        raise ValueError("Few-view pipeline supports at most 8 images")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load images (0-5%) ---
    _progress("Loading images...", 0)
    images: list[np.ndarray] = []
    resized_paths: list[Path] = []

    for i, img_path in enumerate(image_paths):
        pil_img = Image.open(img_path).convert("RGB")
        img = np.array(pil_img)
        h, w = img.shape[:2]

        # Resize to max 1024px
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        images.append(img)

        # Save resized image for depth estimation
        resized_path = output_dir / f"input_{i:02d}.jpg"
        cv2.imwrite(str(resized_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        resized_paths.append(resized_path)

    _progress(f"Loaded {num_images} images", 5)

    # --- 2. Estimate depth for each image (5-40%) ---
    depths: list[np.ndarray] = []
    for i, rpath in enumerate(resized_paths):
        pct = 5 + (i / num_images) * 35
        _progress(f"Estimating depth ({i + 1}/{num_images})...", pct)

        depth = estimate_depth(rpath, model_size="small")

        # Resize depth to match image if needed
        h, w = images[i].shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        depths.append(depth)

    _progress("Depth estimation complete", 40)

    # --- 3. Back-project each to 3D (40-50%) ---
    per_image_points: list[np.ndarray] = []
    per_image_colors: list[np.ndarray] = []

    for i in range(num_images):
        pct = 40 + (i / num_images) * 10
        _progress(f"Back-projecting image {i + 1}/{num_images}...", pct)

        pts, cols = _backproject_image(images[i], depths[i], stride=2)
        per_image_points.append(pts)
        per_image_colors.append(cols)
        logger.info("Image %d: %d points", i, len(pts))

    _progress("Back-projection complete", 50)

    # --- 4. Estimate camera poses (50-55%) ---
    _progress(f"Assigning camera poses ({arrangement})...", 50)

    # Estimate scene radius from the first image's point cloud
    if len(per_image_points[0]) > 0:
        scene_radius = float(np.linalg.norm(
            per_image_points[0].max(axis=0) - per_image_points[0].min(axis=0)
        )) / 2.0
    else:
        scene_radius = 2.0

    poses = _compute_camera_poses(num_images, arrangement, scene_radius)
    _progress("Camera poses assigned", 55)

    # --- 5. Merge point clouds (55-70%) ---
    _progress("Merging point clouds...", 55)
    merged_points, merged_colors = _merge_point_clouds(
        per_image_points, per_image_colors, poses,
        merge_resolution=merge_resolution,
    )
    num_merged = len(merged_points)
    _progress(f"Merged {num_merged} points", 70)

    # --- 6. Densify gaps (70-80%) ---
    num_fill = 0
    if fill_gaps and num_merged > 0:
        _progress("Filling gaps between views...", 70)
        fill_pts, fill_cols, fill_ops = _densify_gaps(merged_points, merged_colors)
        num_fill = len(fill_pts)

        if num_fill > 0:
            # Combine merged + fill
            all_points = np.concatenate([merged_points, fill_pts], axis=0)
            all_colors = np.concatenate([merged_colors, fill_cols], axis=0)
            all_opacities = np.concatenate([
                np.ones(num_merged, dtype=np.float32),  # observed points
                fill_ops,  # fill points (0.7)
            ])
        else:
            all_points = merged_points
            all_colors = merged_colors
            all_opacities = np.ones(num_merged, dtype=np.float32)
    else:
        all_points = merged_points
        all_colors = merged_colors
        all_opacities = np.ones(num_merged, dtype=np.float32)

    _progress(f"Densification complete ({num_fill} fill points)", 80)

    # --- 7. Convert to Gaussians (80-90%) ---
    _progress("Converting to Gaussians...", 80)
    gaussians = _points_to_gaussians(all_points, all_colors, all_opacities)
    num_gaussians = len(gaussians["means"])
    _progress(f"Generated {num_gaussians} Gaussians", 90)

    # --- 8. Write PLY (90-100%) ---
    _progress("Writing PLY file...", 90)
    ply_dir = output_dir / "ply"
    ply_dir.mkdir(parents=True, exist_ok=True)
    ply_path = ply_dir / "point_cloud.ply"
    write_gaussians_ply(gaussians, ply_path)

    # Also copy to output root for consistency with other pipelines
    import shutil
    output_ply = output_dir / "point_cloud.ply"
    shutil.copy2(ply_path, output_ply)

    _progress("Few-view pipeline complete!", 100)

    return FewViewResult(
        ply_path=ply_path,
        num_gaussians=num_gaussians,
        num_images_used=num_images,
        num_merged_points=num_merged,
        num_fill_points=num_fill,
    )
