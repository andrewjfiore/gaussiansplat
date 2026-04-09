"""Portrait Mode pipeline: single photo -> 3D Gaussian splat.

Inspired by AvatarPointillist (single photo -> 3D avatar) and SPAG-4D
(depth -> Gaussians). Uses monocular depth estimation + planar
back-projection to create an initial point cloud from a single portrait
image.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded depth model singleton
# ---------------------------------------------------------------------------

_depth_pipeline = None
_depth_model_name: str | None = None


def _get_depth_pipeline(model_size: str = "small"):
    """Lazy-load the depth estimation model on first use."""
    global _depth_pipeline, _depth_model_name

    model_map = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base": "depth-anything/Depth-Anything-V2-Base-hf",
    }
    model_id = model_map.get(model_size, model_map["small"])

    if _depth_pipeline is not None and _depth_model_name == model_id:
        return _depth_pipeline

    import torch
    from transformers import pipeline as hf_pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading depth model %s on %s (first call may download weights)", model_id, device)
    _depth_pipeline = hf_pipeline(
        "depth-estimation",
        model=model_id,
        device=device,
    )
    _depth_model_name = model_id
    return _depth_pipeline


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PortraitResult:
    ply_path: Path
    num_gaussians: int
    depth_map_path: Path
    novel_view_count: int


# ---------------------------------------------------------------------------
# 2a. Depth Estimation
# ---------------------------------------------------------------------------


def estimate_depth(image_path: Path, model_size: str = "small") -> np.ndarray:
    """Run monocular depth estimation and return a float32 depth map.

    The returned array has shape (H, W) with values in an arbitrary
    relative scale (higher = farther).
    """
    from PIL import Image

    pipe = _get_depth_pipeline(model_size)
    image = Image.open(image_path).convert("RGB")

    result = pipe(image)
    depth = result["depth"]  # PIL Image

    # Convert to numpy float32
    depth_arr = np.array(depth, dtype=np.float32)

    # Normalise to 0..1 range
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max - d_min > 1e-6:
        depth_arr = (depth_arr - d_min) / (d_max - d_min)
    else:
        depth_arr = np.zeros_like(depth_arr)

    # Invert: depth-anything returns disparity (close=high), we want
    # depth (close=low)
    depth_arr = 1.0 - depth_arr
    # Map to a reasonable depth range (e.g., 0.3 to 3.0 metres)
    depth_arr = depth_arr * 2.7 + 0.3

    return depth_arr


# ---------------------------------------------------------------------------
# 2b. Subject Segmentation
# ---------------------------------------------------------------------------


def segment_subject(image: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Create a binary mask isolating the primary subject.

    Strategy: the subject is the foreground (closest depth region).
    We threshold at the median depth and apply morphological cleanup.

    Parameters
    ----------
    image : np.ndarray (H, W, 3) uint8
    depth : np.ndarray (H, W) float32

    Returns
    -------
    mask : np.ndarray (H, W) bool
    """
    import cv2

    median_d = float(np.median(depth))
    # Foreground = pixels closer than median
    fg_mask = (depth < median_d).astype(np.uint8)

    # Morphological close to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Morphological open to remove small noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small)

    # Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    if num_labels > 1:
        # Label 0 is background
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fg_mask = (labels == largest).astype(np.uint8)

    return fg_mask.astype(bool)


# ---------------------------------------------------------------------------
# 2c. Depth-to-Gaussian Conversion (SPAG-4D inspired)
# ---------------------------------------------------------------------------


def depth_to_gaussians(
    image: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    stride: int = 2,
    focal_mult: float = 0.8,
    include_background: bool = False,
) -> dict:
    """Convert depth map + RGB image into 3D Gaussians via planar
    back-projection.

    Parameters
    ----------
    image : (H, W, 3) uint8
    depth : (H, W) float32
    mask  : (H, W) bool — subject mask
    stride : pixel stride for downsampling
    focal_mult : focal length = image_width * focal_mult
    include_background : if True, include background with lower opacity

    Returns
    -------
    dict with keys: means, scales, rotations, colors, opacities
    """
    h, w = depth.shape[:2]
    fx = fy = w * focal_mult
    cx, cy = w / 2.0, h / 2.0

    # Build grid of pixel coordinates at the given stride
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    vs = vs.ravel()
    us = us.ravel()

    d = depth[vs, us]
    m = mask[vs, us]

    if include_background:
        valid = d > 0.01
    else:
        valid = m & (d > 0.01)

    vs = vs[valid]
    us = us[valid]
    d = d[valid]

    # Planar back-projection (pinhole camera model)
    x = (us.astype(np.float32) - cx) * d / fx
    y = (vs.astype(np.float32) - cy) * d / fy
    z = d

    means = np.column_stack([x, y, z]).astype(np.float32)

    # Colors: direct from image (sRGB, 0-1 range)
    colors = image[vs, us].astype(np.float32) / 255.0

    # Scale: proportional to depth * pixel spacing
    pixel_spacing = stride / fx
    base_scale = d * pixel_spacing
    # Clamp to reasonable range
    base_scale = np.clip(base_scale, 1e-5, 0.05)
    scales = np.column_stack([base_scale, base_scale, base_scale * 0.3]).astype(np.float32)

    # Opacity: 1.0 for subject pixels, lower for background
    opacities = np.ones(len(means), dtype=np.float32)
    if include_background:
        bg_idx = ~mask[vs, us]
        opacities[bg_idx] = 0.3

    # Rotation: identity quaternion (w, x, y, z)
    rotations = np.zeros((len(means), 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # w = 1

    n = len(means)
    logger.info("Generated %d Gaussians (stride=%d, focal_mult=%.2f)", n, stride, focal_mult)

    return {
        "means": means,
        "scales": scales,
        "rotations": rotations,
        "colors": colors,
        "opacities": opacities,
    }


# ---------------------------------------------------------------------------
# 2d. Synthetic Novel View Generation
# ---------------------------------------------------------------------------


def generate_novel_views(
    means: np.ndarray,
    colors: np.ndarray,
    image_shape: tuple[int, int],
    num_views: int = 6,
) -> list[dict]:
    """Generate synthetic novel views by rotating the point cloud.

    Uses simple Z-buffer rendering with yaw rotations.

    Parameters
    ----------
    means  : (N, 3) float32 — 3D positions
    colors : (N, 3) float32 — RGB in [0, 1]
    image_shape : (H, W)
    num_views : number of views to generate

    Returns
    -------
    List of dicts with 'image' (H,W,3 uint8) and 'camera_pose' (4x4 float32).
    """
    h, w = image_shape

    # Generate rotation angles: spread across +-30 degrees yaw
    angles_deg = np.linspace(-30, 30, num_views)
    views = []

    # Compute centroid for rotation
    centroid = means.mean(axis=0)

    for angle_deg in angles_deg:
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Rotation matrix around Y axis
        rot = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a],
        ], dtype=np.float32)

        # Rotate points around centroid
        pts = (means - centroid) @ rot.T + centroid

        # Simple pinhole projection
        fx = fy = w * 0.8
        cx, cy = w / 2.0, h / 2.0

        # Only project points with positive z
        valid = pts[:, 2] > 0.01
        pts_v = pts[valid]
        col_v = colors[valid]

        u = (pts_v[:, 0] * fx / pts_v[:, 2] + cx).astype(np.int32)
        v = (pts_v[:, 1] * fy / pts_v[:, 2] + cy).astype(np.int32)

        # Clip to image bounds
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[in_bounds]
        v = v[in_bounds]
        z = pts_v[in_bounds, 2]
        c = col_v[in_bounds]

        # Z-buffer rendering: keep closest point per pixel
        render = np.zeros((h, w, 3), dtype=np.float32)
        zbuf = np.full((h, w), np.inf, dtype=np.float32)

        # Sort by depth (far to near) so closer points overwrite
        order = np.argsort(-z)
        for idx in order:
            pu, pv, pz = u[idx], v[idx], z[idx]
            if pz < zbuf[pv, pu]:
                zbuf[pv, pu] = pz
                render[pv, pu] = c[idx]

        # Convert to uint8
        render_img = (np.clip(render, 0, 1) * 255).astype(np.uint8)

        # Build camera pose (4x4 extrinsic matrix)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot

        views.append({
            "image": render_img,
            "camera_pose": pose,
            "angle_deg": float(angle_deg),
        })

    return views


# ---------------------------------------------------------------------------
# 2e. Write Initial PLY (standard 3DGS format)
# ---------------------------------------------------------------------------

# SH DC coefficient scaling factor
_SH_C0 = 0.28209479177387814


def write_gaussians_ply(gaussians: dict, output_path: Path) -> None:
    """Write Gaussians to a PLY file in the standard 3DGS format.

    Expected format for @mkkellogg/gaussian-splats-3d:
    - x, y, z (position)
    - nx, ny, nz (normals, set to 0)
    - f_dc_0, f_dc_1, f_dc_2 (DC SH coefficients)
    - f_rest_0 .. f_rest_44 (higher-order SH, set to 0)
    - opacity (logit-space)
    - scale_0, scale_1, scale_2 (log-space)
    - rot_0, rot_1, rot_2, rot_3 (quaternion)
    """
    means = gaussians["means"]
    colors = gaussians["colors"]
    scales = gaussians["scales"]
    opacities = gaussians["opacities"]
    rotations = gaussians["rotations"]
    n = len(means)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert colors (sRGB 0-1) to SH DC coefficients
    sh_dc = (colors - 0.5) / _SH_C0

    # Convert scales to log-space
    log_scales = np.log(np.clip(scales, 1e-7, None))

    # Convert opacities to logit-space
    eps = 1e-6
    clamped_op = np.clip(opacities, eps, 1.0 - eps)
    logit_opacities = np.log(clamped_op / (1.0 - clamped_op))

    # Build property list
    properties = [
        "x", "y", "z",
        "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    for i in range(45):
        properties.append(f"f_rest_{i}")
    properties.extend(["opacity", "scale_0", "scale_1", "scale_2"])
    properties.extend(["rot_0", "rot_1", "rot_2", "rot_3"])

    # Write binary PLY
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
    ]
    for prop in properties:
        header_lines.append(f"property float {prop}")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))

        # Pre-allocate zeros for normals and f_rest
        zeros_3 = np.zeros(3, dtype=np.float32)
        zeros_45 = np.zeros(45, dtype=np.float32)

        for i in range(n):
            # Position
            f.write(struct.pack("<3f", *means[i]))
            # Normals (zero)
            f.write(struct.pack("<3f", *zeros_3))
            # SH DC coefficients
            f.write(struct.pack("<3f", *sh_dc[i]))
            # SH rest (45 zeros)
            f.write(zeros_45.tobytes())
            # Opacity
            f.write(struct.pack("<f", logit_opacities[i]))
            # Scales (log-space)
            f.write(struct.pack("<3f", *log_scales[i]))
            # Rotation (quaternion)
            f.write(struct.pack("<4f", *rotations[i]))

    logger.info("Wrote %d Gaussians to %s (%.1f MB)",
                n, output_path, output_path.stat().st_size / 1024 / 1024)


# ---------------------------------------------------------------------------
# 2g. Save depth map visualisation
# ---------------------------------------------------------------------------


def save_depth_preview(depth: np.ndarray, output_path: Path) -> None:
    """Save a colourised depth map as a PNG preview image."""
    import cv2

    # Normalise to 0..255
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)

    # Apply inferno-like colourmap
    coloured = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), coloured)
    logger.info("Saved depth preview to %s", output_path)


# ---------------------------------------------------------------------------
# 2f. Main Pipeline Orchestrator
# ---------------------------------------------------------------------------


def run_portrait_pipeline(
    image_path: Path,
    output_dir: Path,
    *,
    stride: int = 2,
    focal_multiplier: float = 0.8,
    num_novel_views: int = 6,
    include_background: bool = False,
    depth_model: str = "small",
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> PortraitResult:
    """Run the full portrait-to-Gaussian pipeline.

    Steps:
    1. Load image
    2. Estimate depth (0-25%)
    3. Segment subject (25-35%)
    4. Convert to Gaussians (35-50%)
    5. Generate novel views (50-70%)
    6. Write initial PLY (70-80%)
    7. Save novel view images + cameras (80-100%)

    Parameters
    ----------
    image_path : Path to source portrait image
    output_dir : Directory for all outputs
    stride : pixel stride for downsampling Gaussians
    focal_multiplier : focal_length = image_width * this value
    num_novel_views : number of synthetic views to generate
    include_background : keep background Gaussians at lower opacity
    depth_model : "small" or "base" model size
    on_progress : callback(message, percent)

    Returns
    -------
    PortraitResult
    """
    import cv2
    from PIL import Image

    def _progress(msg: str, pct: float):
        logger.info("[portrait %.0f%%] %s", pct, msg)
        if on_progress:
            on_progress(msg, pct)

    # --- 1. Load image ---
    _progress("Loading image...", 0)
    pil_image = Image.open(image_path).convert("RGB")
    image = np.array(pil_image)
    h, w = image.shape[:2]

    # Limit resolution to keep Gaussian count manageable
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
        logger.info("Resized image to %dx%d", w, h)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Estimate depth (0-25%) ---
    _progress("Estimating depth...", 5)

    # Save resized image for depth estimation
    resized_path = output_dir / "input_resized.jpg"
    cv2.imwrite(str(resized_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    depth = estimate_depth(resized_path, model_size=depth_model)

    # Resize depth to match image dimensions if needed
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    _progress("Depth estimation complete", 25)

    # Save depth preview
    depth_preview_path = output_dir / "depth_preview.png"
    save_depth_preview(depth, depth_preview_path)

    # --- 3. Segment subject (25-35%) ---
    _progress("Segmenting subject...", 25)
    mask = segment_subject(image, depth)
    subject_pixels = int(mask.sum())
    total_pixels = h * w
    _progress(
        f"Segmentation complete: {subject_pixels}/{total_pixels} pixels "
        f"({subject_pixels / total_pixels * 100:.1f}% subject)",
        35,
    )

    # Save mask preview
    mask_preview = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output_dir / "subject_mask.png"), mask_preview)

    # --- 4. Convert to Gaussians (35-50%) ---
    _progress("Converting depth to Gaussians...", 35)
    gaussians = depth_to_gaussians(
        image, depth, mask,
        stride=stride,
        focal_mult=focal_multiplier,
        include_background=include_background,
    )
    num_gaussians = len(gaussians["means"])
    _progress(f"Generated {num_gaussians} Gaussians", 50)

    # --- 5. Generate novel views (50-70%) ---
    _progress("Generating synthetic novel views...", 50)
    views = generate_novel_views(
        gaussians["means"],
        gaussians["colors"],
        (h, w),
        num_views=num_novel_views,
    )
    _progress(f"Generated {len(views)} novel views", 70)

    # --- 6. Write initial PLY (70-80%) ---
    _progress("Writing PLY file...", 70)
    ply_dir = output_dir / "ply"
    ply_dir.mkdir(parents=True, exist_ok=True)
    ply_path = ply_dir / "point_cloud.ply"
    write_gaussians_ply(gaussians, ply_path)
    _progress("PLY written", 80)

    # --- 7. Save novel view images + camera poses (80-100%) ---
    _progress("Saving novel views and camera poses...", 80)
    novel_dir = output_dir / "novel_views"
    novel_dir.mkdir(parents=True, exist_ok=True)

    for i, view in enumerate(views):
        # Save image
        img_bgr = cv2.cvtColor(view["image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(novel_dir / f"view_{i:03d}.png"), img_bgr)

        # Save camera pose as numpy
        np.save(str(novel_dir / f"pose_{i:03d}.npy"), view["camera_pose"])

        pct = 80 + (i + 1) / len(views) * 20
        _progress(f"Saved view {i + 1}/{len(views)}", pct)

    _progress("Portrait pipeline complete!", 100)

    return PortraitResult(
        ply_path=ply_path,
        num_gaussians=num_gaussians,
        depth_map_path=depth_preview_path,
        novel_view_count=len(views),
    )
