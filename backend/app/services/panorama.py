"""Panorama Mode pipeline: 360 equirectangular image -> 3D Gaussian splat.

Inspired by SPAG-4D's panorama-to-Gaussian-splat pipeline.  Uses spherical
projection to convert an equirectangular depth map + RGB into oriented 3D
Gaussians that can be viewed from the centre of the sphere.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Re-use depth model from portrait pipeline (same lazy-loaded singleton)
from .portrait import estimate_depth, save_depth_preview, write_gaussians_ply

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PanoramaResult:
    ply_path: Path
    num_gaussians: int
    sky_removed: int
    depth_map_path: Path


# ---------------------------------------------------------------------------
# Spherical Projection (SPAG-4D core math)
# ---------------------------------------------------------------------------


def _equirect_to_gaussians(
    image: np.ndarray,
    depth: np.ndarray,
    stride: int = 2,
    sky_mode: str = "skip",
    depth_scale: float = 5.0,
) -> dict:
    """Convert equirectangular image + depth to 3D Gaussians via spherical
    projection.

    Parameters
    ----------
    image : (H, W, 3) uint8 — equirectangular RGB
    depth : (H, W) float32 — estimated depth (relative)
    stride : pixel stride for downsampling
    sky_mode : "skip" | "low_opacity" | "keep"
    depth_scale : scale factor to map normalised depth to world units

    Returns
    -------
    dict with keys: means, scales, rotations, colors, opacities
    int — number of sky pixels removed
    """
    h, w = depth.shape[:2]

    # Build grid of pixel coordinates at the given stride
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    vs = vs.ravel().astype(np.float64)
    us = us.ravel().astype(np.float64)

    # Compute angular coordinates
    theta = (us / w) * 2.0 * np.pi - np.pi        # longitude: -pi to pi
    phi = (0.5 - vs / h) * np.pi                   # latitude: -pi/2 to pi/2

    # Depth values at sampled pixels
    d = depth[vs.astype(np.intp), us.astype(np.intp)].astype(np.float64)
    # Scale depth to world units
    d = d * depth_scale

    # Sky detection: pixels with depth > 95th percentile
    depth_threshold = np.percentile(d, 95)
    is_sky = d >= depth_threshold

    # Compute unit direction vectors
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    dir_x = cos_phi * sin_theta
    dir_y = sin_phi
    dir_z = cos_phi * cos_theta

    # 3D positions
    px = d * dir_x
    py = d * dir_y
    pz = d * dir_z

    # Gaussian scales (anisotropic disc, SPAG-4D style)
    delta = np.pi / h  # radians per pixel
    base_scale = d * delta * stride

    # Latitude correction to prevent pole convergence
    abs_sin_phi = np.abs(sin_phi)
    scale_xy = base_scale * np.clip(abs_sin_phi, 0.01, 1.0)
    scale_z = base_scale * 0.1  # thin disc along viewing direction

    # Pole thinning: stochastic thinning near poles
    # Keep probability proportional to sin(|phi|), floor at 0.1
    keep_prob = np.clip(np.abs(np.sin(phi)), 0.1, 1.0)
    rng = np.random.default_rng(42)
    pole_keep = rng.random(len(keep_prob)) < keep_prob

    # Apply sky handling
    sky_removed = 0
    if sky_mode == "skip":
        valid = ~is_sky & pole_keep
        sky_removed = int(is_sky.sum())
    elif sky_mode == "low_opacity":
        valid = pole_keep
        sky_removed = 0
    else:  # "keep"
        valid = pole_keep
        sky_removed = 0

    # Filter
    px = px[valid].astype(np.float32)
    py = py[valid].astype(np.float32)
    pz = pz[valid].astype(np.float32)
    scale_xy_f = scale_xy[valid].astype(np.float32)
    scale_z_f = scale_z[valid].astype(np.float32)
    is_sky_filtered = is_sky[valid]

    means = np.column_stack([px, py, pz])

    # Scales: (scale_xy, scale_xy, scale_z) — disc-like
    scales = np.column_stack([
        scale_xy_f,
        scale_xy_f,
        scale_z_f,
    ])
    # Clamp scales
    scales = np.clip(scales, 1e-5, 1.0)

    # Colors from panorama
    vi = vs[valid].astype(np.intp)
    ui = us[valid].astype(np.intp)
    colors = image[vi, ui].astype(np.float32) / 255.0

    # Opacity
    opacities = np.ones(len(means), dtype=np.float32)
    if sky_mode == "low_opacity":
        opacities[is_sky_filtered] = 0.1

    # Rotation: orient disc perpendicular to viewing direction
    # Normal = -direction (pointing toward camera at origin)
    dir_x_f = dir_x[valid].astype(np.float32)
    dir_y_f = dir_y[valid].astype(np.float32)
    dir_z_f = dir_z[valid].astype(np.float32)

    normals = -np.column_stack([dir_x_f, dir_y_f, dir_z_f])
    # Normalise (should already be unit but safety)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normals = normals / norms

    # Build rotation quaternions from normal direction
    rotations = _normals_to_quaternions(normals)

    n = len(means)
    logger.info(
        "Generated %d Gaussians from panorama (stride=%d, sky_mode=%s, sky_removed=%d)",
        n, stride, sky_mode, sky_removed,
    )

    return {
        "means": means,
        "scales": scales,
        "rotations": rotations,
        "colors": colors,
        "opacities": opacities,
    }, sky_removed


def _normals_to_quaternions(normals: np.ndarray) -> np.ndarray:
    """Convert an array of normal vectors to quaternions (WXYZ) that orient
    a disc perpendicular to the viewing direction.

    Parameters
    ----------
    normals : (N, 3) float32 — unit normal vectors

    Returns
    -------
    quaternions : (N, 4) float32 — (w, x, y, z)
    """
    n = len(normals)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # For each normal, compute Right = cross(up, normal), Up = cross(normal, right)
    right = np.cross(up, normals)
    right_norms = np.linalg.norm(right, axis=1, keepdims=True)

    # Handle degenerate case where normal is parallel to up
    degenerate = right_norms.ravel() < 1e-6
    right[degenerate] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right_norms[degenerate] = 1.0

    right = right / np.clip(right_norms, 1e-8, None)
    up_vec = np.cross(normals, right)

    # Build rotation matrices and convert to quaternions
    # Rotation matrix columns: [right, up_vec, normal]
    quats = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        R = np.column_stack([right[i], up_vec[i], normals[i]])
        quats[i] = _rotation_matrix_to_quaternion(R)

    return quats


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float32)
    # Normalise
    q /= max(np.linalg.norm(q), 1e-8)
    return q


# ---------------------------------------------------------------------------
# Main Pipeline Orchestrator
# ---------------------------------------------------------------------------


def run_panorama_pipeline(
    image_path: Path,
    output_dir: Path,
    *,
    stride: int = 2,
    sky_mode: str = "skip",
    depth_mode: str = "single",
    depth_model: str = "small",
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> PanoramaResult:
    """Run the full panorama-to-Gaussian pipeline.

    Steps:
    1. Load equirectangular image (0-5%)
    2. Estimate depth (5-40%)
    3. Spherical projection to Gaussians (40-70%)
    4. Apply pole thinning + sky detection (70-80%)
    5. Write PLY (80-90%)
    6. Save depth preview (90-100%)

    Parameters
    ----------
    image_path : Path to equirectangular panoramic image
    output_dir : Directory for all outputs
    stride : pixel stride for downsampling (1-4)
    sky_mode : "skip" | "low_opacity" | "keep"
    depth_mode : "single" (direct equirectangular) or "cubemap" (future)
    depth_model : "small" or "base"
    on_progress : callback(message, percent)

    Returns
    -------
    PanoramaResult
    """
    import cv2
    from PIL import Image

    def _progress(msg: str, pct: float):
        logger.info("[panorama %.0f%%] %s", pct, msg)
        if on_progress:
            on_progress(msg, pct)

    # --- 1. Load equirectangular image (0-5%) ---
    _progress("Loading panorama...", 0)
    pil_image = Image.open(image_path).convert("RGB")
    image = np.array(pil_image)
    h, w = image.shape[:2]
    logger.info("Panorama dimensions: %dx%d (aspect %.2f:1)", w, h, w / max(h, 1))

    # Limit width for depth estimation (panoramas can be very large)
    max_width = 4096
    if w > max_width:
        scale = max_width / w
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
        logger.info("Resized panorama to %dx%d for processing", w, h)

    output_dir.mkdir(parents=True, exist_ok=True)
    _progress("Panorama loaded", 5)

    # --- 2. Estimate depth (5-40%) ---
    _progress("Estimating depth...", 5)

    # Save resized image for depth estimation
    resized_path = output_dir / "panorama_input.jpg"
    cv2.imwrite(str(resized_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    depth = estimate_depth(resized_path, model_size=depth_model)

    # Resize depth to match image if needed
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    _progress("Depth estimation complete", 40)

    # Save depth preview
    depth_preview_path = output_dir / "depth_preview.png"
    save_depth_preview(depth, depth_preview_path)

    # --- 3-4. Spherical projection + pole thinning + sky detection (40-80%) ---
    _progress("Projecting to 3D...", 40)

    gaussians, sky_removed = _equirect_to_gaussians(
        image, depth,
        stride=stride,
        sky_mode=sky_mode,
    )

    num_gaussians = len(gaussians["means"])
    _progress(f"Generated {num_gaussians} Gaussians (sky removed: {sky_removed})", 80)

    # --- 5. Write PLY (80-90%) ---
    _progress("Writing PLY file...", 80)
    ply_dir = output_dir / "ply"
    ply_dir.mkdir(parents=True, exist_ok=True)
    ply_path = ply_dir / "point_cloud.ply"
    write_gaussians_ply(gaussians, ply_path)
    _progress("PLY written", 90)

    # --- 6. Save depth preview (already done above) (90-100%) ---
    _progress("Panorama pipeline complete!", 100)

    return PanoramaResult(
        ply_path=ply_path,
        num_gaussians=num_gaussians,
        sky_removed=sky_removed,
        depth_map_path=depth_preview_path,
    )
