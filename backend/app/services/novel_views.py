"""Novel View Augmentation service.

Generates synthetic training views from existing frames + depth estimation
to supplement sparse captures before gsplat training. Uses depth-based
view warping inspired by SPAG-4D and Hunyuan3D-2mv approaches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .portrait import _get_depth_pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NovelViewResult:
    source_frames_used: int
    views_generated: int
    output_dir: Path
    synthetic_filenames: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# COLMAP camera intrinsics parser
# ---------------------------------------------------------------------------

def _load_colmap_cameras(project_dir: Path) -> Optional[dict]:
    """Try to load camera intrinsics from COLMAP output.

    Returns dict with fx, fy, cx, cy or None if not available.
    """
    # Check common COLMAP output locations
    candidates = [
        project_dir / "colmap" / "sparse" / "0" / "cameras.txt",
        project_dir / "colmap" / "sparse" / "cameras.txt",
        project_dir / "colmap" / "dense" / "sparse" / "cameras.txt",
    ]
    cameras_file = None
    for c in candidates:
        if c.exists():
            cameras_file = c
            break

    if cameras_file is None:
        return None

    try:
        with open(cameras_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                model = parts[1]
                if model == "PINHOLE" and len(parts) >= 8:
                    fx = float(parts[4])
                    fy = float(parts[5])
                    cx = float(parts[6])
                    cy = float(parts[7])
                    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
                elif model == "SIMPLE_PINHOLE" and len(parts) >= 7:
                    f_val = float(parts[4])
                    cx = float(parts[5])
                    cy = float(parts[6])
                    return {"fx": f_val, "fy": f_val, "cx": cx, "cy": cy}
                elif model in ("SIMPLE_RADIAL", "RADIAL") and len(parts) >= 7:
                    f_val = float(parts[4])
                    cx = float(parts[5])
                    cy = float(parts[6])
                    return {"fx": f_val, "fy": f_val, "cx": cx, "cy": cy}
    except Exception as e:
        logger.warning("Failed to parse COLMAP cameras: %s", e)

    return None


def _load_colmap_points3d(project_dir: Path) -> Optional[np.ndarray]:
    """Try to load 3D points from COLMAP to estimate scene center.

    Returns (N, 3) array or None.
    """
    candidates = [
        project_dir / "colmap" / "sparse" / "0" / "points3D.txt",
        project_dir / "colmap" / "sparse" / "points3D.txt",
        project_dir / "colmap" / "dense" / "sparse" / "points3D.txt",
    ]
    points_file = None
    for c in candidates:
        if c.exists():
            points_file = c
            break

    if points_file is None:
        return None

    try:
        points = []
        with open(points_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    points.append([x, y, z])
        if points:
            return np.array(points, dtype=np.float32)
    except Exception as e:
        logger.warning("Failed to parse COLMAP points3D: %s", e)

    return None


# ---------------------------------------------------------------------------
# Depth estimation for a single frame
# ---------------------------------------------------------------------------

def _estimate_frame_depth(image_path: Path) -> np.ndarray:
    """Run monocular depth estimation on a frame.

    Returns (H, W) float32 depth array with values in a relative range.
    Reuses the lazy-loaded Depth Anything V2 model from portrait.py.
    """
    from PIL import Image

    pipe = _get_depth_pipeline("small")
    image = Image.open(image_path).convert("RGB")

    result = pipe(image)
    depth = result["depth"]  # PIL Image

    depth_arr = np.array(depth, dtype=np.float32)

    # Normalise to 0..1
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max - d_min > 1e-6:
        depth_arr = (depth_arr - d_min) / (d_max - d_min)
    else:
        depth_arr = np.zeros_like(depth_arr)

    # Invert: Depth Anything returns disparity (close=high), we want depth
    depth_arr = 1.0 - depth_arr
    # Map to a reasonable depth range
    depth_arr = depth_arr * 2.7 + 0.3

    return depth_arr


# ---------------------------------------------------------------------------
# Back-project to 3D point cloud
# ---------------------------------------------------------------------------

def _backproject_to_3d(
    image: np.ndarray,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert depth + RGB to a 3D point cloud using pinhole model.

    Returns (points [N,3], colors [N,3] as uint8).
    """
    h, w = depth.shape[:2]

    vs, us = np.mgrid[0:h, 0:w]
    vs = vs.ravel()
    us = us.ravel()
    d = depth.ravel()

    valid = d > 0.01
    vs = vs[valid]
    us = us[valid]
    d = d[valid]

    # Pinhole back-projection
    x = (us.astype(np.float32) - cx) * d / fx
    y = (vs.astype(np.float32) - cy) * d / fy
    z = d

    points = np.column_stack([x, y, z]).astype(np.float32)
    colors = image[vs, us]  # (N, 3) uint8

    return points, colors


# ---------------------------------------------------------------------------
# Novel view warping with Z-buffer
# ---------------------------------------------------------------------------

def _warp_to_novel_view(
    points: np.ndarray,
    colors: np.ndarray,
    rotation_matrix: np.ndarray,
    scene_center: np.ndarray,
    h: int,
    w: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Rotate point cloud around scene center and render to a new view.

    Returns (H, W, 3) uint8 image.
    """
    # Rotate points around scene center
    pts = (points - scene_center) @ rotation_matrix.T + scene_center

    # Only project points with positive z
    valid = pts[:, 2] > 0.01
    pts_v = pts[valid]
    col_v = colors[valid]

    # Pinhole projection
    u = (pts_v[:, 0] * fx / pts_v[:, 2] + cx).astype(np.int32)
    v = (pts_v[:, 1] * fy / pts_v[:, 2] + cy).astype(np.int32)

    # Clip to image bounds
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[in_bounds]
    v = v[in_bounds]
    z = pts_v[in_bounds, 2]
    c = col_v[in_bounds]

    # Z-buffer rendering
    render = np.zeros((h, w, 3), dtype=np.uint8)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)

    # Sort by depth (far to near) so closer points overwrite
    order = np.argsort(-z)
    for idx in order:
        pu, pv, pz = int(u[idx]), int(v[idx]), z[idx]
        if pz < zbuf[pv, pu]:
            zbuf[pv, pu] = pz
            render[pv, pu] = c[idx]

    # Inpaint small holes: for each empty pixel, average nearest non-empty
    # pixels in a 5x5 window
    render = _inpaint_holes(render, zbuf, window=5)

    return render


def _inpaint_holes(
    render: np.ndarray,
    zbuf: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Fill empty pixels by averaging nearby non-empty pixels in a window."""
    h, w = render.shape[:2]
    empty = zbuf == np.inf
    if not np.any(empty):
        return render

    result = render.copy()
    half = window // 2

    # Get coordinates of empty pixels
    ey, ex = np.where(empty)

    for i in range(len(ey)):
        py, px = ey[i], ex[i]
        y0 = max(0, py - half)
        y1 = min(h, py + half + 1)
        x0 = max(0, px - half)
        x1 = min(w, px + half + 1)

        patch_zbuf = zbuf[y0:y1, x0:x1]
        patch_rgb = render[y0:y1, x0:x1]
        filled = patch_zbuf != np.inf

        if np.any(filled):
            result[py, px] = patch_rgb[filled].mean(axis=0).astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
# Camera pose generation
# ---------------------------------------------------------------------------

def _make_rotation_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Create a 3x3 rotation matrix from yaw and pitch angles."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    # Yaw rotation (around Y axis)
    cy, sy = np.cos(yaw), np.sin(yaw)
    ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ], dtype=np.float32)

    # Pitch rotation (around X axis)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rx = np.array([
        [1, 0, 0],
        [0, cp, -sp],
        [0, sp, cp],
    ], dtype=np.float32)

    return ry @ rx


def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
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

    return np.array([w, x, y, z], dtype=np.float32)


# ---------------------------------------------------------------------------
# COLMAP integration: write synthetic cameras
# ---------------------------------------------------------------------------

def _write_colmap_synthetic_cameras(
    output_dir: Path,
    synthetic_entries: list[dict],
    camera_intrinsics: dict,
    image_w: int,
    image_h: int,
) -> None:
    """Write COLMAP-format cameras.txt and images.txt for synthetic views."""
    cameras_file = output_dir / "synthetic_cameras.txt"
    images_file = output_dir / "synthetic_images.txt"

    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]

    # Write cameras.txt (single camera for all synthetic views)
    with open(cameras_file, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {image_w} {image_h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    # Write images.txt
    with open(images_file, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for i, entry in enumerate(synthetic_entries):
            img_id = 10000 + i  # avoid collision with real images
            quat = entry["quaternion"]  # (w, x, y, z)
            trans = entry["translation"]  # (tx, ty, tz)
            name = entry["filename"]
            f.write(
                f"{img_id} {quat[0]:.8f} {quat[1]:.8f} {quat[2]:.8f} {quat[3]:.8f} "
                f"{trans[0]:.8f} {trans[1]:.8f} {trans[2]:.8f} 1 {name}\n"
            )
            f.write("\n")  # empty line for POINTS2D

    logger.info(
        "Wrote %d synthetic camera entries to %s",
        len(synthetic_entries), output_dir,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_novel_views(
    project_dir: Path,
    num_views_per_frame: int = 2,
    angle_range: float = 15.0,
    max_source_frames: int = 10,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> NovelViewResult:
    """Generate synthetic novel views to augment sparse captures.

    For each selected source frame:
    1. Estimate depth using Depth Anything V2
    2. Back-project to 3D point cloud
    3. Warp to novel viewpoints with Z-buffer rendering
    4. Inpaint disocclusion holes
    5. Save synthesized images with COLMAP camera entries

    Parameters
    ----------
    project_dir : Path to the project directory
    num_views_per_frame : Number of novel views per source frame (1-4)
    angle_range : Maximum rotation angle in degrees (5-30)
    max_source_frames : Maximum number of source frames to process (5-20)
    on_progress : callback(message, percent)

    Returns
    -------
    NovelViewResult
    """
    import cv2
    from PIL import Image

    def _progress(msg: str, pct: float):
        logger.info("[novel_views %.0f%%] %s", pct, msg)
        if on_progress:
            on_progress(msg, pct)

    _progress("Starting novel view augmentation...", 0)

    frames_dir = project_dir / "frames"
    synth_dir = frames_dir / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)

    # Collect source frames
    frame_files = sorted(
        [f for f in frames_dir.iterdir()
         if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.is_file()],
        key=lambda p: p.name,
    )

    if not frame_files:
        raise FileNotFoundError("No frames found in project frames directory")

    # Select evenly spaced frames up to max_source_frames
    n_frames = len(frame_files)
    if n_frames <= max_source_frames:
        selected = frame_files
    else:
        indices = np.linspace(0, n_frames - 1, max_source_frames, dtype=int)
        selected = [frame_files[i] for i in indices]

    _progress(f"Selected {len(selected)}/{n_frames} source frames", 5)

    # Load camera intrinsics from COLMAP if available
    cam_intrinsics = _load_colmap_cameras(project_dir)

    # Load 3D points for scene center estimation
    colmap_points = _load_colmap_points3d(project_dir)

    # Generate rotation angles for novel views
    # Spread views evenly across the angle range with both yaw and pitch
    view_angles: list[tuple[float, float]] = []
    for i in range(num_views_per_frame):
        frac = (i + 1) / (num_views_per_frame + 1)
        yaw = angle_range * (2.0 * frac - 1.0)  # range: -angle_range to +angle_range
        # Alternate pitch direction, smaller magnitude
        pitch = (angle_range * 0.5) * ((-1) ** i) * frac
        view_angles.append((yaw, pitch))

    synthetic_entries: list[dict] = []
    synth_filenames: list[str] = []
    total_views = len(selected) * num_views_per_frame
    views_done = 0

    for frame_idx, frame_path in enumerate(selected):
        frame_pct_start = 10 + (frame_idx / len(selected)) * 80
        _progress(
            f"Processing frame {frame_idx + 1}/{len(selected)}: {frame_path.name}",
            frame_pct_start,
        )

        # Load image
        pil_img = Image.open(frame_path).convert("RGB")
        image = np.array(pil_img)
        h, w = image.shape[:2]

        # Estimate depth
        _progress(
            f"Estimating depth for {frame_path.name}...",
            frame_pct_start + 2,
        )
        depth = _estimate_frame_depth(frame_path)

        # Resize depth to match image if needed
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Determine camera intrinsics
        if cam_intrinsics:
            fx = cam_intrinsics["fx"]
            fy = cam_intrinsics["fy"]
            cx = cam_intrinsics["cx"]
            cy = cam_intrinsics["cy"]
        else:
            # Estimate focal length from image width
            fx = fy = w * 0.8
            cx = w / 2.0
            cy = h / 2.0

        # Back-project to 3D
        points, colors_3d = _backproject_to_3d(image, depth, fx, fy, cx, cy)

        # Estimate scene center
        if colmap_points is not None and len(colmap_points) > 0:
            scene_center = colmap_points.mean(axis=0)
        else:
            scene_center = points.mean(axis=0)

        # Generate novel views for this frame
        for view_idx, (yaw, pitch) in enumerate(view_angles):
            view_pct = frame_pct_start + (
                (view_idx + 1) / num_views_per_frame
            ) * (80 / len(selected))
            _progress(
                f"Warping view {view_idx + 1}/{num_views_per_frame} "
                f"(yaw={yaw:.1f}, pitch={pitch:.1f})",
                view_pct,
            )

            # Build rotation matrix
            rot = _make_rotation_yaw_pitch(yaw, pitch)

            # Warp to novel viewpoint
            novel_img = _warp_to_novel_view(
                points, colors_3d, rot, scene_center,
                h, w, fx, fy, cx, cy,
            )

            # Save synthetic image
            synth_name = f"synth_{frame_idx:04d}_{view_idx:02d}.jpg"
            synth_path = synth_dir / synth_name
            cv2.imwrite(
                str(synth_path),
                cv2.cvtColor(novel_img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            synth_filenames.append(synth_name)

            # Compute camera pose for COLMAP
            quat = _rotation_to_quaternion(rot)
            # Translation: rotation moves the camera relative to scene center
            t = -rot @ scene_center + scene_center
            synthetic_entries.append({
                "filename": synth_name,
                "quaternion": quat.tolist(),
                "translation": t.tolist(),
                "source_frame": frame_path.name,
                "yaw_deg": float(yaw),
                "pitch_deg": float(pitch),
            })

            views_done += 1

    # Write COLMAP-format camera entries for synthetic views
    _progress("Writing synthetic camera entries...", 92)
    intrinsics = cam_intrinsics or {"fx": w * 0.8, "fy": w * 0.8, "cx": w / 2.0, "cy": h / 2.0}
    _write_colmap_synthetic_cameras(
        synth_dir, synthetic_entries, intrinsics,
        image_w=w, image_h=h,
    )

    _progress(
        f"Novel view augmentation complete: {views_done} views generated",
        100,
    )

    return NovelViewResult(
        source_frames_used=len(selected),
        views_generated=views_done,
        output_dir=synth_dir,
        synthetic_filenames=synth_filenames,
    )
