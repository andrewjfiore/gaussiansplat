"""Analyze COLMAP sparse reconstruction to derive optimal training parameters."""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SceneStats:
    """Raw statistics computed from the COLMAP reconstruction."""
    num_points: int = 0
    num_cameras: int = 0
    num_images: int = 0
    bbox_min: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bbox_max: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    centroid: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scene_radius: float = 0.0
    mean_point_density: float = 0.0
    camera_baseline: float = 0.0
    mean_track_length: float = 0.0


@dataclass
class SceneConfig:
    """Recommended training parameters with reasoning."""
    max_steps: int = 7000
    phase1_steps: int = 5600
    phase2_steps: int = 1400
    densify_grad_thresh: float = 0.0002
    sh_degree: int = 3
    scene_complexity: str = "medium"  # low, medium, high
    reasoning: list[str] = field(default_factory=list)
    stats: Optional[SceneStats] = None


# ---------------------------------------------------------------------------
# COLMAP binary readers
# ---------------------------------------------------------------------------

def _read_cameras_binary(path: Path) -> dict:
    """Read cameras.bin. Returns {camera_id: {model, width, height, params}}."""
    cameras = {}
    if not path.exists():
        return cameras
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            # Number of params depends on model
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5}.get(model_id, 4)
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def _read_images_binary(path: Path) -> list[dict]:
    """Read images.bin. Returns list of {qw, qx, qy, qz, tx, ty, tz, name}."""
    images = []
    if not path.exists():
        return images
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            # Read null-terminated name
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8", errors="replace")
            # Read 2D points (point2D_idx, x, y for each)
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            # Skip the 2D point data: each is (x:double, y:double, point3d_id:int64)
            f.read(num_points2d * 24)
            images.append({
                "image_id": image_id,
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": tx, "ty": ty, "tz": tz,
                "camera_id": camera_id,
                "name": name,
            })
    return images


def _read_points3d_binary(path: Path) -> np.ndarray:
    """Read points3D.bin. Returns Nx3 array of 3D positions."""
    if not path.exists():
        return np.empty((0, 3))

    points = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            # Skip track data (image_id:uint32, point2d_idx:uint32 per entry)
            f.read(track_length * 8)
            points.append([x, y, z])

    return np.array(points) if points else np.empty((0, 3))


def _read_cameras_text(path: Path) -> dict:
    """Read cameras.txt fallback."""
    cameras = {}
    if not path.exists():
        return cameras
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            cameras[cam_id] = {
                "model_id": -1,
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": tuple(float(p) for p in parts[4:]),
            }
    return cameras


def _read_images_text(path: Path) -> list[dict]:
    """Read images.txt fallback."""
    images = []
    if not path.exists():
        return images
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    # images.txt has pairs of lines: metadata then 2D points
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 9:
            continue
        images.append({
            "image_id": int(parts[0]),
            "qw": float(parts[1]), "qx": float(parts[2]),
            "qy": float(parts[3]), "qz": float(parts[4]),
            "tx": float(parts[5]), "ty": float(parts[6]), "tz": float(parts[7]),
            "camera_id": int(parts[8]),
            "name": parts[9] if len(parts) > 9 else "",
        })
    return images


def _read_points3d_text(path: Path) -> np.ndarray:
    """Read points3D.txt fallback."""
    if not path.exists():
        return np.empty((0, 3))
    points = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(points) if points else np.empty((0, 3))


def _camera_position(img: dict) -> np.ndarray:
    """Compute camera center from COLMAP image extrinsics (quaternion + translation)."""
    qw, qx, qy, qz = img["qw"], img["qx"], img["qy"], img["qz"]
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])
    t = np.array([img["tx"], img["ty"], img["tz"]])
    # Camera center = -R^T * t
    return -R.T @ t


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def _find_sparse_dir(project_dir: Path) -> Optional[Path]:
    """Find the COLMAP sparse model directory."""
    # Standard locations to check
    candidates = [
        project_dir / "colmap" / "sparse" / "0",
        project_dir / "colmap" / "sparse",
        project_dir / "colmap" / "dense" / "sparse",
    ]
    for c in candidates:
        if c.exists() and (
            (c / "points3D.bin").exists()
            or (c / "points3D.txt").exists()
        ):
            return c
    # Recursive search as last resort
    for p in (project_dir / "colmap").rglob("points3D.bin"):
        return p.parent
    for p in (project_dir / "colmap").rglob("points3D.txt"):
        return p.parent
    return None


def analyze_scene(project_dir: Path) -> SceneConfig:
    """Analyze a COLMAP reconstruction and return recommended training params."""
    sparse_dir = _find_sparse_dir(project_dir)
    if sparse_dir is None:
        logger.warning("No COLMAP sparse model found in %s", project_dir)
        return SceneConfig(
            reasoning=["No COLMAP reconstruction found; using defaults."],
        )

    # Try binary first, fall back to text
    if (sparse_dir / "points3D.bin").exists():
        points3d = _read_points3d_binary(sparse_dir / "points3D.bin")
        images = _read_images_binary(sparse_dir / "images.bin")
        cameras = _read_cameras_binary(sparse_dir / "cameras.bin")
    else:
        points3d = _read_points3d_text(sparse_dir / "points3D.txt")
        images = _read_images_text(sparse_dir / "images.txt")
        cameras = _read_cameras_text(sparse_dir / "cameras.txt")

    stats = SceneStats(
        num_points=len(points3d),
        num_cameras=len(cameras),
        num_images=len(images),
    )
    reasoning: list[str] = []

    # --- Point cloud statistics ---
    if len(points3d) > 0:
        bbox_min = points3d.min(axis=0).tolist()
        bbox_max = points3d.max(axis=0).tolist()
        centroid = points3d.mean(axis=0).tolist()
        dists = np.linalg.norm(points3d - np.array(centroid), axis=1)
        scene_radius = float(np.percentile(dists, 95))  # 95th percentile to ignore outliers

        stats.bbox_min = bbox_min
        stats.bbox_max = bbox_max
        stats.centroid = centroid
        stats.scene_radius = scene_radius

        # Point density: points per unit volume of bounding box
        bbox_size = np.array(bbox_max) - np.array(bbox_min)
        volume = float(np.prod(np.maximum(bbox_size, 1e-6)))
        stats.mean_point_density = len(points3d) / volume

    # --- Camera statistics ---
    if len(images) >= 2:
        cam_positions = np.array([_camera_position(img) for img in images])
        # Average pairwise distance between consecutive cameras
        if len(cam_positions) >= 2:
            diffs = np.diff(cam_positions, axis=0)
            baselines = np.linalg.norm(diffs, axis=1)
            stats.camera_baseline = float(np.median(baselines))

    # --- Derive complexity ---
    num_pts = stats.num_points
    if num_pts < 5000:
        complexity = "low"
        reasoning.append(f"Low point count ({num_pts:,}) suggests a simple scene.")
    elif num_pts < 50000:
        complexity = "medium"
        reasoning.append(f"Moderate point count ({num_pts:,}) suggests a typical scene.")
    else:
        complexity = "high"
        reasoning.append(f"High point count ({num_pts:,}) suggests a complex scene.")

    stats_info = (
        f"Scene has {stats.num_images} views, "
        f"radius={stats.scene_radius:.2f}, "
        f"baseline={stats.camera_baseline:.3f}."
    )
    reasoning.append(stats_info)

    # --- Derive training parameters ---

    # Iterations
    if complexity == "low":
        max_steps = 5000
        reasoning.append("Using 5,000 iterations for a simple scene.")
    elif complexity == "medium":
        max_steps = 7000
        reasoning.append("Using 7,000 iterations for a typical scene.")
    else:
        max_steps = 12000
        reasoning.append("Using 12,000 iterations for a complex scene.")

    # Scale with view count
    if stats.num_images > 100:
        max_steps = min(max_steps + 3000, 15000)
        reasoning.append(f"Added iterations for {stats.num_images} views (many viewpoints).")
    elif stats.num_images < 20:
        max_steps = max(max_steps - 1000, 3000)
        reasoning.append(f"Reduced iterations for {stats.num_images} views (few viewpoints).")

    # Two-phase split: 80/20
    phase1_steps = int(max_steps * 0.8)
    phase2_steps = max_steps - phase1_steps
    reasoning.append(
        f"Phase 1 (geometry+color): {phase1_steps} steps, "
        f"Phase 2 (color refinement): {phase2_steps} steps."
    )

    # Densify gradient threshold
    if complexity == "low" or stats.mean_point_density < 100:
        densify_grad_thresh = 0.00015
        reasoning.append("Lower densification threshold for sparse scenes.")
    elif complexity == "high" or stats.mean_point_density > 10000:
        densify_grad_thresh = 0.0003
        reasoning.append("Higher densification threshold for dense scenes.")
    else:
        densify_grad_thresh = 0.0002
        reasoning.append("Standard densification threshold.")

    # SH degree
    if complexity == "low" and stats.num_images < 30:
        sh_degree = 2
        reasoning.append("SH degree 2 for simple scene (fewer view-dependent effects).")
    else:
        sh_degree = 3
        reasoning.append("SH degree 3 for full view-dependent color modeling.")

    return SceneConfig(
        max_steps=max_steps,
        phase1_steps=phase1_steps,
        phase2_steps=phase2_steps,
        densify_grad_thresh=densify_grad_thresh,
        sh_degree=sh_degree,
        scene_complexity=complexity,
        reasoning=reasoning,
        stats=stats,
    )
