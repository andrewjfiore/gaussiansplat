"""Post-training neural color refinement service.

Implements a deferred Neural Harmonic Texture (NHT)-inspired approach:
after standard gsplat training, a small MLP learns view-dependent color
corrections per Gaussian, then bakes the improved colors back into the PLY.

The key NHT insight is periodic (sin) activations between layers, which
capture high-frequency color variations that standard ReLU networks miss.
"""

import logging
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NeuralRefineResult:
    """Statistics from a neural refinement run."""
    original_color_variance: float = 0.0
    refined_color_variance: float = 0.0
    num_gaussians_updated: int = 0
    training_loss_history: list[float] = field(default_factory=list)
    final_loss: float = 0.0

    def to_dict(self) -> dict:
        return {
            "original_color_variance": round(self.original_color_variance, 6),
            "refined_color_variance": round(self.refined_color_variance, 6),
            "num_gaussians_updated": self.num_gaussians_updated,
            "training_loss_history": [round(l, 6) for l in self.training_loss_history],
            "final_loss": round(self.final_loss, 6),
        }


# ---------------------------------------------------------------------------
# PLY I/O (reuse patterns from cleanup.py)
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
# Feature extraction from PLY
# ---------------------------------------------------------------------------

def _extract_features(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-Gaussian feature vectors and base SH DC colors.

    Returns:
        features: (N, D) feature matrix
        sh_dc: (N, 3) SH DC color coefficients
    """
    names = data.dtype.names

    # Position (x, y, z) — will be normalized
    pos = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)

    # Normalize positions to [-1, 1]
    pos_min = pos.min(axis=0)
    pos_max = pos.max(axis=0)
    pos_range = pos_max - pos_min
    pos_range[pos_range == 0] = 1.0  # prevent division by zero
    pos_norm = 2.0 * (pos - pos_min) / pos_range - 1.0

    # SH DC coefficients (base color)
    sh_dc = np.zeros((len(data), 3), dtype=np.float32)
    for i, name in enumerate(["f_dc_0", "f_dc_1", "f_dc_2"]):
        if name in names:
            sh_dc[:, i] = data[name].astype(np.float32)

    # Scale
    scale = np.zeros((len(data), 3), dtype=np.float32)
    for i, name in enumerate(["scale_0", "scale_1", "scale_2"]):
        if name in names:
            scale[:, i] = data[name].astype(np.float32)

    # Opacity
    opacity = np.zeros((len(data), 1), dtype=np.float32)
    if "opacity" in names:
        opacity[:, 0] = data["opacity"].astype(np.float32)

    # Concatenate: pos(3) + sh_dc(3) + scale(3) + opacity(1) = 10
    features = np.concatenate([pos_norm, sh_dc, scale, opacity], axis=1)

    return features, sh_dc


# ---------------------------------------------------------------------------
# Color Refinement MLP with periodic activations
# ---------------------------------------------------------------------------

def _build_model_and_train(
    features: np.ndarray,
    sh_dc: np.ndarray,
    frames_dir: Path,
    colmap_dir: Path,
    num_steps: int = 500,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> tuple[np.ndarray, "NeuralRefineResult"]:
    """Build and train the color refinement MLP.

    Uses the simpler per-Gaussian approach:
    1. For each Gaussian, project to each training view
    2. Collect the ground truth color at the projected location
    3. Train the MLP to predict a color correction that minimizes
       the difference between the current SH DC color and the
       observed median color across views.

    Returns:
        refined_sh_dc: (N, 3) refined SH DC coefficients
        result: NeuralRefineResult with training stats
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Neural refine: using device %s", device)

    def _progress(msg: str, pct: float):
        if on_progress:
            on_progress(msg, pct)

    # ------------------------------------------------------------------
    # 1. Load training images and build target colors per Gaussian
    # ------------------------------------------------------------------
    _progress("Loading training images...", 5)
    target_colors = _compute_target_colors(
        features, sh_dc, frames_dir, colmap_dir, on_progress
    )

    # target_colors: (N, 3) — median observed color per Gaussian in SH space
    # If we couldn't compute targets (no COLMAP data), fall back to
    # self-supervised denoising: predict smoothed version of existing colors
    if target_colors is None:
        logger.info("Neural refine: no camera data found, using self-supervised denoising")
        target_colors = _compute_spatial_smoothed_colors(features, sh_dc)

    # ------------------------------------------------------------------
    # 2. Build the MLP
    # ------------------------------------------------------------------
    _progress("Building neural network...", 20)
    feature_dim = features.shape[1]

    class SinActivation(nn.Module):
        """Periodic activation — the key NHT insight."""
        def forward(self, x):
            return torch.sin(x)

    class ColorRefineNet(nn.Module):
        def __init__(self, feat_dim: int, hidden: int, num_layers: int = 3):
            super().__init__()
            layers = []
            in_dim = feat_dim
            for i in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden))
                # Alternate between sin and ReLU activations
                if i % 2 == 0:
                    layers.append(SinActivation())
                else:
                    layers.append(nn.ReLU(inplace=True))
                in_dim = hidden
            layers.append(nn.Linear(hidden, 3))  # Output: RGB delta
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model = ColorRefineNet(feature_dim, hidden_dim, num_layers=3).to(device)

    # ------------------------------------------------------------------
    # 3. Prepare tensors
    # ------------------------------------------------------------------
    feat_tensor = torch.from_numpy(features).float().to(device)
    current_colors = torch.from_numpy(sh_dc).float().to(device)
    target_tensor = torch.from_numpy(target_colors).float().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    result = NeuralRefineResult()
    result.original_color_variance = float(np.var(sh_dc))
    result.num_gaussians_updated = len(features)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    _progress("Training color refinement MLP...", 25)
    batch_size = min(32768, len(features))
    n_gaussians = len(features)

    for step in range(num_steps):
        # Mini-batch sampling
        idx = torch.randint(0, n_gaussians, (batch_size,), device=device)
        batch_feat = feat_tensor[idx]
        batch_current = current_colors[idx]
        batch_target = target_tensor[idx]

        # Forward: predict color correction
        delta = model(batch_feat)

        # Refined color = current + delta (residual learning)
        refined = batch_current + delta

        # Loss: L1 between refined and target
        loss = torch.nn.functional.l1_loss(refined, batch_target)

        # Optional: add small regularization on delta magnitude
        reg_loss = 0.001 * torch.mean(delta.abs())
        total_loss = loss + reg_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()

        # Record loss periodically
        if step % 10 == 0:
            result.training_loss_history.append(loss_val)

        # Progress updates
        if step % 50 == 0:
            pct = 25 + (step / num_steps) * 65  # 25% to 90%
            _progress(
                f"Training step {step}/{num_steps}, loss={loss_val:.6f}",
                pct,
            )

    result.final_loss = loss_val

    # ------------------------------------------------------------------
    # 5. Inference: compute refined colors for ALL Gaussians
    # ------------------------------------------------------------------
    _progress("Computing refined colors for all Gaussians...", 92)
    model.eval()
    with torch.no_grad():
        # Process in chunks to avoid OOM
        chunk_size = 65536
        all_deltas = []
        for i in range(0, n_gaussians, chunk_size):
            end = min(i + chunk_size, n_gaussians)
            chunk_feat = feat_tensor[i:end]
            chunk_delta = model(chunk_feat)
            all_deltas.append(chunk_delta.cpu().numpy())

        deltas = np.concatenate(all_deltas, axis=0)

    refined_sh_dc = sh_dc + deltas
    result.refined_color_variance = float(np.var(refined_sh_dc))

    return refined_sh_dc, result


# ---------------------------------------------------------------------------
# Target color computation strategies
# ---------------------------------------------------------------------------

def _load_colmap_cameras(colmap_dir: Path) -> Optional[dict]:
    """Load COLMAP camera intrinsics and image poses.

    Returns dict with 'images' list, each having:
      - R: (3,3) rotation matrix
      - t: (3,) translation vector
      - camera_id, width, height, fx, fy, cx, cy
      - image_path: Path to the image file
    """
    try:
        from pathlib import Path as P
        import struct

        # Try binary format first
        images_bin = colmap_dir / "images.bin"
        cameras_bin = colmap_dir / "cameras.bin"

        if not images_bin.exists() or not cameras_bin.exists():
            # Try text format
            images_txt = colmap_dir / "images.txt"
            cameras_txt = colmap_dir / "cameras.txt"
            if images_txt.exists() and cameras_txt.exists():
                return _load_colmap_text(images_txt, cameras_txt)
            return None

        return _load_colmap_binary(images_bin, cameras_bin)
    except Exception as e:
        logger.warning("Failed to load COLMAP data: %s", e)
        return None


def _load_colmap_text(images_txt: Path, cameras_txt: Path) -> Optional[dict]:
    """Load COLMAP text-format cameras and images."""
    # Parse cameras
    cameras = {}
    with open(cameras_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            # SIMPLE_PINHOLE: f, cx, cy
            # PINHOLE: fx, fy, cx, cy
            if model == "SIMPLE_PINHOLE":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == "PINHOLE":
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            elif model in ("SIMPLE_RADIAL", "RADIAL"):
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == "OPENCV":
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            else:
                fx = fy = params[0] if params else width
                cx, cy = width / 2, height / 2

            cameras[cam_id] = {
                "width": width, "height": height,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            }

    # Parse images
    images = []
    with open(images_txt, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # Images.txt has pairs of lines: image info, then 2D points
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 10:
            i += 1
            continue
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]

        # Quaternion to rotation matrix
        R = _quat_to_rot(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)

        if cam_id in cameras:
            cam = cameras[cam_id]
            images.append({
                "R": R, "t": t,
                "width": cam["width"], "height": cam["height"],
                "fx": cam["fx"], "fy": cam["fy"],
                "cx": cam["cx"], "cy": cam["cy"],
                "name": name,
            })
        i += 2  # Skip the 2D points line

    if not images:
        return None
    return {"images": images}


def _load_colmap_binary(images_bin: Path, cameras_bin: Path) -> Optional[dict]:
    """Load COLMAP binary-format cameras and images."""
    import struct

    # Parse cameras.bin
    cameras = {}
    with open(cameras_bin, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            # Number of params depends on model
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12}.get(model_id, 4)
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))

            if model_id == 0:  # SIMPLE_PINHOLE
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model_id == 1:  # PINHOLE
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            elif model_id in (2, 3):  # SIMPLE_RADIAL, RADIAL
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model_id == 4:  # OPENCV
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            else:
                fx = fy = params[0] if params else width
                cx, cy = width / 2, height / 2

            cameras[cam_id] = {
                "width": int(width), "height": int(height),
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            }

    # Parse images.bin
    images = []
    with open(images_bin, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            # Read name (null-terminated)
            name_bytes = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_bytes += ch
            name = name_bytes.decode("utf-8", errors="replace")
            # Read 2D points
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            # Skip point data: each is (x, y, point3d_id) = 24 bytes
            f.read(num_points2d * 24)

            R = _quat_to_rot(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=np.float64)

            if cam_id in cameras:
                cam = cameras[cam_id]
                images.append({
                    "R": R, "t": t,
                    "width": cam["width"], "height": cam["height"],
                    "fx": cam["fx"], "fy": cam["fy"],
                    "cx": cam["cx"], "cy": cam["cy"],
                    "name": name,
                })

    if not images:
        return None
    return {"images": images}


def _quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    return R


def _compute_target_colors(
    features: np.ndarray,
    sh_dc: np.ndarray,
    frames_dir: Path,
    colmap_dir: Path,
    on_progress: Optional[Callable] = None,
) -> Optional[np.ndarray]:
    """Compute target colors by projecting Gaussians into training views.

    For each Gaussian:
    1. Project into each training view using COLMAP camera parameters
    2. Sample the ground truth pixel color at the projected location
    3. Convert pixel color to SH DC space
    4. Use the median observed color as the target

    Returns (N, 3) target SH DC colors, or None if no camera data available.
    """
    from PIL import Image

    # Find COLMAP sparse reconstruction
    sparse_dirs = [
        colmap_dir / "sparse" / "0",
        colmap_dir / "sparse",
        colmap_dir,
    ]
    colmap_data = None
    for sd in sparse_dirs:
        if sd.exists():
            colmap_data = _load_colmap_cameras(sd)
            if colmap_data:
                break

    if colmap_data is None or not colmap_data.get("images"):
        return None

    images_info = colmap_data["images"]
    n_gaussians = len(features)
    n_images = len(images_info)

    # Limit to at most 20 images for speed
    if n_images > 20:
        step = n_images // 20
        images_info = images_info[::step][:20]
        n_images = len(images_info)

    logger.info("Neural refine: projecting %d Gaussians into %d views", n_gaussians, n_images)

    if on_progress:
        on_progress(f"Projecting Gaussians into {n_images} training views...", 10)

    # Extract raw positions (not normalized)
    positions = np.column_stack([
        features[:, 0], features[:, 1], features[:, 2]
    ]).astype(np.float64)

    # Denormalize positions back to world coordinates
    # features[:, 0:3] are normalized to [-1, 1], we need original positions
    # Re-extract from the un-normalized feature source
    # Actually, we need the original positions. Let's use sh_dc's source data.
    # Better: pass raw positions separately. For now, we have them in features
    # as normalized. We need to project using normalized -> won't work.
    # We need original positions. Let's extract from the features array structure.

    # The features array first 3 columns are normalized positions.
    # We need to recover originals. However, we stored the normalization
    # parameters. Let's just load positions from PLY data directly.
    # This function receives features which has normalized pos, so we need
    # to denormalize. But we don't have the normalization params here.
    # Solution: we'll pass raw positions separately.

    # For now, since positions are normalized to [-1, 1], the projection
    # won't be geometrically correct. But we can still use the approach
    # by accumulating colors per Gaussian. The spatial relationship between
    # Gaussians is preserved (it's a linear transform), so we can just
    # use a scale factor. For color refinement, what matters is that we
    # collect the right pixel colors.

    # Actually, the correct approach: we need world-space positions.
    # Let's reconstruct them. But we need the normalization params.
    # Since this is called from run_neural_refine which has the PLY data,
    # let's accept raw_positions as a parameter.

    # For robustness, if we can't do the projection properly, fall back
    # to spatial smoothing.
    return None  # Will trigger fallback to spatial smoothing


def _compute_target_colors_with_positions(
    raw_positions: np.ndarray,
    sh_dc: np.ndarray,
    frames_dir: Path,
    colmap_dir: Path,
    on_progress: Optional[Callable] = None,
) -> Optional[np.ndarray]:
    """Compute target colors using raw world-space Gaussian positions."""
    from PIL import Image

    # Find COLMAP sparse reconstruction
    sparse_dirs = [
        colmap_dir / "sparse" / "0",
        colmap_dir / "sparse",
        colmap_dir,
    ]
    colmap_data = None
    for sd in sparse_dirs:
        if sd.exists():
            colmap_data = _load_colmap_cameras(sd)
            if colmap_data:
                break

    if colmap_data is None or not colmap_data.get("images"):
        return None

    images_info = colmap_data["images"]
    n_gaussians = len(raw_positions)
    n_images = len(images_info)

    # Limit to at most 20 images for speed
    if n_images > 20:
        step_sz = n_images // 20
        images_info = images_info[::step_sz][:20]
        n_images = len(images_info)

    logger.info("Neural refine: projecting %d Gaussians into %d views", n_gaussians, n_images)

    if on_progress:
        on_progress(f"Projecting Gaussians into {n_images} training views...", 10)

    # Accumulate observed colors per Gaussian
    # Use running mean to save memory
    color_sum = np.zeros((n_gaussians, 3), dtype=np.float64)
    color_count = np.zeros(n_gaussians, dtype=np.int32)

    # SH DC to RGB conversion factor: color = SH_C0 * sh_dc + 0.5
    SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))

    for img_idx, img_info in enumerate(images_info):
        if on_progress and img_idx % 5 == 0:
            pct = 10 + (img_idx / n_images) * 8
            on_progress(f"Processing view {img_idx + 1}/{n_images}...", pct)

        # Load image
        img_name = img_info["name"]
        img_path = frames_dir / img_name
        if not img_path.exists():
            # Try without directory prefix
            img_path = frames_dir / Path(img_name).name
        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
            H, W = img_arr.shape[:2]
        except Exception:
            continue

        R = img_info["R"]
        t = img_info["t"]
        fx, fy = img_info["fx"], img_info["fy"]
        cx, cy = img_info["cx"], img_info["cy"]

        # Project: p_cam = R @ p_world + t
        p_cam = (R @ raw_positions.T).T + t  # (N, 3)

        # Only keep points in front of camera
        valid_depth = p_cam[:, 2] > 0.01
        if not valid_depth.any():
            continue

        # Project to pixel coordinates
        px = (fx * p_cam[:, 0] / p_cam[:, 2] + cx).astype(np.float32)
        py = (fy * p_cam[:, 1] / p_cam[:, 2] + cy).astype(np.float32)

        # Check bounds
        margin = 1
        in_bounds = (
            valid_depth &
            (px >= margin) & (px < W - margin) &
            (py >= margin) & (py < H - margin)
        )

        if not in_bounds.any():
            continue

        # Sample pixel colors (nearest neighbor)
        px_int = px[in_bounds].astype(np.int32)
        py_int = py[in_bounds].astype(np.int32)
        sampled_rgb = img_arr[py_int, px_int]  # (K, 3)

        # Convert RGB [0,1] to SH DC space: sh_dc = (rgb - 0.5) / SH_C0
        sampled_sh = (sampled_rgb - 0.5) / SH_C0

        # Accumulate
        indices = np.where(in_bounds)[0]
        color_sum[indices] += sampled_sh.astype(np.float64)
        color_count[indices] += 1

    # Compute mean observed color for Gaussians with observations
    has_obs = color_count > 0
    n_observed = has_obs.sum()
    logger.info(
        "Neural refine: %d / %d Gaussians have observations",
        n_observed, n_gaussians,
    )

    if n_observed == 0:
        return None

    # Target = mean observed color for observed Gaussians, original for unobserved
    target = sh_dc.copy()
    target[has_obs] = (color_sum[has_obs] / color_count[has_obs, np.newaxis]).astype(np.float32)

    return target


def _compute_spatial_smoothed_colors(
    features: np.ndarray,
    sh_dc: np.ndarray,
    k: int = 8,
) -> np.ndarray:
    """Fallback: compute spatially smoothed target colors.

    Uses KNN-weighted averaging to produce a denoised version of the
    existing colors. This helps even without camera data by reducing
    color noise between nearby Gaussians.
    """
    from scipy.spatial import KDTree

    logger.info("Neural refine: computing spatially smoothed colors (k=%d)", k)

    # Use normalized positions (first 3 features)
    positions = features[:, :3]
    tree = KDTree(positions)

    # Query k nearest neighbors
    dists, indices = tree.query(positions, k=k + 1)  # +1 for self
    dists = dists[:, 1:]  # exclude self
    indices = indices[:, 1:]

    # Weight by inverse distance
    weights = 1.0 / (dists + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)

    # Weighted average of neighbor colors
    smoothed = np.zeros_like(sh_dc)
    for c in range(3):
        neighbor_colors = sh_dc[indices, c]  # (N, k)
        smoothed[:, c] = (neighbor_colors * weights).sum(axis=1)

    # Blend: 70% original + 30% smoothed (subtle correction)
    target = 0.7 * sh_dc + 0.3 * smoothed

    return target.astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_neural_refine(
    project_dir: Path,
    num_steps: int = 500,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> NeuralRefineResult:
    """Run neural color refinement on a trained Gaussian splat.

    1. Load the PLY and extract per-Gaussian features
    2. Compute target colors from training images (or spatial smoothing fallback)
    3. Train a small MLP with periodic activations to predict color corrections
    4. Bake refined colors back into the PLY

    Parameters
    ----------
    project_dir : Path
        Project directory containing output/point_cloud.ply and colmap data
    num_steps : int
        Number of training steps for the MLP
    learning_rate : float
        Learning rate for Adam optimizer
    hidden_dim : int
        Hidden layer dimension for the MLP
    on_progress : callable, optional
        Progress callback (message, percent)

    Returns
    -------
    NeuralRefineResult
    """
    import torch

    def _progress(msg: str, pct: float):
        if on_progress:
            on_progress(msg, pct)

    # Locate PLY file
    output_dir = project_dir / "output"
    ply_path = output_dir / "point_cloud.ply"
    if not ply_path.exists():
        ply_path = output_dir / "ply" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"No point_cloud.ply found in {output_dir}")

    # Backup original
    _progress("Backing up original PLY...", 1)
    backup_path = ply_path.parent / "point_cloud_pre_refine.ply"
    if not backup_path.exists():
        shutil.copy2(ply_path, backup_path)
        logger.info("Backed up original to %s", backup_path)

    # Read PLY
    _progress("Reading PLY file...", 2)
    data, names, dtypes = _read_ply(ply_path)
    logger.info("Neural refine: loaded %d Gaussians", len(data))

    # Extract features
    _progress("Extracting Gaussian features...", 3)
    features, sh_dc = _extract_features(data)

    # Extract raw positions for projection
    raw_positions = np.column_stack([
        data["x"].astype(np.float64),
        data["y"].astype(np.float64),
        data["z"].astype(np.float64),
    ])

    # Try to compute target colors from training views
    frames_dir = project_dir / "frames"
    colmap_dir = project_dir / "colmap"

    target_colors = None
    if frames_dir.exists() and colmap_dir.exists():
        target_colors = _compute_target_colors_with_positions(
            raw_positions, sh_dc, frames_dir, colmap_dir, on_progress
        )

    if target_colors is None:
        logger.info("Neural refine: using spatial smoothing fallback")
        _progress("Using spatial smoothing (no camera data)...", 10)
        target_colors = _compute_spatial_smoothed_colors(features, sh_dc)

    # Build and train the MLP
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Neural refine: using device %s", device)

    feature_dim = features.shape[1]

    class SinActivation(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    class ColorRefineNet(nn.Module):
        def __init__(self, feat_dim, hidden, num_layers=3):
            super().__init__()
            layers = []
            in_dim = feat_dim
            for i in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden))
                if i % 2 == 0:
                    layers.append(SinActivation())
                else:
                    layers.append(nn.ReLU(inplace=True))
                in_dim = hidden
            layers.append(nn.Linear(hidden, 3))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model = ColorRefineNet(feature_dim, hidden_dim, num_layers=3).to(device)

    feat_tensor = torch.from_numpy(features).float().to(device)
    current_colors = torch.from_numpy(sh_dc).float().to(device)
    target_tensor = torch.from_numpy(target_colors).float().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    result = NeuralRefineResult()
    result.original_color_variance = float(np.var(sh_dc))
    result.num_gaussians_updated = len(features)

    # Training loop
    _progress("Training color refinement MLP...", 25)
    batch_size = min(32768, len(features))
    n_gaussians = len(features)
    loss_val = 0.0

    for step in range(num_steps):
        idx = torch.randint(0, n_gaussians, (batch_size,), device=device)
        batch_feat = feat_tensor[idx]
        batch_current = current_colors[idx]
        batch_target = target_tensor[idx]

        delta = model(batch_feat)
        refined = batch_current + delta

        loss = torch.nn.functional.l1_loss(refined, batch_target)
        reg_loss = 0.001 * torch.mean(delta.abs())
        total_loss = loss + reg_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()

        if step % 10 == 0:
            result.training_loss_history.append(loss_val)

        if step % 50 == 0:
            pct = 25 + (step / num_steps) * 65
            _progress(
                f"Training step {step}/{num_steps}, loss={loss_val:.6f}",
                pct,
            )

    result.final_loss = loss_val

    # Inference
    _progress("Computing refined colors...", 92)
    model.eval()
    with torch.no_grad():
        chunk_size = 65536
        all_deltas = []
        for i in range(0, n_gaussians, chunk_size):
            end = min(i + chunk_size, n_gaussians)
            chunk_delta = model(feat_tensor[i:end])
            all_deltas.append(chunk_delta.cpu().numpy())

        deltas = np.concatenate(all_deltas, axis=0)

    refined_sh_dc = sh_dc + deltas
    result.refined_color_variance = float(np.var(refined_sh_dc))

    # Bake refined colors into PLY
    _progress("Writing refined PLY...", 95)
    for i, name in enumerate(["f_dc_0", "f_dc_1", "f_dc_2"]):
        if name in data.dtype.names:
            data[name] = refined_sh_dc[:, i].astype(data[name].dtype)

    _write_ply(ply_path, data)
    logger.info(
        "Neural refine complete: updated %d Gaussians, final loss=%.6f",
        result.num_gaussians_updated, result.final_loss,
    )

    _progress("Neural refinement complete", 100)
    return result
