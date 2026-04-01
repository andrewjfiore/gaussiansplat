"""
colmap_io.py — Shared COLMAP binary/text readers for training scripts.

Provides unified I/O for COLMAP sparse reconstruction files (cameras, images,
points3D) in both binary and text formats. Auto-detects format via
load_colmap_model().

All dict keys use the short convention:
  cameras:  {cid: {"model", "W", "H", "params"}}
  images:   {iid: {"qvec", "tvec", "cid", "name"}}
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Number of intrinsic params per COLMAP camera model ID
_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 8, 9: 3, 10: 6}

_MODEL_NAME_TO_ID = {
    "SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2,
    "RADIAL": 3, "OPENCV": 4, "FULL_OPENCV": 6,
}


# ──────────────────── Binary readers ────────────────────


def read_cameras_bin(path: Path) -> dict:
    cameras = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (cid,) = struct.unpack("<I", f.read(4))
            (model,) = struct.unpack("<i", f.read(4))
            (width,) = struct.unpack("<Q", f.read(8))
            (height,) = struct.unpack("<Q", f.read(8))
            np_ = _NPARAMS.get(model, 4)
            params = list(struct.unpack(f"<{np_}d", f.read(8 * np_)))
            cameras[cid] = dict(model=model, W=int(width), H=int(height), params=params)
    return cameras


def read_images_bin(path: Path) -> dict:
    images = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (iid,) = struct.unpack("<I", f.read(4))
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            (cid,) = struct.unpack("<I", f.read(4))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (n2d,) = struct.unpack("<Q", f.read(8))
            f.read(n2d * 24)  # skip xys + point3D_ids
            images[iid] = dict(qvec=qvec, tvec=tvec, cid=cid, name=name.decode("utf-8"))
    return images


def read_points3d_bin(path: Path):
    """Return (xyz [N,3] float32, rgb [N,3] float32 in 0-1)."""
    xyzs, rgbs = [], []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(8)  # point3D_id (uint64)
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            f.read(8)  # error (double)
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.read(track_len * 8)  # track entries (image_id u32 + point2D_idx u32)
            xyzs.append(xyz)
            rgbs.append([r / 255.0 for r in rgb])
    if not xyzs:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return np.array(xyzs, np.float32), np.array(rgbs, np.float32)


# ──────────────────── Text readers (fallback) ────────────────────


def read_cameras_txt(path: Path) -> dict:
    cameras = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cid = int(parts[0])
            model = _MODEL_NAME_TO_ID.get(parts[1], 1)
            W, H = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cid] = dict(model=model, W=W, H=H, params=params)
    return cameras


def read_images_txt(path: Path) -> dict:
    images = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        iid = int(parts[0])
        qvec = tuple(float(x) for x in parts[1:5])
        tvec = tuple(float(x) for x in parts[5:8])
        cid = int(parts[8])
        name = parts[9]
        images[iid] = dict(qvec=qvec, tvec=tvec, cid=cid, name=name)
        i += 2  # each image uses 2 lines (header + 2D points)
    return images


def read_points3d_txt(path: Path):
    """Return (xyz [N,3] float32, rgb [N,3] float32 in 0-1)."""
    xyzs, rgbs = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            xyzs.append([float(x) for x in parts[1:4]])
            rgbs.append([int(x) / 255.0 for x in parts[4:7]])
    if not xyzs:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return np.array(xyzs, np.float32), np.array(rgbs, np.float32)


# ──────────────────── Geometry helpers ────────────────────


def qvec_to_rotmat(q) -> np.ndarray:
    """Convert COLMAP quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def get_intrinsics(cam: dict):
    """Return (fx, fy, cx, cy) from a COLMAP camera dict."""
    p, m = cam["params"], cam["model"]
    if m == 0:  # SIMPLE_PINHOLE
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    elif m == 1:  # PINHOLE
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    else:  # radial models: fx=fy=p[0]
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])


# ──────────────────── High-level loader ────────────────────


@dataclass
class ColmapModel:
    cameras: dict
    images: dict
    points_xyz: np.ndarray  # [N, 3] float32
    points_rgb: np.ndarray  # [N, 3] float32, 0-1
    fmt: str  # "binary" or "text"


def load_colmap_model(sparse_dir: Path) -> ColmapModel:
    """Auto-detect binary vs text COLMAP model and load all components."""
    if (sparse_dir / "cameras.bin").exists():
        cameras = read_cameras_bin(sparse_dir / "cameras.bin")
        images = read_images_bin(sparse_dir / "images.bin")
        pts_xyz, pts_rgb = read_points3d_bin(sparse_dir / "points3D.bin")
        return ColmapModel(cameras, images, pts_xyz, pts_rgb, fmt="binary")
    elif (sparse_dir / "cameras.txt").exists():
        cameras = read_cameras_txt(sparse_dir / "cameras.txt")
        images = read_images_txt(sparse_dir / "images.txt")
        pts_xyz, pts_rgb = read_points3d_txt(sparse_dir / "points3D.txt")
        return ColmapModel(cameras, images, pts_xyz, pts_rgb, fmt="text")
    else:
        raise FileNotFoundError(f"No COLMAP model found in {sparse_dir}")
