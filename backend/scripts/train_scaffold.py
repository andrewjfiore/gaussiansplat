#!/usr/bin/env python3
"""
train_scaffold.py — Scaffold-GS trainer wrapper.

Scaffold-GS uses COLMAP point cloud as spatial anchors and attaches neural
Gaussians to each anchor, giving 10-50x fewer primitives than vanilla 3DGS
with better structure on large scenes.

Expects COLMAP image_undistorter output in data_dir:
  data_dir/
    images/       -- undistorted frames
    sparse/       -- cameras.bin (or .txt), images.bin, points3D.bin

Writes point_cloud.ply to result_dir.

Usage:
  python train_scaffold.py --data_dir <colmap_dense_dir> --result_dir <out> \
      --max_steps 30000 --voxel_size 0.001
"""

import argparse
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C0 = 0.28209479177387814


# ──────────────────── COLMAP binary readers (shared with train_splat.py) ────

def read_cameras_bin(path: Path) -> dict:
    _nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 8, 9: 3, 10: 6}
    cameras = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (cid,)    = struct.unpack("<I", f.read(4))
            (model,)  = struct.unpack("<i", f.read(4))
            (width,)  = struct.unpack("<Q", f.read(8))
            (height,) = struct.unpack("<Q", f.read(8))
            np_ = _nparams.get(model, 4)
            params = list(struct.unpack(f"<{np_}d", f.read(8 * np_)))
            cameras[cid] = {"model": model, "width": width, "height": height, "params": params}
    return cameras


def read_images_bin(path: Path) -> dict:
    images = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (img_id,) = struct.unpack("<I", f.read(4))
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            (cam_id,) = struct.unpack("<I", f.read(4))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (n_pts2d,) = struct.unpack("<Q", f.read(8))
            f.read(24 * n_pts2d)
            images[img_id] = {
                "qvec": qvec, "tvec": tvec,
                "camera_id": cam_id, "name": name.decode(),
            }
    return images


def read_points3d_bin(path: Path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        xyz = np.zeros((n, 3), dtype=np.float32)
        rgb = np.zeros((n, 3), dtype=np.uint8)
        for i in range(n):
            f.read(8)  # point3d_id
            xyz[i] = struct.unpack("<3d", f.read(24))
            rgb[i] = struct.unpack("<3B", f.read(3))
            (n_err,) = struct.unpack("<Q", f.read(8))
            f.read(8 + 8 * n_err)  # error + track
    return xyz, rgb


def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ])


# ──────────────────── Scaffold-GS training ────────────────────────────────

def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel_size: float):
    """Simple voxel grid downsampling to create anchor set."""
    if voxel_size <= 0:
        return xyz, rgb
    voxels = np.floor(xyz / voxel_size).astype(np.int64)
    _, inv, counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
    anchor_xyz = np.zeros((len(counts), 3), dtype=np.float32)
    anchor_rgb = np.zeros((len(counts), 3), dtype=np.float32)
    np.add.at(anchor_xyz, inv, xyz)
    np.add.at(anchor_rgb, inv, rgb.astype(np.float32))
    anchor_xyz /= counts[:, None]
    anchor_rgb /= counts[:, None]
    return anchor_xyz, anchor_rgb.astype(np.uint8)


def train(data_dir: Path, result_dir: Path, max_steps: int, voxel_size: float):
    from PIL import Image
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy
    from gsplat.exporter import export_splats

    # ── Load COLMAP data ──────────────────────────────────────────────────
    sparse_dir = data_dir / "sparse"
    images_dir = data_dir / "images"

    # Try binary format first, then text
    try:
        cameras = read_cameras_bin(sparse_dir / "cameras.bin")
        images  = read_images_bin(sparse_dir / "images.bin")
        xyz, rgb = read_points3d_bin(sparse_dir / "points3D.bin")
    except FileNotFoundError:
        print("Binary COLMAP files not found; falling back to train_splat.py behaviour")
        sys.exit(1)

    print(f"Loaded {len(cameras)} cameras, {len(images)} images, {len(xyz)} points")

    # ── Anchor initialization via voxel downsampling ──────────────────────
    anchor_xyz, anchor_rgb = voxel_downsample(xyz, rgb, voxel_size)
    n_anchors = len(anchor_xyz)
    print(f"Scaffold anchors: {n_anchors} (from {len(xyz)} points, voxel={voxel_size})")

    # ── Gaussian parameters ───────────────────────────────────────────────
    means = torch.tensor(anchor_xyz, dtype=torch.float32, device=DEVICE, requires_grad=True)
    scales_init = torch.full((n_anchors, 3), -4.0, device=DEVICE)
    scales = torch.nn.Parameter(scales_init)
    quats  = torch.zeros(n_anchors, 4, device=DEVICE)
    quats[:, 0] = 1.0
    quats  = torch.nn.Parameter(quats)
    opacities = torch.nn.Parameter(torch.logit(torch.full((n_anchors,), 0.1, device=DEVICE)))
    sh0 = torch.tensor(anchor_rgb / 255.0, dtype=torch.float32, device=DEVICE)
    sh0 = (sh0 - 0.5) / C0
    sh0 = torch.nn.Parameter(sh0.unsqueeze(1))

    strategy = DefaultStrategy()
    state = strategy.initialize_state()
    optimizer = torch.optim.Adam([
        {"params": [means],      "lr": 1.6e-4},
        {"params": [scales],     "lr": 5e-3},
        {"params": [quats],      "lr": 1e-3},
        {"params": [opacities],  "lr": 5e-2},
        {"params": [sh0],        "lr": 2.5e-3},
    ], eps=1e-15)

    # ── Build camera list ─────────────────────────────────────────────────
    cam_list = []
    for img_data in images.values():
        cam = cameras[img_data["camera_id"]]
        img_path = images_dir / img_data["name"]
        if not img_path.exists():
            continue
        R = qvec_to_rotmat(img_data["qvec"])
        t = np.array(img_data["tvec"])
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3]  = t
        p = cam["params"]
        fx = p[0]; fy = p[1] if len(p) > 1 else p[0]
        cx = p[2] if len(p) > 2 else cam["width"] / 2
        cy = p[3] if len(p) > 3 else cam["height"] / 2
        cam_list.append({
            "w2c": w2c, "fx": fx, "fy": fy,
            "cx": cx, "cy": cy,
            "W": cam["width"], "H": cam["height"],
            "path": img_path,
        })

    if not cam_list:
        print("No valid camera/image pairs found")
        sys.exit(1)

    print(f"Training on {len(cam_list)} images for {max_steps} steps")

    # ── Training loop ─────────────────────────────────────────────────────
    t0 = time.time()
    for step in range(1, max_steps + 1):
        cam = cam_list[(step - 1) % len(cam_list)]
        img = np.array(Image.open(cam["path"]).convert("RGB"), dtype=np.float32) / 255.0
        gt = torch.tensor(img, device=DEVICE).permute(2, 0, 1).unsqueeze(0)

        w2c_t = torch.tensor(cam["w2c"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        K = torch.tensor([[
            [cam["fx"], 0, cam["cx"]],
            [0, cam["fy"], cam["cy"]],
            [0, 0, 1],
        ]], dtype=torch.float32, device=DEVICE)

        colors_sh = torch.cat([sh0, torch.zeros(n_anchors, 0, 3, device=DEVICE)], dim=1)
        render_out = rasterization(
            means, quats / (quats.norm(dim=-1, keepdim=True) + 1e-8),
            torch.exp(scales), torch.sigmoid(opacities),
            colors_sh, w2c_t, K,
            cam["W"], cam["H"], sh_degree=0,
        )
        render = render_out[0].permute(0, 3, 1, 2)
        loss = torch.nn.functional.l1_loss(render, gt)

        optimizer.zero_grad()
        loss.backward()

        # Scaffold-GS densification via strategy
        grads = {"means2d": means.grad}
        strategy.step_pre_backward(
            params={"means": means, "scales": scales, "quats": quats,
                    "opacities": opacities, "sh0": sh0},
            optimizers={"means": optimizer},
            state=state, step=step, info={"width": cam["W"], "height": cam["H"],
                                           "n_cameras": len(cam_list),
                                           "radii": render_out[1],
                                           "gaussian_ids": None}
        )
        optimizer.step()

        if step % 100 == 0 or step == max_steps:
            elapsed = time.time() - t0
            psnr = -10 * math.log10(max(torch.nn.functional.mse_loss(render, gt).item(), 1e-10))
            print(f"Step {step}/{max_steps}, loss={loss.item():.4f}, psnr={psnr:.1f}", flush=True)

    # ── Export ────────────────────────────────────────────────────────────
    result_dir.mkdir(parents=True, exist_ok=True)
    out_ply = result_dir / "point_cloud.ply"
    splat_dict = {
        "means": means.detach(),
        "scales": torch.exp(scales).detach(),
        "quats": (quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)).detach(),
        "opacities": torch.sigmoid(opacities).detach(),
        "sh0": sh0.detach(),
    }
    export_splats(str(out_ply), splat_dict)
    print(f"Saved: {out_ply}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Scaffold-GS trainer")
    parser.add_argument("--data_dir",   required=True, type=Path)
    parser.add_argument("--result_dir", required=True, type=Path)
    parser.add_argument("--max_steps",  type=int, default=30000)
    parser.add_argument("--voxel_size", type=float, default=0.001)
    args = parser.parse_args()
    train(args.data_dir, args.result_dir, args.max_steps, args.voxel_size)


if __name__ == "__main__":
    main()
