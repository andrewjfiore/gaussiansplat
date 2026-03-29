#!/usr/bin/env python3
"""
train_splat.py — Gaussian Splatting trainer using gsplat 1.5.3.

Expects COLMAP image_undistorter output in data_dir:
  data_dir/
    images/       -- undistorted frames
    sparse/       -- cameras.bin (or .txt), images.bin, points3D.bin

Writes point_cloud.ply to result_dir.

Usage:
  python train_splat.py --data_dir <colmap_dense_dir> --result_dir <out> --max_steps 7000
"""

import argparse
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from gsplat.exporter import export_splats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C0 = 0.28209479177387814  # SH coefficient for degree 0


# ──────────────────── COLMAP binary readers ────────────────────

def read_cameras_bin(path: Path) -> dict:
    # Number of params by COLMAP camera model ID
    _nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 8, 9: 3, 10: 6}
    cameras = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (cid,)   = struct.unpack("<I", f.read(4))
            (model,) = struct.unpack("<i", f.read(4))
            (width,) = struct.unpack("<Q", f.read(8))
            (height,)= struct.unpack("<Q", f.read(8))
            np_ = _nparams.get(model, 4)
            params = list(struct.unpack(f"<{np_}d", f.read(8 * np_)))
            cameras[cid] = dict(model=model, W=int(width), H=int(height), params=params)
    return cameras


def read_images_bin(path: Path) -> dict:
    images = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (iid,) = struct.unpack("<I", f.read(4))
            qvec   = struct.unpack("<4d", f.read(32))   # w, x, y, z
            tvec   = struct.unpack("<3d", f.read(24))
            (cid,) = struct.unpack("<I", f.read(4))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (n2d,) = struct.unpack("<Q", f.read(8))
            f.read(n2d * 24)  # skip xys (2×8B) + point3D_ids (1×8B each)
            images[iid] = dict(qvec=qvec, tvec=tvec, cid=cid, name=name.decode("utf-8"))
    return images


def read_points3d_bin(path: Path):
    xyzs, rgbs = [], []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(8)   # point3D_id (uint64)
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            f.read(8)   # error (double)
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.read(track_len * 8)  # track entries (image_id uint32 + point2D_idx uint32)
            xyzs.append(xyz)
            rgbs.append([r / 255.0 for r in rgb])
    if not xyzs:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return np.array(xyzs, np.float32), np.array(rgbs, np.float32)


# ──────────────────── COLMAP text readers (fallback) ────────────────────

def read_cameras_txt(path: Path) -> dict:
    _model_map = {
        "SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2,
        "RADIAL": 3, "OPENCV": 4, "FULL_OPENCV": 6,
    }
    cameras = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cid = int(parts[0])
            model = _model_map.get(parts[1], 1)
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
    """Convert COLMAP quaternion (w, x, y, z) to 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def get_intrinsics(cam: dict):
    """Return (fx, fy, cx, cy) from a COLMAP camera dict."""
    p, m = cam["params"], cam["model"]
    if m == 0:                           # SIMPLE_PINHOLE
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    elif m == 1:                         # PINHOLE
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    else:                                # radial models: fx=fy=p[0]
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])


# ──────────────────── Main ────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train a 3D Gaussian Splat from COLMAP data")
    ap.add_argument("--data_dir",   required=True, type=Path,
                    help="COLMAP image_undistorter output dir")
    ap.add_argument("--result_dir", required=True, type=Path,
                    help="Directory to write point_cloud.ply")
    ap.add_argument("--max_steps",  default=7000,  type=int,
                    help="Number of training iterations")
    args = ap.parse_args()

    data_dir   = args.data_dir
    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    sparse   = data_dir / "sparse"
    imgs_dir = data_dir / "images"

    print(f"[INFO] Device: {DEVICE}", flush=True)
    print(f"[INFO] Data dir: {data_dir}", flush=True)
    print(f"[INFO] Sparse dir: {sparse}", flush=True)
    print(f"[INFO] Images dir: {imgs_dir}", flush=True)

    # ── Load COLMAP model ──────────────────────────────────────────────────
    if (sparse / "cameras.bin").exists():
        cameras    = read_cameras_bin(sparse / "cameras.bin")
        images_dat = read_images_bin(sparse / "images.bin")
        pts_xyz, pts_rgb = read_points3d_bin(sparse / "points3D.bin")
        fmt = "binary"
    elif (sparse / "cameras.txt").exists():
        cameras    = read_cameras_txt(sparse / "cameras.txt")
        images_dat = read_images_txt(sparse / "images.txt")
        pts_xyz, pts_rgb = read_points3d_txt(sparse / "points3D.txt")
        fmt = "text"
    else:
        print(f"[ERROR] No COLMAP model found in {sparse}", flush=True)
        sys.exit(1)

    print(
        f"[INFO] Loaded COLMAP ({fmt}): {len(cameras)} cam(s), "
        f"{len(images_dat)} images, {len(pts_xyz)} 3D points",
        flush=True,
    )

    if len(images_dat) == 0:
        print("[ERROR] COLMAP model contains no images", flush=True)
        sys.exit(1)

    if len(pts_xyz) < 50:
        print(f"[WARN] Sparse point cloud has only {len(pts_xyz)} points — "
              "using random initialization", flush=True)
        pts_xyz = np.random.randn(10000, 3).astype(np.float32) * 2.0
        pts_rgb = np.random.rand(10000, 3).astype(np.float32)

    # ── Build per-image viewmats + Ks + load GT images ────────────────────
    sorted_imgs = sorted(images_dat.values(), key=lambda x: x["name"])
    ref_cam = cameras[sorted_imgs[0]["cid"]]
    W, H = ref_cam["W"], ref_cam["H"]
    print(f"[INFO] Building {len(sorted_imgs)} views at {W}×{H}", flush=True)

    viewmats_np, Ks_np, gt_list = [], [], []
    for img in sorted_imgs:
        cam = cameras[img["cid"]]
        R   = qvec_to_rotmat(img["qvec"])
        t   = np.array(img["tvec"])
        vm  = np.eye(4, dtype=np.float32)
        vm[:3, :3] = R.astype(np.float32)
        vm[:3,  3] = t.astype(np.float32)
        viewmats_np.append(vm)

        fx, fy, cx, cy = get_intrinsics(cam)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        Ks_np.append(K)

        img_path = imgs_dir / img["name"]
        if img_path.exists():
            gt_np = np.array(
                Image.open(img_path).convert("RGB").resize((W, H), Image.LANCZOS),
                dtype=np.float32,
            ) / 255.0
        else:
            print(f"[WARN] Image not found: {img_path}", flush=True)
            gt_np = np.zeros((H, W, 3), dtype=np.float32)
        gt_list.append(torch.from_numpy(gt_np).to(DEVICE))

    viewmats = torch.from_numpy(np.stack(viewmats_np)).to(DEVICE)   # [N, 4, 4]
    Ks       = torch.from_numpy(np.stack(Ks_np)).to(DEVICE)        # [N, 3, 3]
    N        = len(sorted_imgs)

    # ── Initialize Gaussians from COLMAP point cloud ───────────────────────
    P = len(pts_xyz)
    print(f"[INFO] Initializing {P:,} Gaussians on {DEVICE}", flush=True)

    means     = torch.nn.Parameter(torch.from_numpy(pts_xyz).float().to(DEVICE))
    quats     = torch.nn.Parameter(torch.zeros(P, 4, device=DEVICE))
    quats.data[:, 0] = 1.0   # w=1, identity quaternion

    # Estimate scene scale from sparse point cloud
    sample_pts = means.data[:min(P, 1000)]
    if sample_pts.shape[0] > 1:
        dists = torch.cdist(sample_pts, sample_pts)
        nonzero_dists = dists[dists > 0]
        scene_scale = float(nonzero_dists.median().item()) if nonzero_dists.numel() > 0 else 1.0
    else:
        scene_scale = 1.0

    init_log_scale = math.log(max(scene_scale * 0.02, 1e-6))
    scales    = torch.nn.Parameter(
        torch.full((P, 3), init_log_scale, device=DEVICE)
    )
    opacities = torch.nn.Parameter(torch.full((P,), -3.0, device=DEVICE))  # σ(-3)≈0.05
    colors    = torch.nn.Parameter(
        torch.from_numpy(pts_rgb).float().clamp(0.0, 1.0).to(DEVICE)
    )

    params = {
        "means":     means,
        "scales":    scales,
        "quats":     quats,
        "opacities": opacities,
        "colors":    colors,
    }
    optimizers = {
        "means":     torch.optim.Adam([means],     lr=1.6e-4, eps=1e-15),
        "scales":    torch.optim.Adam([scales],    lr=5e-3,   eps=1e-15),
        "quats":     torch.optim.Adam([quats],     lr=1e-3,   eps=1e-15),
        "opacities": torch.optim.Adam([opacities], lr=5e-2,   eps=1e-15),
        "colors":    torch.optim.Adam([colors],    lr=2.5e-3, eps=1e-15),
    }

    strategy = DefaultStrategy(
        verbose=False,
        refine_start_iter=500,
        refine_stop_iter=min(args.max_steps - 200, 15_000),
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    print(
        f"[INFO] Training {args.max_steps} steps | "
        f"scene_scale={scene_scale:.4f} | {N} views",
        flush=True,
    )

    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        idx = (step - 1) % N
        vm  = viewmats[idx : idx + 1]   # [1, 4, 4]
        K   = Ks[idx : idx + 1]        # [1, 3, 3]
        gt  = gt_list[idx]              # [H, W, 3]

        q = F.normalize(params["quats"], dim=-1)

        renders, _alphas, info = rasterization(
            means=params["means"],
            quats=q,
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=params["colors"].clamp(0.0, 1.0),
            viewmats=vm,
            Ks=K,
            width=W,
            height=H,
            packed=True,
            absgrad=False,
        )

        # Must retain grad on means2d before backward
        strategy.step_pre_backward(params, optimizers, state, step, info)

        pred = renders[0].clamp(0.0, 1.0)   # [H, W, 3]
        loss = F.l1_loss(pred, gt)
        loss.backward()

        strategy.step_post_backward(params, optimizers, state, step, info, packed=True)

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        # Keep quaternions normalized
        with torch.no_grad():
            params["quats"].data.copy_(F.normalize(params["quats"].data, dim=-1))

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - t0
            n_pts   = params["means"].shape[0]
            mse     = F.mse_loss(pred.detach(), gt).item()
            psnr    = -10.0 * math.log10(max(mse, 1e-10))
            print(
                f"Step {step}/{args.max_steps}, "
                f"loss={loss.item():.4f}, "
                f"psnr={psnr:.2f}, "
                f"pts={n_pts:,}, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    # ── Export PLY ─────────────────────────────────────────────────────────
    ply_path = result_dir / "point_cloud.ply"
    print(f"\n[INFO] Exporting PLY to {ply_path}", flush=True)

    with torch.no_grad():
        m_f  = params["means"].detach()
        s_f  = torch.exp(params["scales"]).detach()
        q_f  = F.normalize(params["quats"], dim=-1).detach()
        o_f  = torch.sigmoid(params["opacities"]).detach()
        c_f  = params["colors"].clamp(0.0, 1.0).detach()

        # Convert direct RGB → SH degree-0 coefficients
        sh0 = ((c_f - 0.5) / C0).unsqueeze(1)                         # [P, 1, 3]
        shN = torch.zeros(m_f.shape[0], 0, 3, device=DEVICE)           # [P, 0, 3]

    ply_bytes = export_splats(m_f, s_f, q_f, o_f, sh0, shN)
    ply_path.write_bytes(ply_bytes)

    total = time.time() - t0
    size_kb = ply_path.stat().st_size // 1024
    print(
        f"[INFO] Done! PLY saved to {ply_path} "
        f"({size_kb:,} KB, {params['means'].shape[0]:,} Gaussians) "
        f"in {total:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
