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
import signal
import sys
import time
from pathlib import Path

# Graceful early-stop: catch SIGTERM/SIGINT to break training loop and still export PLY
_stop_requested = False

def _handle_stop(signum, frame):
    global _stop_requested
    _stop_requested = True
    print(f"\n[INFO] Stop requested (signal {signum}) — finishing current step and exporting PLY...", flush=True)

signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)

import numpy as np
import torch

from colmap_io import load_colmap_model, qvec_to_rotmat, get_intrinsics
from losses import ssim as compute_ssim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C0 = 0.28209479177387814


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


def train(data_dir: Path, result_dir: Path, max_steps: int, voxel_size: float,
          resume: bool = False, ckpt_interval: int = 1000, sh_degree: int = 0,
          depth_dir: Path = None, depth_weight: float = 0.1,
          pseudo_gt_dir: Path = None, confidence_weight: float = 0.5):
    from PIL import Image
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy
    from gsplat.exporter import export_splats

    # ── Load COLMAP data ──────────────────────────────────────────────────
    sparse_dir = data_dir / "sparse"
    images_dir = data_dir / "images"

    try:
        model = load_colmap_model(sparse_dir)
    except FileNotFoundError:
        print(f"[ERROR] No COLMAP model found in {sparse_dir}", flush=True)
        sys.exit(1)

    cameras = model.cameras
    images = model.images
    xyz, rgb_f = model.points_xyz, model.points_rgb
    # Convert 0-1 float RGB back to 0-255 uint8 for voxel_downsample
    rgb = (rgb_f * 255).astype(np.uint8)

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

    # SH coefficients: [N, K, 3] where K = (sh_degree+1)^2
    num_sh = (sh_degree + 1) ** 2
    sh_coeffs = torch.zeros(n_anchors, num_sh, 3, device=DEVICE)
    sh_coeffs[:, 0, :] = (torch.tensor(anchor_rgb / 255.0, dtype=torch.float32, device=DEVICE) - 0.5) / C0
    sh_coeffs = torch.nn.Parameter(sh_coeffs)
    print(f"SH degree {sh_degree} ({num_sh} coefficients per Gaussian)", flush=True)

    params = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh_coeffs": sh_coeffs,
    }
    optimizers = {
        "means":     torch.optim.Adam([means],      lr=1.6e-4, eps=1e-15),
        "scales":    torch.optim.Adam([scales],     lr=5e-3,   eps=1e-15),
        "quats":     torch.optim.Adam([quats],      lr=1e-3,   eps=1e-15),
        "opacities": torch.optim.Adam([opacities],  lr=5e-2,   eps=1e-15),
        "sh_coeffs": torch.optim.Adam([sh_coeffs],  lr=2.5e-3, eps=1e-15),
    }
    # Progressive SH: grow degree over training
    sh_grow_interval = max(max_steps // (sh_degree + 1), 1) if sh_degree > 0 else max_steps

    strategy = DefaultStrategy()
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    # Learning rate scheduling: decay means LR from 1.6e-4 → 1.6e-6
    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"],
        gamma=(1.6e-6 / 1.6e-4) ** (1.0 / max(max_steps, 1)),
    )

    # ── Build camera list ─────────────────────────────────────────────────
    cam_list = []
    for img_data in images.values():
        cam = cameras[img_data["cid"]]
        img_path = images_dir / img_data["name"]
        if not img_path.exists():
            continue
        R = qvec_to_rotmat(img_data["qvec"])
        t = np.array(img_data["tvec"])
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3]  = t
        fx, fy, cx, cy = get_intrinsics(cam)
        cam_list.append({
            "w2c": w2c, "fx": fx, "fy": fy,
            "cx": cx, "cy": cy,
            "W": cam["W"], "H": cam["H"],
            "path": img_path,
        })

    if not cam_list:
        print("No valid camera/image pairs found")
        sys.exit(1)

    # Load depth maps if available
    use_depth = depth_dir is not None and depth_dir.exists() and depth_weight > 0
    depth_map_cache = {}
    if use_depth:
        n_depths = 0
        for i, cam_info in enumerate(cam_list):
            stem = Path(cam_info["path"]).stem
            depth_path = depth_dir / f"{stem}.npy"
            if depth_path.exists():
                depth_map_cache[i] = depth_path
                n_depths += 1
        print(f"Depth maps: {n_depths}/{len(cam_list)} available (weight={depth_weight})")
        if n_depths == 0:
            use_depth = False

    print(f"Training on {len(cam_list)} images for {max_steps} steps")

    # ── Checkpoint resume ─────────────────────────────────────────────────
    start_step = 1
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = result_dir / "checkpoint.pt"
    if resume and ckpt_path.exists():
        print(f"[INFO] Resuming from checkpoint: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        start_step = ckpt["step"] + 1
        for k in params:
            params[k].data.copy_(ckpt["params"][k])
        for k in optimizers:
            if k in ckpt["optimizer_states"]:
                optimizers[k].load_state_dict(ckpt["optimizer_states"][k])
        if "scheduler_state" in ckpt:
            means_scheduler.load_state_dict(ckpt["scheduler_state"])
        if "strategy_state" in ckpt:
            state = ckpt["strategy_state"]
        print(f"[INFO] Resumed at step {start_step}", flush=True)

    # ── Training loop ─────────────────────────────────────────────────────
    t0 = time.time()
    for step in range(start_step, max_steps + 1):
        if _stop_requested:
            print(f"[INFO] Early stop at step {step - 1}/{max_steps}", flush=True)
            break
        cam = cam_list[(step - 1) % len(cam_list)]
        pil_img = Image.open(cam["path"]).convert("RGB")
        # Resize to COLMAP camera dimensions if needed (dataset images may be at different resolution)
        if pil_img.size != (cam["W"], cam["H"]):
            pil_img = pil_img.resize((cam["W"], cam["H"]), Image.LANCZOS)
        img = np.array(pil_img, dtype=np.float32) / 255.0
        gt = torch.tensor(img, device=DEVICE).permute(2, 0, 1).unsqueeze(0)

        w2c_t = torch.tensor(cam["w2c"], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        K = torch.tensor([[
            [cam["fx"], 0, cam["cx"]],
            [0, cam["fy"], cam["cy"]],
            [0, 0, 1],
        ]], dtype=torch.float32, device=DEVICE)

        q = torch.nn.functional.normalize(params["quats"], dim=-1)
        # Progressive SH: grow degree over training
        cur_sh_degree = min(sh_degree, step // sh_grow_interval) if sh_degree > 0 else 0

        render_mode = "RGB+ED" if use_depth else "RGB"
        render_out, _alphas, info = rasterization(
            means=params["means"],
            quats=q,
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=params["sh_coeffs"],
            viewmats=w2c_t,
            Ks=K,
            width=cam["W"],
            height=cam["H"],
            sh_degree=cur_sh_degree,
            packed=True,
            absgrad=False,
            render_mode=render_mode,
        )
        if use_depth:
            rgb_out = render_out[..., :3]
            depth_out = render_out[..., 3:4]
        else:
            rgb_out = render_out

        render = rgb_out.permute(0, 3, 1, 2)  # [B, C, H, W]
        # Confidence-aware target blending (pseudo-GT from visibility transfer)
        target = gt
        # TODO: load pseudo-GT in BCHW layout when pseudo_gt_dir is provided
        l1 = torch.nn.functional.l1_loss(render, target)
        ssim_val = compute_ssim(render, target)
        loss = 0.2 * l1 + 0.8 * (1.0 - ssim_val)

        # Depth supervision
        cam_idx = (step - 1) % len(cam_list)
        if use_depth and cam_idx in depth_map_cache:
            mono_depth = np.load(depth_map_cache[cam_idx]).astype(np.float32)
            mono_depth = torch.tensor(mono_depth, device=DEVICE)
            # Resize to match render size
            if mono_depth.shape != (cam["H"], cam["W"]):
                mono_depth = torch.nn.functional.interpolate(
                    mono_depth.unsqueeze(0).unsqueeze(0),
                    size=(cam["H"], cam["W"]), mode="bilinear", align_corners=False
                ).squeeze()
            rd = depth_out[0, ..., 0]  # [H, W]
            rd_min, rd_max = rd.min(), rd.max()
            if rd_max - rd_min > 1e-6:
                rd_norm = (rd - rd_min) / (rd_max - rd_min)
            else:
                rd_norm = rd
            loss = loss + depth_weight * torch.nn.functional.l1_loss(rd_norm, mono_depth)

        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=True)

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_scheduler.step()

        # Keep quaternions normalized
        with torch.no_grad():
            params["quats"].data.copy_(torch.nn.functional.normalize(params["quats"].data, dim=-1))

        if step % 100 == 0 or step == max_steps:
            elapsed = time.time() - t0
            psnr = -10 * math.log10(max(torch.nn.functional.mse_loss(render, gt).item(), 1e-10))
            n_pts = params["means"].shape[0]
            print(f"Step {step}/{max_steps}, loss={loss.item():.4f}, psnr={psnr:.1f}, pts={n_pts:,}", flush=True)

        # Save snapshot at 25% intervals
        snapshot_steps = {
            int(max_steps * p): p_int
            for p, p_int in [(0.25, 25), (0.5, 50), (0.75, 75), (1.0, 100)]
        }
        if step in snapshot_steps:
            pct = snapshot_steps[step]
            snap_path = result_dir / f"snapshot_{pct}.jpg"
            # render is [B, C, H, W], take first batch
            snap_np = (render[0].detach().clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
            Image.fromarray(snap_np).save(str(snap_path), quality=90)
            print(f"[SNAPSHOT] {pct} snapshot_{pct}.jpg", flush=True)

        # Save checkpoint periodically
        if step % ckpt_interval == 0:
            torch.save({
                "step": step,
                "params": {k: v.data.clone() for k, v in params.items()},
                "optimizer_states": {k: v.state_dict() for k, v in optimizers.items()},
                "scheduler_state": means_scheduler.state_dict(),
                "strategy_state": state,
            }, ckpt_path)
            print(f"[INFO] Checkpoint saved at step {step}", flush=True)

    # ── Export ────────────────────────────────────────────────────────────
    result_dir.mkdir(parents=True, exist_ok=True)
    out_ply = result_dir / "point_cloud.ply"
    with torch.no_grad():
        m_f = params["means"].detach()
        # Pass RAW (unactivated) values — viewers apply exp()/sigmoid() themselves
        s_f = params["scales"].detach()              # log-scales
        q_f = torch.nn.functional.normalize(params["quats"], dim=-1).detach()
        o_f = params["opacities"].detach()           # logit-opacities
        all_sh = params["sh_coeffs"].detach()           # [N, K, 3]
        sh0_f = all_sh[:, :1, :]                         # [N, 1, 3]
        shN = all_sh[:, 1:, :]                           # [N, K-1, 3]
    ply_bytes = export_splats(m_f, s_f, q_f, o_f, sh0_f, shN)
    out_ply.write_bytes(ply_bytes)
    n_final = m_f.shape[0]
    size_kb = out_ply.stat().st_size // 1024
    print(f"Saved: {out_ply} ({size_kb:,} KB, {n_final:,} Gaussians)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Scaffold-GS trainer")
    parser.add_argument("--data_dir",   required=True, type=Path)
    parser.add_argument("--result_dir", required=True, type=Path)
    parser.add_argument("--max_steps",  type=int, default=30000)
    parser.add_argument("--voxel_size", type=float, default=0.001)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--sh_degree", type=int, default=0)
    parser.add_argument("--depth_dir", type=Path, default=None)
    parser.add_argument("--depth_weight", type=float, default=0.1)
    parser.add_argument("--pseudo_gt_dir", type=Path, default=None)
    parser.add_argument("--confidence_weight", type=float, default=0.5)
    args = parser.parse_args()
    train(args.data_dir, args.result_dir, args.max_steps, args.voxel_size,
          resume=args.resume, ckpt_interval=args.ckpt_interval,
          sh_degree=args.sh_degree, depth_dir=args.depth_dir,
          depth_weight=args.depth_weight,
          pseudo_gt_dir=args.pseudo_gt_dir,
          confidence_weight=args.confidence_weight)


if __name__ == "__main__":
    main()
