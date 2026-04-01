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

from colmap_io import load_colmap_model, qvec_to_rotmat, get_intrinsics
from losses import ssim as compute_ssim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C0 = 0.28209479177387814  # SH coefficient for degree 0


# ──────────────────── Main ────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train a 3D Gaussian Splat from COLMAP data")
    ap.add_argument("--data_dir",   required=True, type=Path,
                    help="COLMAP image_undistorter output dir")
    ap.add_argument("--result_dir", required=True, type=Path,
                    help="Directory to write point_cloud.ply")
    ap.add_argument("--max_steps",  default=7000,  type=int,
                    help="Number of training iterations")
    ap.add_argument("--resume",     action="store_true",
                    help="Resume from checkpoint if available")
    ap.add_argument("--ckpt_interval", default=1000, type=int,
                    help="Save checkpoint every N steps")
    ap.add_argument("--sh_degree",  default=0, type=int,
                    help="Spherical harmonics degree (0-3, higher = view-dependent color)")
    ap.add_argument("--depth_dir",  default=None, type=Path,
                    help="Directory with .npy depth maps for depth supervision")
    ap.add_argument("--depth_weight", default=0.1, type=float,
                    help="Weight for depth loss (0.0 = disabled)")
    ap.add_argument("--pseudo_gt_dir", default=None, type=Path,
                    help="Directory with pseudo-GT and confidence maps from visibility transfer")
    ap.add_argument("--confidence_weight", default=0.5, type=float,
                    help="Blend weight for pseudo-GT in low-confidence regions")
    ap.add_argument("--novel_views_dir", default=None, type=Path,
                    help="Directory with inpainted novel view pseudo-GT for additional supervision")
    ap.add_argument("--novel_view_weight", default=0.3, type=float,
                    help="Loss weight for novel view supervision (lower = less trust)")
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
    try:
        model = load_colmap_model(sparse)
    except FileNotFoundError:
        print(f"[ERROR] No COLMAP model found in {sparse}", flush=True)
        sys.exit(1)

    cameras = model.cameras
    images_dat = model.images
    pts_xyz, pts_rgb = model.points_xyz, model.points_rgb

    print(
        f"[INFO] Loaded COLMAP ({model.fmt}): {len(cameras)} cam(s), "
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

    # Per-view resolution — supports multi-camera with different image sizes
    widths  = [cameras[img["cid"]]["W"] for img in sorted_imgs]
    heights = [cameras[img["cid"]]["H"] for img in sorted_imgs]
    n_unique_res = len(set(zip(widths, heights)))
    if n_unique_res == 1:
        print(f"[INFO] Building {len(sorted_imgs)} views at {widths[0]}×{heights[0]}", flush=True)
    else:
        print(f"[INFO] Building {len(sorted_imgs)} views with {n_unique_res} different resolutions", flush=True)

    viewmats_np, Ks_np, gt_list = [], [], []
    for i, img in enumerate(sorted_imgs):
        cam = cameras[img["cid"]]
        W_i, H_i = widths[i], heights[i]
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
                Image.open(img_path).convert("RGB").resize((W_i, H_i), Image.LANCZOS),
                dtype=np.float32,
            ) / 255.0
        else:
            print(f"[WARN] Image not found: {img_path}", flush=True)
            gt_np = np.zeros((H_i, W_i, 3), dtype=np.float32)
        gt_list.append(torch.from_numpy(gt_np).to(DEVICE))

    # Load depth maps if available
    depth_list = []
    use_depth = args.depth_dir is not None and args.depth_dir.exists() and args.depth_weight > 0
    if use_depth:
        for i, img in enumerate(sorted_imgs):
            depth_path = args.depth_dir / f"{Path(img['name']).stem}.npy"
            if depth_path.exists():
                d = np.load(depth_path).astype(np.float32)
                d = np.array(Image.fromarray(d).resize((widths[i], heights[i]), Image.BILINEAR))
                depth_list.append(torch.from_numpy(d).to(DEVICE))
            else:
                depth_list.append(None)
        n_depths = sum(1 for d in depth_list if d is not None)
        print(f"[INFO] Loaded {n_depths}/{len(sorted_imgs)} depth maps (weight={args.depth_weight})", flush=True)
        if n_depths == 0:
            use_depth = False

    # Load pseudo-GT and confidence maps from visibility transfer
    pseudo_gt_list = []
    confidence_list = []
    use_pseudo_gt = (
        args.pseudo_gt_dir is not None
        and args.pseudo_gt_dir.exists()
        and args.confidence_weight > 0
    )
    if use_pseudo_gt:
        for img in sorted_imgs:
            stem = Path(img["name"]).stem
            pg_path = args.pseudo_gt_dir / f"pseudo_gt_{stem}.npy"
            cf_path = args.pseudo_gt_dir / f"confidence_{stem}.npy"
            if pg_path.exists() and cf_path.exists():
                pg = torch.from_numpy(np.load(pg_path)).to(DEVICE)
                cf = torch.from_numpy(np.load(cf_path)).to(DEVICE)
                pseudo_gt_list.append(pg)
                confidence_list.append(cf)
            else:
                pseudo_gt_list.append(None)
                confidence_list.append(None)
        n_pg = sum(1 for p in pseudo_gt_list if p is not None)
        print(f"[INFO] Loaded {n_pg}/{len(sorted_imgs)} pseudo-GT maps (weight={args.confidence_weight})", flush=True)
        if n_pg == 0:
            use_pseudo_gt = False

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

    # Spherical harmonics coefficients: [P, K, 3] where K = (sh_degree+1)^2
    sh_degree = args.sh_degree
    K = (sh_degree + 1) ** 2
    sh_coeffs = torch.zeros(P, K, 3, device=DEVICE)
    sh_coeffs[:, 0, :] = (torch.from_numpy(pts_rgb).float().clamp(0.0, 1.0).to(DEVICE) - 0.5) / C0
    sh_coeffs = torch.nn.Parameter(sh_coeffs)
    print(f"[INFO] SH degree {sh_degree} ({K} coefficients per Gaussian)", flush=True)

    params = {
        "means":     means,
        "scales":    scales,
        "quats":     quats,
        "opacities": opacities,
        "sh_coeffs": sh_coeffs,
    }
    optimizers = {
        "means":     torch.optim.Adam([means],     lr=1.6e-4, eps=1e-15),
        "scales":    torch.optim.Adam([scales],    lr=5e-3,   eps=1e-15),
        "quats":     torch.optim.Adam([quats],     lr=1e-3,   eps=1e-15),
        "opacities": torch.optim.Adam([opacities], lr=5e-2,   eps=1e-15),
        "sh_coeffs": torch.optim.Adam([sh_coeffs], lr=2.5e-3, eps=1e-15),
    }
    # Progressive SH: start at degree 0, grow by 1 every interval
    sh_grow_interval = max(args.max_steps // (sh_degree + 1), 1) if sh_degree > 0 else args.max_steps

    strategy = DefaultStrategy(
        verbose=False,
        refine_start_iter=500,
        refine_stop_iter=min(args.max_steps - 200, 15_000),
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    # Learning rate scheduling: decay means LR from 1.6e-4 → 1.6e-6 over training
    means_lr_end = 1.6e-6
    means_lr_start = 1.6e-4
    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"],
        gamma=(means_lr_end / means_lr_start) ** (1.0 / max(args.max_steps, 1)),
    )

    # ── Checkpoint resume ────────────────────────────────────────────────
    start_step = 1
    ckpt_path = result_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
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

    print(
        f"[INFO] Training {args.max_steps} steps (starting at {start_step}) | "
        f"scene_scale={scene_scale:.4f} | {N} views",
        flush=True,
    )

    t0 = time.time()
    for step in range(start_step, args.max_steps + 1):
        idx = (step - 1) % N
        vm  = viewmats[idx : idx + 1]   # [1, 4, 4]
        Km  = Ks[idx : idx + 1]        # [1, 3, 3]
        gt  = gt_list[idx]              # [H, W, 3]

        q = F.normalize(params["quats"], dim=-1)

        # Progressive SH: grow degree over training
        cur_sh_degree = min(sh_degree, step // sh_grow_interval) if sh_degree > 0 else 0

        W_cur, H_cur = widths[idx], heights[idx]
        render_mode = "RGB+ED" if use_depth else "RGB"
        renders, _alphas, info = rasterization(
            means=params["means"],
            quats=q,
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=params["sh_coeffs"],
            viewmats=vm,
            Ks=Km,
            width=W_cur,
            height=H_cur,
            sh_degree=cur_sh_degree,
            packed=True,
            absgrad=False,
            render_mode=render_mode,
        )

        # Must retain grad on means2d before backward
        strategy.step_pre_backward(params, optimizers, state, step, info)

        if use_depth:
            pred = renders[0, ..., :3].clamp(0.0, 1.0)   # [H, W, 3]
            pred_depth = renders[0, ..., 3:4]              # [H, W, 1]
        else:
            pred = renders[0].clamp(0.0, 1.0)              # [H, W, 3]

        # Confidence-aware target: blend GT with pseudo-GT where confidence is low
        if use_pseudo_gt and pseudo_gt_list[idx] is not None:
            conf = confidence_list[idx].unsqueeze(-1)  # [H, W, 1]
            target = conf * gt + (1.0 - conf) * pseudo_gt_list[idx]
            # Weight map: zero out truly unseen pixels (contribute no loss)
            weight_map = (confidence_list[idx] > 0.01).float()
        else:
            target = gt
            weight_map = None

        if weight_map is not None:
            l1 = (weight_map.unsqueeze(-1) * (pred - target).abs()).sum() / weight_map.sum().clamp(min=1)
        else:
            l1 = F.l1_loss(pred, target)
        # SSIM needs [B, C, H, W] layout
        pred_bchw = pred.permute(2, 0, 1).unsqueeze(0)
        target_bchw = target.permute(2, 0, 1).unsqueeze(0)
        ssim_val = compute_ssim(pred_bchw, target_bchw)
        loss = 0.2 * l1 + 0.8 * (1.0 - ssim_val)

        # Depth supervision
        if use_depth and depth_list[idx] is not None:
            mono_depth = depth_list[idx]  # [H, W] normalized 0-1
            rd = pred_depth.squeeze(-1)   # [H, W]
            # Normalize rendered depth to [0, 1] for comparison
            rd_min, rd_max = rd.min(), rd.max()
            if rd_max - rd_min > 1e-6:
                rd_norm = (rd - rd_min) / (rd_max - rd_min)
            else:
                rd_norm = rd
            loss = loss + args.depth_weight * F.l1_loss(rd_norm, mono_depth)

        loss.backward()

        strategy.step_post_backward(params, optimizers, state, step, info, packed=True)

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_scheduler.step()

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

        # Save snapshot at 25% intervals
        snapshot_steps = {
            int(args.max_steps * p): p_int
            for p, p_int in [(0.25, 25), (0.5, 50), (0.75, 75), (1.0, 100)]
        }
        if step in snapshot_steps:
            pct = snapshot_steps[step]
            snap_path = result_dir / f"snapshot_{pct}.jpg"
            snap_img = (pred.detach().clamp(0, 1) * 255).byte().cpu().numpy()
            Image.fromarray(snap_img).save(str(snap_path), quality=90)
            print(f"[SNAPSHOT] {pct} snapshot_{pct}.jpg", flush=True)

        # Save checkpoint periodically
        if step % args.ckpt_interval == 0:
            torch.save({
                "step": step,
                "params": {k: v.data.clone() for k, v in params.items()},
                "optimizer_states": {k: v.state_dict() for k, v in optimizers.items()},
                "scheduler_state": means_scheduler.state_dict(),
                "strategy_state": state,
            }, ckpt_path)
            print(f"[INFO] Checkpoint saved at step {step}", flush=True)

    # ── Export PLY ─────────────────────────────────────────────────────────
    ply_path = result_dir / "point_cloud.ply"
    print(f"\n[INFO] Exporting PLY to {ply_path}", flush=True)

    with torch.no_grad():
        m_f  = params["means"].detach()
        # Pass RAW (unactivated) values — viewers apply exp()/sigmoid() themselves
        s_f  = params["scales"].detach()            # log-scales (viewer applies exp)
        q_f  = F.normalize(params["quats"], dim=-1).detach()
        o_f  = params["opacities"].detach()         # logit-opacities (viewer applies sigmoid)

        # SH coefficients: split into sh0 (degree 0) and shN (higher degrees)
        all_sh = params["sh_coeffs"].detach()           # [P, K, 3]
        sh0 = all_sh[:, :1, :]                          # [P, 1, 3]
        shN = all_sh[:, 1:, :]                          # [P, K-1, 3]

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
