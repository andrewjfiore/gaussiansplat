#!/usr/bin/env python3
"""
train_4d.py — Deformable 4D Gaussian Splatting trainer.

Trains a canonical set of Gaussians plus a deformation MLP that predicts
per-Gaussian position/rotation/scale offsets at each timestamp. The canonical
Gaussians represent the scene's "average" state, and the MLP learns how
each Gaussian moves over time.

Input: COLMAP dense directory + frame manifest with timestamps
Output: canonical point_cloud.ply + deformation.pt + per-frame baked PLYs

Usage:
  python train_4d.py --data_dir <colmap_dense> --result_dir <out> \
      --manifest <frames/manifest.json> --max_steps 10000
"""

import argparse
import json
import math
import random
import signal
import sys
import time
from pathlib import Path

# Graceful early-stop: catch SIGTERM/SIGINT to break training loop and still export PLY
_stop_requested = False

def _handle_stop(signum, frame):
    global _stop_requested
    _stop_requested = True
    print(f"\n[INFO] Stop requested (signal {signum}) — finishing current step and exporting...", flush=True)

signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from gsplat.exporter import export_splats

from colmap_io import load_colmap_model, qvec_to_rotmat, get_intrinsics
from losses import ssim as compute_ssim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C0 = 0.28209479177387814


# ──────────────────── Deformation MLP ────────────────────

class DeformationMLP(nn.Module):
    """Predicts per-Gaussian offsets given time and Gaussian index."""

    def __init__(self, n_gaussians: int, embed_dim: int = 16, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_gaussians = n_gaussians

        # Time encoding: sinusoidal positional encoding
        self.time_freqs = 8  # 8 frequency bands
        time_input_dim = 1 + 2 * self.time_freqs  # 1 raw + 16 sinusoidal = 17

        self.time_mlp = nn.Sequential(
            nn.Linear(time_input_dim, hidden),
            nn.ReLU(),
        )

        # Per-Gaussian learnable embeddings
        self.gaussian_embed = nn.Embedding(n_gaussians, embed_dim)
        nn.init.normal_(self.gaussian_embed.weight, std=0.01)

        # Shared MLP
        layers = [nn.Linear(embed_dim + hidden, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        self.mlp = nn.Sequential(*layers)

        # Output heads (initialized near zero for small initial deformations)
        self.head_pos = nn.Linear(hidden, 3)
        self.head_rot = nn.Linear(hidden, 4)
        self.head_scale = nn.Linear(hidden, 3)

        nn.init.zeros_(self.head_pos.weight)
        nn.init.zeros_(self.head_pos.bias)
        nn.init.zeros_(self.head_rot.weight)
        nn.init.zeros_(self.head_rot.bias)
        nn.init.zeros_(self.head_scale.weight)
        nn.init.zeros_(self.head_scale.bias)

    def _encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal positional encoding for time."""
        freqs = 2.0 ** torch.arange(self.time_freqs, device=t.device).float()
        t_scaled = t.unsqueeze(-1) * freqs  # [1, n_freqs]
        return torch.cat([t.unsqueeze(-1), t_scaled.sin(), t_scaled.cos()], dim=-1)

    def forward(self, t: torch.Tensor, gaussian_ids: torch.Tensor):
        """
        Args:
            t: scalar tensor, normalized time in [0, 1]
            gaussian_ids: [P] long tensor of Gaussian indices

        Returns:
            d_pos: [P, 3] position offsets
            d_rot: [P, 4] rotation offsets (quaternion, small)
            d_scale: [P, 3] log-scale offsets
        """
        P = gaussian_ids.shape[0]
        t_enc = self._encode_time(t)                     # [1, 17]
        t_feat = self.time_mlp(t_enc)                     # [1, hidden]
        g_feat = self.gaussian_embed(gaussian_ids)        # [P, embed_dim]
        feat = torch.cat([g_feat, t_feat.expand(P, -1)], dim=-1)
        h = self.mlp(feat)
        return self.head_pos(h), self.head_rot(h), self.head_scale(h)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternion tensors (wxyz convention)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


# ──────────────────── Main ────────────────────

def scale_invariant_depth_loss(
    rendered_depth: torch.Tensor,
    mono_depth: torch.Tensor,
    alpha_mask: torch.Tensor,
    min_valid: int = 100,
) -> torch.Tensor:
    """
    Compute scale-and-shift-invariant depth loss between rendered depth and
    monocular (relative) depth.

    MiDAS outputs inverse-depth-like values with arbitrary per-frame scale/shift.
    We solve for optimal scale s and shift t that align rendered depth to mono
    via least-squares, then compute L1 on the aligned result.

    Args:
        rendered_depth: [H, W] expected depth from rasterizer
        mono_depth:     [H, W] monocular depth (normalised 0-1, higher = farther)
        alpha_mask:     [H, W] opacity mask from rasterizer (>0 where Gaussians render)
        min_valid:      minimum number of valid pixels required

    Returns:
        Scalar loss, or 0 if insufficient valid pixels.
    """
    # Only supervise pixels where Gaussians actually rendered
    mask = alpha_mask > 0.1
    if mask.sum() < min_valid:
        return torch.tensor(0.0, device=rendered_depth.device)

    rd = rendered_depth[mask]  # rendered
    md = mono_depth[mask]      # monocular

    # Guard against degenerate monocular depth
    if md.max() - md.min() < 1e-6:
        return torch.tensor(0.0, device=rendered_depth.device)

    # Solve for scale s and shift t:  rd ≈ s * md + t
    # Using least-squares: [md, 1] @ [s, t]^T = rd
    A = torch.stack([md, torch.ones_like(md)], dim=-1)  # [N, 2]
    # Normal equations: (A^T A) x = A^T b
    AtA = A.T @ A       # [2, 2]
    Atb = A.T @ rd       # [2]
    try:
        x = torch.linalg.solve(AtA, Atb)  # [s, t]
    except torch.linalg.LinAlgError:
        # Degenerate — fall back to median alignment
        s = rd.median() / md.median().clamp(min=1e-6)
        t = torch.tensor(0.0, device=rd.device)
        x = torch.stack([s, t])

    aligned_mono = x[0] * md + x[1]
    return F.l1_loss(rd, aligned_mono)


def main():
    ap = argparse.ArgumentParser(description="Train a 4D Deformable Gaussian Splat")
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--result_dir", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--max_steps", default=10000, type=int)
    ap.add_argument("--sh_degree", default=0, type=int)
    ap.add_argument("--temporal_smoothness", default=0.01, type=float)
    ap.add_argument("--ckpt_interval", default=1000, type=int)
    ap.add_argument("--depth_dir", type=Path, default=None)
    ap.add_argument("--depth_weight", type=float, default=0.1)
    ap.add_argument("--max_gaussians", type=int, default=200_000,
                     help="Cap Gaussian count to prevent VRAM exhaustion")
    args = ap.parse_args()

    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    # ── Load manifest ─────────────────────────────────────────────────
    with open(args.manifest) as f:
        manifest = json.load(f)
    timestamps = [fr["t_norm"] for fr in manifest["frames"]]
    frame_files = [fr["file"] for fr in manifest["frames"]]
    print(f"[INFO] Loaded manifest: {len(timestamps)} frames, duration={manifest.get('duration', 0):.1f}s", flush=True)

    # ── Load COLMAP model ─────────────────────────────────────────────
    sparse = args.data_dir / "sparse"
    imgs_dir = args.data_dir / "images"

    try:
        model = load_colmap_model(sparse)
    except FileNotFoundError:
        print(f"[ERROR] No COLMAP model found in {sparse}", flush=True)
        sys.exit(1)

    cameras = model.cameras
    images_dat = model.images
    pts_xyz, pts_rgb = model.points_xyz, model.points_rgb

    print(f"[INFO] Device: {DEVICE}", flush=True)
    print(f"[INFO] COLMAP: {len(cameras)} cam(s), {len(images_dat)} images, {len(pts_xyz)} points", flush=True)

    if len(pts_xyz) < 50:
        pts_xyz = np.random.randn(5000, 3).astype(np.float32) * 2.0
        pts_rgb = np.random.rand(5000, 3).astype(np.float32)

    # ── Build per-image data ──────────────────────────────────────────
    sorted_imgs = sorted(images_dat.values(), key=lambda x: x["name"])

    # Per-view resolution — supports multi-camera with different image sizes
    widths  = [cameras[img["cid"]]["W"] for img in sorted_imgs]
    heights = [cameras[img["cid"]]["H"] for img in sorted_imgs]

    viewmats_np, Ks_np, gt_list = [], [], []
    frame_to_view = {}  # Map frame filename to view index

    for i, img in enumerate(sorted_imgs):
        cam = cameras[img["cid"]]
        W_i, H_i = widths[i], heights[i]
        R = qvec_to_rotmat(img["qvec"])
        t = np.array(img["tvec"])
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = R.astype(np.float32)
        vm[:3, 3] = t.astype(np.float32)
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
            gt_np = np.zeros((H_i, W_i, 3), dtype=np.float32)
        gt_list.append(torch.from_numpy(gt_np).to(DEVICE))
        frame_to_view[img["name"]] = i

    viewmats = torch.from_numpy(np.stack(viewmats_np)).to(DEVICE)
    Ks = torch.from_numpy(np.stack(Ks_np)).to(DEVICE)
    N = len(sorted_imgs)

    # Build timestamp mapping: frame index -> normalized time
    frame_timestamps = []
    for fr in manifest["frames"]:
        view_idx = frame_to_view.get(fr["file"])
        if view_idx is not None:
            frame_timestamps.append((view_idx, fr["t_norm"]))
    if not frame_timestamps:
        # Fallback: assign uniform timestamps
        frame_timestamps = [(i, i / max(N - 1, 1)) for i in range(N)]

    n_res = len(set(zip(widths, heights)))
    print(f"[INFO] Matched {len(frame_timestamps)} frames to views ({n_res} resolution{'s' if n_res > 1 else ''})", flush=True)

    # ── Load depth maps ──────────────────────────────────────────────
    use_depth = args.depth_dir is not None and args.depth_dir.exists() and args.depth_weight > 0
    depth_map_paths: dict[int, Path] = {}
    if use_depth:
        for view_idx in range(N):
            stem = Path(sorted_imgs[view_idx]["name"]).stem
            depth_path = args.depth_dir / f"{stem}.npy"
            if depth_path.exists():
                depth_map_paths[view_idx] = depth_path
        print(f"[INFO] Depth maps: {len(depth_map_paths)}/{N} available (weight={args.depth_weight})", flush=True)
        if not depth_map_paths:
            use_depth = False
            print("[WARN] No depth maps matched — depth supervision disabled", flush=True)

    # ── Initialize Gaussians ──────────────────────────────────────────
    P = len(pts_xyz)
    means = torch.nn.Parameter(torch.from_numpy(pts_xyz).float().to(DEVICE))
    quats = torch.nn.Parameter(torch.zeros(P, 4, device=DEVICE))
    quats.data[:, 0] = 1.0

    sample_pts = means.data[:min(P, 1000)]
    if sample_pts.shape[0] > 1:
        dists = torch.cdist(sample_pts, sample_pts)
        nonzero = dists[dists > 0]
        scene_scale = float(nonzero.median().item()) if nonzero.numel() > 0 else 1.0
    else:
        scene_scale = 1.0

    init_log_scale = math.log(max(scene_scale * 0.02, 1e-6))
    scales = torch.nn.Parameter(torch.full((P, 3), init_log_scale, device=DEVICE))
    opacities = torch.nn.Parameter(torch.full((P,), -3.0, device=DEVICE))

    sh_degree = args.sh_degree
    num_sh = (sh_degree + 1) ** 2
    sh_coeffs = torch.zeros(P, num_sh, 3, device=DEVICE)
    sh_coeffs[:, 0, :] = (torch.from_numpy(pts_rgb).float().clamp(0.0, 1.0).to(DEVICE) - 0.5) / C0
    sh_coeffs = torch.nn.Parameter(sh_coeffs)

    print(f"[INFO] Initialized {P:,} Gaussians, SH degree {sh_degree}", flush=True)

    # ── Deformation MLP ───────────────────────────────────────────────
    # Cap embedding count for VRAM: if too many Gaussians, use modular indexing
    MAX_EMBEDS = min(P, 200_000)
    deform_mlp = DeformationMLP(MAX_EMBEDS, embed_dim=16, hidden=64, n_layers=3).to(DEVICE)
    deform_param_count = sum(p.numel() for p in deform_mlp.parameters())
    print(f"[INFO] Deformation MLP: {deform_param_count:,} parameters (max {MAX_EMBEDS:,} embeddings)", flush=True)

    # ── Optimizers ────────────────────────────────────────────────────
    params = {
        "means": means, "scales": scales, "quats": quats,
        "opacities": opacities, "sh_coeffs": sh_coeffs,
    }
    optimizers = {
        "means":     torch.optim.Adam([means], lr=1.6e-4, eps=1e-15),
        "scales":    torch.optim.Adam([scales], lr=5e-3, eps=1e-15),
        "quats":     torch.optim.Adam([quats], lr=1e-3, eps=1e-15),
        "opacities": torch.optim.Adam([opacities], lr=5e-2, eps=1e-15),
        "sh_coeffs": torch.optim.Adam([sh_coeffs], lr=2.5e-3, eps=1e-15),
    }
    deform_optimizer = torch.optim.Adam(deform_mlp.parameters(), lr=5e-4)

    strategy = DefaultStrategy(verbose=False, refine_start_iter=500,
                               refine_stop_iter=min(args.max_steps - 200, 15_000))
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=(1.6e-6 / 1.6e-4) ** (1.0 / max(args.max_steps, 1)))

    # ── Training loop ─────────────────────────────────────────────────
    print(f"[INFO] Training {args.max_steps} steps | 4D mode | {len(frame_timestamps)} temporal frames", flush=True)

    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        if _stop_requested:
            print(f"[INFO] Early stop at step {step - 1}/{args.max_steps}", flush=True)
            break
        # Random frame with timestamp
        view_idx, t_norm = random.choice(frame_timestamps)
        vm = viewmats[view_idx:view_idx + 1]
        Km = Ks[view_idx:view_idx + 1]
        gt = gt_list[view_idx]

        # Compute deformations at time t
        P_cur = params["means"].shape[0]
        t_tensor = torch.tensor(t_norm, device=DEVICE, dtype=torch.float32)
        ids = torch.arange(P_cur, device=DEVICE) % MAX_EMBEDS  # Modular for densified Gaussians
        d_pos, d_rot, d_scale = deform_mlp(t_tensor, ids)

        # Apply deformations to canonical Gaussians
        means_t = params["means"] + d_pos
        rot_offset = F.normalize(d_rot + torch.tensor([1, 0, 0, 0], device=DEVICE, dtype=torch.float32), dim=-1)
        quats_t = quaternion_multiply(F.normalize(params["quats"], dim=-1), rot_offset)
        scales_t = params["scales"] + d_scale

        # Rasterize
        cur_sh = min(sh_degree, step // max(args.max_steps // (sh_degree + 1), 1)) if sh_degree > 0 else 0
        render_mode = "RGB+ED" if use_depth else "RGB"
        renders, alphas, info = rasterization(
            means=means_t, quats=F.normalize(quats_t, dim=-1),
            scales=torch.exp(scales_t),
            opacities=torch.sigmoid(params["opacities"]),
            colors=params["sh_coeffs"],
            viewmats=vm, Ks=Km, width=widths[view_idx], height=heights[view_idx],
            sh_degree=cur_sh, packed=True, absgrad=False,
            render_mode=render_mode,
        )

        if use_depth:
            rgb_render = renders[0, ..., :3]
            depth_render = renders[0, ..., 3]   # [H, W] expected depth
        else:
            rgb_render = renders[0]

        strategy.step_pre_backward(params, optimizers, state, step, info)

        pred = rgb_render.clamp(0.0, 1.0)
        # TODO: confidence-aware loss for 4D (load pseudo-GT when --pseudo_gt_dir provided)
        target = gt
        l1 = F.l1_loss(pred, target)
        pred_bchw = pred.permute(2, 0, 1).unsqueeze(0)
        target_bchw = target.permute(2, 0, 1).unsqueeze(0)
        ssim_val = compute_ssim(pred_bchw, target_bchw)
        rgb_loss = 0.2 * l1 + 0.8 * (1.0 - ssim_val)

        # Depth supervision (scale-invariant against monocular depth)
        depth_loss = torch.tensor(0.0, device=DEVICE)
        if use_depth and view_idx in depth_map_paths:
            # Load depth map on-demand (keeps VRAM free)
            mono_np = np.load(depth_map_paths[view_idx]).astype(np.float32)
            mono_depth = torch.from_numpy(mono_np).to(DEVICE)
            if mono_depth.shape != (heights[view_idx], widths[view_idx]):
                mono_depth = F.interpolate(
                    mono_depth.unsqueeze(0).unsqueeze(0),
                    size=(heights[view_idx], widths[view_idx]), mode="bilinear", align_corners=False,
                ).squeeze()
            # Warm-up: ramp depth weight from 0 → full over first 20% of training
            warmup_frac = min(step / (args.max_steps * 0.2), 1.0)
            alpha_map = alphas[0, ..., 0] if alphas.dim() == 4 else alphas[0]  # [H, W]
            depth_loss = scale_invariant_depth_loss(
                depth_render, mono_depth, alpha_map,
            )
            depth_loss = args.depth_weight * warmup_frac * depth_loss

        # Temporal smoothness: penalize large deformations
        temporal_loss = (d_pos ** 2).mean() + 0.1 * (d_rot ** 2).mean() + 0.1 * (d_scale ** 2).mean()
        loss = rgb_loss + args.temporal_smoothness * temporal_loss + depth_loss

        loss.backward()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=True)

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        deform_optimizer.step()
        deform_optimizer.zero_grad(set_to_none=True)
        means_scheduler.step()

        with torch.no_grad():
            params["quats"].data.copy_(F.normalize(params["quats"].data, dim=-1))

        # Cap Gaussian count: stop refinement early if over budget
        P_now = params["means"].shape[0]
        if P_now > args.max_gaussians and strategy.refine_stop_iter > step:
            strategy.refine_stop_iter = step
            print(f"[INFO] Gaussian cap reached ({P_now:,} > {args.max_gaussians:,}) — "
                  f"densification stopped at step {step}", flush=True)

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - t0
            n_pts = params["means"].shape[0]
            mse = F.mse_loss(pred.detach(), gt).item()
            psnr = -10.0 * math.log10(max(mse, 1e-10))
            depth_str = f", depth={depth_loss.item():.4f}" if use_depth else ""
            print(f"Step {step}/{args.max_steps}, loss={loss.item():.4f}, "
                  f"psnr={psnr:.2f}, pts={n_pts:,}, t={t_norm:.3f}{depth_str}, "
                  f"elapsed={elapsed:.1f}s", flush=True)

        # Snapshots at 25% intervals
        snapshot_steps = {int(args.max_steps * p): pct for p, pct in [(0.25, 25), (0.5, 50), (0.75, 75), (1.0, 100)]}
        if step in snapshot_steps:
            pct = snapshot_steps[step]
            snap_img = (pred.detach().clamp(0, 1) * 255).byte().cpu().numpy()
            Image.fromarray(snap_img).save(str(result_dir / f"snapshot_{pct}.jpg"), quality=90)
            print(f"[SNAPSHOT] {pct} snapshot_{pct}.jpg", flush=True)

        if step % args.ckpt_interval == 0:
            torch.save({
                "step": step, "params": {k: v.data.clone() for k, v in params.items()},
                "deform_mlp": deform_mlp.state_dict(),
                "optimizer_states": {k: v.state_dict() for k, v in optimizers.items()},
                "deform_optimizer": deform_optimizer.state_dict(),
            }, result_dir / "checkpoint.pt")
            print(f"[INFO] Checkpoint saved at step {step}", flush=True)

    # ── Export canonical PLY ──────────────────────────────────────────
    print(f"\n[INFO] Exporting canonical PLY + deformation weights", flush=True)
    with torch.no_grad():
        m_f = params["means"].detach()
        s_f = params["scales"].detach()
        q_f = F.normalize(params["quats"], dim=-1).detach()
        o_f = params["opacities"].detach()
        all_sh = params["sh_coeffs"].detach()
        sh0 = all_sh[:, :1, :]
        shN = all_sh[:, 1:, :]

    ply_bytes = export_splats(m_f, s_f, q_f, o_f, sh0, shN)
    (result_dir / "point_cloud.ply").write_bytes(ply_bytes)

    # Save deformation MLP
    torch.save({
        "mlp_state": deform_mlp.state_dict(),
        "max_embeds": MAX_EMBEDS,
        "n_gaussians": params["means"].shape[0],
        "timestamps": timestamps,
    }, result_dir / "deformation.pt")

    # ── Bake per-frame PLYs ───────────────────────────────────────────
    print(f"[INFO] Baking per-frame PLYs...", flush=True)
    frames_out = result_dir / "temporal_frames"
    frames_out.mkdir(exist_ok=True)

    deform_mlp.eval()
    with torch.no_grad():
        P_final = params["means"].shape[0]
        ids = torch.arange(P_final, device=DEVICE) % MAX_EMBEDS

        for i, t_val in enumerate(timestamps):
            t_tensor = torch.tensor(t_val, device=DEVICE, dtype=torch.float32)
            d_pos, d_rot, d_scale = deform_mlp(t_tensor, ids)

            means_t = params["means"] + d_pos
            rot_offset = F.normalize(d_rot + torch.tensor([1, 0, 0, 0], device=DEVICE), dim=-1)
            quats_t = quaternion_multiply(F.normalize(params["quats"], dim=-1), rot_offset)
            scales_t = params["scales"] + d_scale

            frame_ply = export_splats(means_t, scales_t,
                                      F.normalize(quats_t, dim=-1),
                                      params["opacities"], sh0, shN)
            (frames_out / f"frame_{i:04d}.ply").write_bytes(frame_ply)

            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Baked frame {i+1}/{len(timestamps)}", flush=True)

    total = time.time() - t0
    print(f"[INFO] Done! Canonical PLY + {len(timestamps)} temporal frames in {total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
