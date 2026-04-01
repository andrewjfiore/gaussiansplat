#!/usr/bin/env python3
"""
visibility_transfer.py — Cross-frame geometric visibility transfer.

After an initial GS training pass, renders each camera view, identifies
low-confidence regions (alpha < threshold), and fills them by projecting
into other views that DO see those 3D points. Produces pseudo-GT images,
confidence maps, and unseen masks for refinement training.

Usage:
  python visibility_transfer.py \
    --data_dir <colmap_dense_dir> \
    --checkpoint <path/to/checkpoint.pt or point_cloud.ply> \
    --output_dir <visibility_output_dir>
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

from colmap_io import load_colmap_model, qvec_to_rotmat, get_intrinsics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C0 = 0.28209479177387814


def load_gaussians_from_checkpoint(ckpt_path: Path) -> dict:
    """Load Gaussian parameters from a training checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    params = {}
    for k, v in ckpt["params"].items():
        params[k] = v.to(DEVICE)
    return params


def load_gaussians_from_ply(ply_path: Path) -> dict:
    """Load Gaussian parameters from an exported PLY file."""
    from gsplat.exporter import export_splats  # noqa — ensure gsplat is available
    import struct

    with open(ply_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break

        header_str = header.decode("utf-8", errors="replace")
        n_verts = 0
        props = []
        for hline in header_str.split("\n"):
            if hline.startswith("element vertex"):
                n_verts = int(hline.split()[-1])
            if hline.startswith("property float"):
                props.append(hline.split()[-1])

        n_floats = len(props)
        data = np.frombuffer(f.read(n_verts * n_floats * 4), dtype=np.float32)
        data = data.reshape(n_verts, n_floats)

    # Standard 3DGS PLY layout: x,y,z, f_dc_0..2, opacity, scale_0..2, rot_0..3
    means = torch.from_numpy(data[:, 0:3]).to(DEVICE)

    # Find property indices
    prop_idx = {name: i for i, name in enumerate(props)}

    # DC color (SH degree 0)
    dc_cols = []
    for c in ["f_dc_0", "f_dc_1", "f_dc_2"]:
        if c in prop_idx:
            dc_cols.append(prop_idx[c])
    sh0 = torch.from_numpy(data[:, dc_cols]).to(DEVICE).unsqueeze(1)  # [P, 1, 3]

    opacity = torch.from_numpy(data[:, prop_idx["opacity"]]).to(DEVICE)
    scales = torch.from_numpy(
        data[:, [prop_idx["scale_0"], prop_idx["scale_1"], prop_idx["scale_2"]]]
    ).to(DEVICE)
    quats = torch.from_numpy(
        data[:, [prop_idx["rot_0"], prop_idx["rot_1"], prop_idx["rot_2"], prop_idx["rot_3"]]]
    ).to(DEVICE)

    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacity,
        "sh_coeffs": sh0,
    }


def render_view(params, viewmat, K, W, H):
    """Render a single view and return RGB, depth, alpha (all on CPU)."""
    with torch.no_grad():
        q = F.normalize(params["quats"], dim=-1)
        renders, alphas, _ = rasterization(
            means=params["means"],
            quats=q,
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=params["sh_coeffs"],
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            sh_degree=0,
            packed=False,
            render_mode="RGB+ED",
        )
        rgb = renders[0, ..., :3].clamp(0.0, 1.0).cpu().numpy()       # [H, W, 3]
        depth = renders[0, ..., 3].cpu().numpy()                        # [H, W]
        alpha = alphas[0, ..., 0].cpu().numpy()                         # [H, W]
    return rgb, depth, alpha


def unproject_pixels(us, vs, depths, K_inv, R_T, t):
    """Unproject pixel coords to 3D world coordinates.

    Args:
        us, vs: [M] pixel coordinates
        depths: [M] depth values
        K_inv: [3, 3] inverse intrinsic matrix
        R_T: [3, 3] transposed rotation (world-to-cam R transposed = cam-to-world rotation)
        t: [3] translation vector from viewmat

    Returns:
        points_3d: [M, 3] world coordinates
    """
    ones = np.ones_like(us)
    pixels_h = np.stack([us, vs, ones], axis=-1)  # [M, 3]
    cam_pts = (K_inv @ pixels_h.T).T * depths[:, None]  # [M, 3] in camera space
    world_pts = (R_T @ (cam_pts - t[None, :]).T).T  # [M, 3] in world space
    return world_pts


def project_to_view(points_3d, K, R, t, W, H):
    """Project 3D points into a camera view.

    Returns:
        us, vs: [M] pixel coordinates (float)
        zs: [M] depth in camera frame
        valid: [M] bool mask for in-bounds + positive depth
    """
    cam_pts = (R @ points_3d.T).T + t[None, :]  # [M, 3]
    zs = cam_pts[:, 2]
    proj = (K @ cam_pts.T).T  # [M, 3]
    us = proj[:, 0] / (zs + 1e-8)
    vs = proj[:, 1] / (zs + 1e-8)
    valid = (zs > 0.01) & (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
    return us, vs, zs, valid


def main():
    ap = argparse.ArgumentParser(description="Cross-frame visibility transfer")
    ap.add_argument("--data_dir", required=True, type=Path,
                    help="COLMAP image_undistorter output dir")
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="Training checkpoint (.pt) or exported PLY (.ply)")
    ap.add_argument("--output_dir", required=True, type=Path,
                    help="Directory to write pseudo-GT, confidence, and unseen masks")
    ap.add_argument("--alpha_low", type=float, default=0.5,
                    help="Alpha threshold below which pixels are considered low-confidence")
    ap.add_argument("--alpha_high", type=float, default=0.8,
                    help="Alpha threshold above which source pixels are considered reliable")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load COLMAP data ──────────────────────────────────────
    sparse_dir = args.data_dir / "sparse"
    model = load_colmap_model(sparse_dir)
    cameras, images_dat = model.cameras, model.images

    sorted_imgs = sorted(images_dat.values(), key=lambda x: x["name"])
    N = len(sorted_imgs)
    imgs_dir = args.data_dir / "images"

    # Per-view resolution
    widths  = [cameras[img["cid"]]["W"] for img in sorted_imgs]
    heights = [cameras[img["cid"]]["H"] for img in sorted_imgs]
    print(f"[INFO] {N} views", flush=True)

    # Build camera data
    viewmats_np = []
    Ks_np = []
    gt_images = []
    for i, img in enumerate(sorted_imgs):
        cam = cameras[img["cid"]]
        W_i, H_i = widths[i], heights[i]
        R = qvec_to_rotmat(img["qvec"]).astype(np.float32)
        t = np.array(img["tvec"], dtype=np.float32)
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = R
        vm[:3, 3] = t
        viewmats_np.append(vm)

        fx, fy, cx, cy = get_intrinsics(cam)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        Ks_np.append(K)

        img_path = imgs_dir / img["name"]
        if img_path.exists():
            gt = np.array(
                Image.open(img_path).convert("RGB").resize((W_i, H_i), Image.LANCZOS),
                dtype=np.float32,
            ) / 255.0
        else:
            gt = np.zeros((H_i, W_i, 3), dtype=np.float32)
        gt_images.append(gt)

    # ── Load Gaussians ────────────────────────────────────────
    if args.checkpoint.suffix == ".ply":
        print(f"[INFO] Loading Gaussians from PLY: {args.checkpoint}", flush=True)
        params = load_gaussians_from_ply(args.checkpoint)
    else:
        print(f"[INFO] Loading Gaussians from checkpoint: {args.checkpoint}", flush=True)
        params = load_gaussians_from_checkpoint(args.checkpoint)
    print(f"[INFO] {params['means'].shape[0]:,} Gaussians loaded", flush=True)

    # ── Render all views ──────────────────────────────────────
    print("[INFO] Rendering all views...", flush=True)
    rendered_rgbs = []
    rendered_depths = []
    rendered_alphas = []

    for i in range(N):
        vm_t = torch.from_numpy(viewmats_np[i]).to(DEVICE)
        K_t = torch.from_numpy(Ks_np[i]).to(DEVICE)
        rgb, depth, alpha = render_view(params, vm_t, K_t, widths[i], heights[i])
        rendered_rgbs.append(rgb)
        rendered_depths.append(depth)
        rendered_alphas.append(alpha)
        if (i + 1) % 10 == 0 or i == N - 1:
            print(f"  Rendered {i + 1}/{N}", flush=True)

    # Free GPU memory — rest is CPU
    del params
    torch.cuda.empty_cache()

    # ── Cross-frame visibility transfer ───────────────────────
    print("[INFO] Running cross-frame visibility transfer...", flush=True)
    t_start = time.time()

    K_invs = [np.linalg.inv(K) for K in Ks_np]

    for i in range(N):
        alpha_i = rendered_alphas[i]
        depth_i = rendered_depths[i]
        gt_i = gt_images[i].copy()

        # Confidence starts at alpha (high alpha = high confidence in existing rendering)
        confidence = alpha_i.copy()

        # Find low-confidence pixels
        low_mask = alpha_i < args.alpha_low
        low_ys, low_xs = np.where(low_mask)

        if len(low_ys) == 0:
            # No low-confidence pixels — save originals
            pseudo_gt = gt_i
        else:
            # Filter to pixels that have SOME depth (alpha > 0 but below threshold)
            has_depth = depth_i[low_ys, low_xs] > 0.01
            transferable_ys = low_ys[has_depth]
            transferable_xs = low_xs[has_depth]

            pseudo_gt = gt_i.copy()

            if len(transferable_ys) > 0:
                # Unproject to 3D
                R_i = viewmats_np[i][:3, :3]
                t_i = viewmats_np[i][:3, 3]
                depths_sel = depth_i[transferable_ys, transferable_xs]

                pts_3d = unproject_pixels(
                    transferable_xs.astype(np.float32),
                    transferable_ys.astype(np.float32),
                    depths_sel,
                    K_invs[i], R_i.T, t_i,
                )

                # Find best source view for each point
                best_colors = np.zeros((len(transferable_ys), 3), dtype=np.float32)
                best_alphas = np.zeros(len(transferable_ys), dtype=np.float32)

                for j in range(N):
                    if j == i:
                        continue
                    R_j = viewmats_np[j][:3, :3]
                    t_j = viewmats_np[j][:3, 3]

                    us_j, vs_j, _, valid = project_to_view(pts_3d, Ks_np[j], R_j, t_j, widths[j], heights[j])

                    if not valid.any():
                        continue

                    us_int = us_j[valid].astype(int).clip(0, W - 1)
                    vs_int = vs_j[valid].astype(int).clip(0, H - 1)
                    alpha_j_at = rendered_alphas[j][vs_int, us_int]

                    # Only use pixels where source view has high confidence
                    good = alpha_j_at > args.alpha_high
                    if not good.any():
                        continue

                    # Among valid+good pixels, update if this source has higher alpha
                    valid_indices = np.where(valid)[0]
                    good_indices = valid_indices[good]
                    better = alpha_j_at[good] > best_alphas[good_indices]

                    if better.any():
                        update_idx = good_indices[better]
                        src_us = us_int[good][better]
                        src_vs = vs_int[good][better]
                        best_colors[update_idx] = gt_images[j][src_vs, src_us]
                        best_alphas[update_idx] = alpha_j_at[good][better]

                # Apply transferred colors
                transferred = best_alphas > 0
                if transferred.any():
                    t_ys = transferable_ys[transferred]
                    t_xs = transferable_xs[transferred]
                    pseudo_gt[t_ys, t_xs] = best_colors[transferred]
                    # Confidence for transferred pixels: scaled by source alpha
                    confidence[t_ys, t_xs] = best_alphas[transferred] * 0.7  # discount factor

        # Truly unseen: low alpha AND no transfer found
        unseen = (alpha_i < 0.1) & (confidence < 0.1)

        # Save outputs
        name_stem = Path(sorted_imgs[i]["name"]).stem
        np.save(args.output_dir / f"pseudo_gt_{name_stem}.npy", pseudo_gt.astype(np.float32))
        np.save(args.output_dir / f"confidence_{name_stem}.npy", confidence.astype(np.float32))

        unseen_img = Image.fromarray((unseen * 255).astype(np.uint8))
        unseen_img.save(args.output_dir / f"unseen_{name_stem}.png")

        if (i + 1) % 10 == 0 or i == N - 1:
            n_low = low_mask.sum()
            n_transferred = (confidence[low_ys, low_xs] > 0.1).sum() if len(low_ys) > 0 else 0
            n_unseen = unseen.sum()
            print(
                f"  View {i + 1}/{N}: {n_low} low-conf pixels, "
                f"{n_transferred} transferred, {n_unseen} truly unseen",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(f"[INFO] Visibility transfer complete in {elapsed:.1f}s", flush=True)
    print(f"[INFO] Output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
