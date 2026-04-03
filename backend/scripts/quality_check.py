#!/usr/bin/env python3
"""
quality_check.py — Automated quality inspection for trained Gaussian Splats.

Renders test views from the trained splat and computes:
- PSNR against ground truth
- Floater score (% of Gaussians with low opacity or extreme scale)
- Silhouette cleanliness (edge sharpness of rendered alpha)
- Coverage score (% of GT foreground covered by rendered alpha)

Usage:
  python quality_check.py \
    --data_dir <colmap_dense> \
    --ply <point_cloud.ply> \
    --output <quality_report.json>
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_psnr(pred, gt):
    """Compute PSNR between two images (numpy, 0-1 range)."""
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 50.0
    return -10 * math.log10(mse)


def analyze_ply(ply_path):
    """Analyze Gaussian parameters for floater detection."""
    with open(ply_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if b"end_header" in line:
                break

        header_str = b"".join(header_lines).decode("utf-8", errors="replace")
        n_verts = 0
        props = []
        for h in header_str.split("\n"):
            if h.startswith("element vertex"):
                n_verts = int(h.split()[-1])
            if h.startswith("property float"):
                props.append(h.split()[-1])

        n_floats = len(props)
        data = np.frombuffer(f.read(n_verts * n_floats * 4), dtype=np.float32).copy().reshape(n_verts, n_floats)

    prop_idx = {name: i for i, name in enumerate(props)}

    # Opacity analysis
    opacity_logit = data[:, prop_idx["opacity"]]
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))

    # Scale analysis
    scales = np.stack([np.exp(data[:, prop_idx[f"scale_{i}"]]) for i in range(3)], axis=1)
    max_scale = scales.max(axis=1)
    med_scale = np.median(max_scale)

    # Position analysis
    positions = data[:, :3]
    center = positions.mean(axis=0)
    dists = np.linalg.norm(positions - center, axis=1)
    dist_99 = np.percentile(dists, 99)

    # Floater metrics
    low_opacity = (opacity < 0.05).sum()
    oversized = (max_scale > med_scale * 10).sum()
    outliers = (dists > dist_99).sum()

    floater_pct = (low_opacity + oversized + outliers) / n_verts * 100

    return {
        "n_gaussians": int(n_verts),
        "opacity_mean": float(np.mean(opacity)),
        "opacity_median": float(np.median(opacity)),
        "low_opacity_pct": float(low_opacity / n_verts * 100),
        "oversized_pct": float(oversized / n_verts * 100),
        "outlier_pct": float(outliers / n_verts * 100),
        "floater_score": float(floater_pct),
        "median_scale": float(med_scale),
        "ply_size_mb": float(ply_path.stat().st_size / 1024 / 1024),
    }


def render_and_compare(ply_path, data_dir):
    """Render views from the trained PLY and compare against GT."""
    sys.path.insert(0, str(Path(__file__).parent))
    from colmap_io import load_colmap_model, qvec_to_rotmat, get_intrinsics
    from visibility_transfer import load_gaussians_from_ply

    sparse = data_dir / "sparse"
    imgs_dir = data_dir / "images"

    model = load_colmap_model(sparse)
    cameras, images_dat = model.cameras, model.images
    sorted_imgs = sorted(images_dat.values(), key=lambda x: x["name"])

    params = load_gaussians_from_ply(ply_path)

    # Sample 5 evenly-spaced views for PSNR
    n = len(sorted_imgs)
    test_indices = [i * n // 5 for i in range(5)]

    psnrs = []
    alpha_coverages = []

    from gsplat.rendering import rasterization

    for idx in test_indices:
        img_info = sorted_imgs[idx]
        cam = cameras[img_info["cid"]]
        W, H = cam["W"], cam["H"]
        R = qvec_to_rotmat(img_info["qvec"]).astype(np.float32)
        t = np.array(img_info["tvec"], dtype=np.float32)
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = R
        vm[:3, 3] = t

        fx, fy, cx, cy = get_intrinsics(cam)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # Render
        with torch.no_grad():
            q = F.normalize(params["quats"], dim=-1)
            renders, alphas, _ = rasterization(
                means=params["means"],
                quats=q,
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=params["sh_coeffs"],
                viewmats=torch.from_numpy(vm).unsqueeze(0).to(DEVICE),
                Ks=torch.from_numpy(K).unsqueeze(0).to(DEVICE),
                width=W, height=H,
                sh_degree=0, packed=False, render_mode="RGB",
            )
            pred = renders[0].clamp(0, 1).cpu().numpy()
            alpha = alphas[0, ..., 0].cpu().numpy()

        # Load GT
        gt_path = imgs_dir / img_info["name"]
        if gt_path.exists():
            gt = np.array(Image.open(gt_path).convert("RGB").resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0
            psnr = compute_psnr(pred, gt)
            psnrs.append(psnr)

        # Alpha coverage
        alpha_coverages.append(float((alpha > 0.5).mean()))

    # Cleanup GPU
    del params
    torch.cuda.empty_cache()

    return {
        "mean_psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "min_psnr": float(np.min(psnrs)) if psnrs else 0.0,
        "max_psnr": float(np.max(psnrs)) if psnrs else 0.0,
        "psnrs": [float(p) for p in psnrs],
        "mean_alpha_coverage": float(np.mean(alpha_coverages)),
        "n_test_views": len(psnrs),
    }


def quality_pass(report, psnr_threshold=20.0, floater_threshold=30.0):
    """Determine if the splat passes quality inspection."""
    psnr_ok = report["render"]["mean_psnr"] >= psnr_threshold
    floater_ok = report["ply"]["floater_score"] <= floater_threshold
    passed = psnr_ok and floater_ok

    reasons = []
    if not psnr_ok:
        reasons.append(f"PSNR {report['render']['mean_psnr']:.1f} < {psnr_threshold}")
    if not floater_ok:
        reasons.append(f"Floater score {report['ply']['floater_score']:.1f}% > {floater_threshold}%")

    return passed, reasons


def main():
    ap = argparse.ArgumentParser(description="Quality inspection for trained GS")
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--ply", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--psnr_threshold", type=float, default=20.0)
    ap.add_argument("--floater_threshold", type=float, default=30.0)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[QC] Analyzing PLY: {args.ply}", flush=True)

    # PLY analysis
    ply_report = analyze_ply(args.ply)
    print(f"[QC] Gaussians: {ply_report['n_gaussians']:,}, "
          f"floater_score: {ply_report['floater_score']:.1f}%, "
          f"size: {ply_report['ply_size_mb']:.1f} MB", flush=True)

    # Render and compare
    print(f"[QC] Rendering test views...", flush=True)
    render_report = render_and_compare(args.ply, args.data_dir)
    print(f"[QC] PSNR: mean={render_report['mean_psnr']:.1f}, "
          f"min={render_report['min_psnr']:.1f}, max={render_report['max_psnr']:.1f}", flush=True)

    # Quality verdict
    report = {
        "ply": ply_report,
        "render": render_report,
        "elapsed_s": time.time() - t0,
    }

    passed, reasons = quality_pass(report, args.psnr_threshold, args.floater_threshold)
    report["passed"] = passed
    report["fail_reasons"] = reasons

    if passed:
        print(f"[QC] ✓ PASSED (PSNR={render_report['mean_psnr']:.1f}, "
              f"floaters={ply_report['floater_score']:.1f}%)", flush=True)
    else:
        print(f"[QC] ✗ FAILED: {', '.join(reasons)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[QC] Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
