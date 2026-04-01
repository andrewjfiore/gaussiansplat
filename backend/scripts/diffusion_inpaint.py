#!/usr/bin/env python3
"""
diffusion_inpaint.py — Stable Diffusion inpainting for unseen GS regions.

After visibility transfer + refinement training, renders from novel viewpoints
where holes remain (alpha < threshold), runs SD inpainting to fill them, and
saves the results as pseudo-GT for a final training pass.

Runs OFFLINE as a preprocessing step — never concurrent with GS training.
Requires: pip install diffusers transformers accelerate

Usage:
  python diffusion_inpaint.py \
    --data_dir <colmap_dense_dir> \
    --checkpoint <path/to/checkpoint.pt or point_cloud.ply> \
    --output_dir <inpainted_output_dir> \
    --num_novel_views 8
"""

import argparse
import gc
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


def load_gaussians(ckpt_path: Path) -> dict:
    """Load Gaussians from checkpoint or PLY."""
    if ckpt_path.suffix == ".ply":
        from visibility_transfer import load_gaussians_from_ply
        return load_gaussians_from_ply(ckpt_path)
    else:
        ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
        return {k: v.to(DEVICE) for k, v in ckpt["params"].items()}


def interpolate_cameras(viewmats_np, Ks_np, n_novel):
    """Generate novel camera poses by interpolating between existing cameras.

    Uses spherical linear interpolation for rotation and linear for translation.
    Focuses on viewpoints between existing cameras to fill coverage gaps.
    """
    N = len(viewmats_np)
    novel_vms = []
    novel_Ks = []

    # Generate evenly spaced interpolated views between consecutive cameras
    step_size = max(1, N // n_novel)
    for k in range(n_novel):
        i = (k * step_size) % N
        j = (i + step_size) % N

        # Random interpolation factor biased toward the middle
        t = 0.3 + 0.4 * np.random.random()

        # Interpolate translation
        t_i = viewmats_np[i][:3, 3]
        t_j = viewmats_np[j][:3, 3]
        t_new = (1 - t) * t_i + t * t_j

        # Interpolate rotation via averaging + re-orthogonalizing
        R_i = viewmats_np[i][:3, :3]
        R_j = viewmats_np[j][:3, :3]
        R_avg = (1 - t) * R_i + t * R_j
        U, _, Vt = np.linalg.svd(R_avg)
        R_new = U @ Vt
        if np.linalg.det(R_new) < 0:
            U[:, -1] *= -1
            R_new = U @ Vt

        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = R_new.astype(np.float32)
        vm[:3, 3] = t_new.astype(np.float32)
        novel_vms.append(vm)

        # Use average intrinsics
        K_new = ((1 - t) * Ks_np[i] + t * Ks_np[j]).astype(np.float32)
        novel_Ks.append(K_new)

    return novel_vms, novel_Ks


def render_view(params, viewmat, K, W, H):
    """Render a single view, return RGB + alpha on CPU."""
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
            render_mode="RGB",
        )
        rgb = renders[0].clamp(0.0, 1.0).cpu().numpy()
        alpha = alphas[0, ..., 0].cpu().numpy()
    return rgb, alpha


def run_inpainting(rendered_rgb, alpha_mask, pipe, alpha_threshold=0.1,
                   min_hole_fraction=0.05, steps=20, guidance=3.0):
    """Run SD inpainting on holes in a rendered image.

    Args:
        rendered_rgb: [H, W, 3] float32 0-1
        alpha_mask: [H, W] float32 0-1
        pipe: StableDiffusionInpaintPipeline
        alpha_threshold: below this, pixel is considered a hole
        min_hole_fraction: skip if holes are less than this fraction of image

    Returns:
        inpainted: [H, W, 3] float32 0-1, or None if skipped
    """
    hole_mask = alpha_mask < alpha_threshold
    hole_fraction = hole_mask.sum() / hole_mask.size

    if hole_fraction < min_hole_fraction:
        return None  # Not enough holes to warrant inpainting

    if hole_fraction > 0.8:
        return None  # Too much missing — inpainting won't help

    # Prepare PIL images for the pipeline
    H, W = rendered_rgb.shape[:2]
    # SD inpainting expects 512x512 — resize, inpaint, resize back
    rgb_pil = Image.fromarray((rendered_rgb * 255).astype(np.uint8)).resize((512, 512))
    mask_pil = Image.fromarray((hole_mask * 255).astype(np.uint8)).resize((512, 512))

    result = pipe(
        prompt="",
        image=rgb_pil,
        mask_image=mask_pil,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]

    # Resize back and convert to float
    result = result.resize((W, H), Image.LANCZOS)
    inpainted = np.array(result, dtype=np.float32) / 255.0

    # Blend: keep original pixels where alpha is high, use inpainted where holes are
    blend_mask = (alpha_mask < alpha_threshold).astype(np.float32)
    # Smooth the blend boundary
    from scipy.ndimage import gaussian_filter
    blend_mask = gaussian_filter(blend_mask, sigma=3.0)
    blend_mask = blend_mask[..., None]  # [H, W, 1]

    output = (1 - blend_mask) * rendered_rgb + blend_mask * inpainted
    return output.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Diffusion inpainting for unseen GS regions")
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--num_novel_views", type=int, default=8,
                    help="Number of novel viewpoints to generate and inpaint")
    ap.add_argument("--steps", type=int, default=20,
                    help="SD inference steps (fewer = faster, more = better quality)")
    ap.add_argument("--guidance", type=float, default=3.0,
                    help="SD guidance scale (lower = more faithful to existing content)")
    ap.add_argument("--model_id", type=str,
                    default="stabilityai/stable-diffusion-2-inpainting",
                    help="HuggingFace model ID for inpainting pipeline")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load COLMAP data ──────────────────────────────────────
    sparse_dir = args.data_dir / "sparse"
    model = load_colmap_model(sparse_dir)
    cameras, images_dat = model.cameras, model.images

    sorted_imgs = sorted(images_dat.values(), key=lambda x: x["name"])

    # Use the max resolution among all cameras for novel view rendering
    all_W = [cameras[img["cid"]]["W"] for img in sorted_imgs]
    all_H = [cameras[img["cid"]]["H"] for img in sorted_imgs]
    W = max(all_W)
    H = max(all_H)

    viewmats_np = []
    Ks_np = []
    for img in sorted_imgs:
        cam = cameras[img["cid"]]
        R = qvec_to_rotmat(img["qvec"]).astype(np.float32)
        t = np.array(img["tvec"], dtype=np.float32)
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = R
        vm[:3, 3] = t
        viewmats_np.append(vm)

        fx, fy, cx, cy = get_intrinsics(cam)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        Ks_np.append(K)

    print(f"[INFO] {len(sorted_imgs)} training views, novel view resolution: {W}x{H}", flush=True)

    # ── Load Gaussians ────────────────────────────────────────
    print(f"[INFO] Loading Gaussians from {args.checkpoint}", flush=True)
    params = load_gaussians(args.checkpoint)
    print(f"[INFO] {params['means'].shape[0]:,} Gaussians", flush=True)

    # ── Generate novel viewpoints ─────────────────────────────
    novel_vms, novel_Ks = interpolate_cameras(viewmats_np, Ks_np, args.num_novel_views)
    print(f"[INFO] Generated {len(novel_vms)} novel viewpoints", flush=True)

    # ── Render novel views ────────────────────────────────────
    print("[INFO] Rendering novel views...", flush=True)
    novel_renders = []
    for k, (vm, K) in enumerate(zip(novel_vms, novel_Ks)):
        vm_t = torch.from_numpy(vm).to(DEVICE)
        K_t = torch.from_numpy(K).to(DEVICE)
        rgb, alpha = render_view(params, vm_t, K_t, W, H)
        hole_frac = (alpha < 0.1).sum() / alpha.size
        novel_renders.append((rgb, alpha, vm, K))
        print(f"  Novel view {k}: {hole_frac:.1%} holes", flush=True)

    # Free GPU for diffusion model
    del params
    torch.cuda.empty_cache()
    gc.collect()

    # ── Load SD inpainting pipeline ───────────────────────────
    print(f"[INFO] Loading inpainting model: {args.model_id}", flush=True)
    try:
        from diffusers import StableDiffusionInpaintPipeline

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
        ).to(DEVICE)
        pipe.enable_attention_slicing()
        print("[INFO] Inpainting model loaded", flush=True)
    except ImportError:
        print("[ERROR] diffusers not installed. Run: pip install diffusers transformers accelerate", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load inpainting model: {e}", flush=True)
        # Try CPU offload as fallback
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipe.enable_sequential_cpu_offload()
            print("[INFO] Using CPU offload mode (slower but fits in VRAM)", flush=True)
        except Exception as e2:
            print(f"[ERROR] Fallback also failed: {e2}", flush=True)
            sys.exit(1)

    # ── Run inpainting ────────────────────────────────────────
    print("[INFO] Running inpainting...", flush=True)
    n_inpainted = 0

    for k, (rgb, alpha, vm, K_mat) in enumerate(novel_renders):
        result = run_inpainting(rgb, alpha, pipe,
                                steps=args.steps, guidance=args.guidance)
        if result is not None:
            np.save(args.output_dir / f"inpainted_{k:04d}.npy", result)
            np.save(args.output_dir / f"viewmat_{k:04d}.npy", vm)
            np.save(args.output_dir / f"K_{k:04d}.npy", K_mat)

            # Save preview
            preview = Image.fromarray((result * 255).astype(np.uint8))
            preview.save(args.output_dir / f"preview_{k:04d}.jpg", quality=90)
            n_inpainted += 1
            print(f"  Inpainted novel view {k}", flush=True)
        else:
            print(f"  Skipped novel view {k} (insufficient holes)", flush=True)

    # ── Cleanup ───────────────────────────────────────────────
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[INFO] Inpainting complete: {n_inpainted}/{len(novel_renders)} views inpainted", flush=True)
    print(f"[INFO] Output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
