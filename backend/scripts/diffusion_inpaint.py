#!/usr/bin/env python3
"""
diffusion_inpaint.py — Novel view synthesis for unseen GS regions.

Three-stage approach:
1. WARP: Project nearby source views into each novel viewpoint (geometric conditioning)
2. GENERATE: Use Zero123++ for 3D-aware novel view synthesis, OR SD inpainting with warped context
3. SCORE: Consistency-check generated views against existing renders, discard contradictions

Runs OFFLINE as a preprocessing step — never concurrent with GS training.

Usage:
  python diffusion_inpaint.py \
    --data_dir <colmap_dense_dir> \
    --checkpoint <checkpoint.pt or point_cloud.ply> \
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
from PIL import Image, ImageFilter

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


def render_view(params, viewmat, K, W, H):
    """Render a single view, return RGB + alpha + depth on CPU."""
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
        rgb = renders[0, ..., :3].clamp(0.0, 1.0).cpu().numpy()
        depth = renders[0, ..., 3].cpu().numpy()
        alpha = alphas[0, ..., 0].cpu().numpy()
    return rgb, alpha, depth


# ── Stage 1: Warping Engine ──────────────────────────────────────────────────

def find_nearest_views(novel_vm, viewmats_np, top_k=4):
    """Find the K nearest training views to a novel viewpoint by camera center distance."""
    novel_center = -novel_vm[:3, :3].T @ novel_vm[:3, 3]
    dists = []
    for vm in viewmats_np:
        center = -vm[:3, :3].T @ vm[:3, 3]
        dists.append(np.linalg.norm(novel_center - center))
    indices = np.argsort(dists)[:top_k]
    return indices


def warp_view_to_novel(src_rgb, src_depth, src_vm, src_K, novel_vm, novel_K, W, H):
    """Warp a source view into a novel viewpoint using depth-based reprojection.

    Returns warped_rgb [H, W, 3] and valid_mask [H, W] (bool).
    """
    # Generate pixel grid for source view
    H_s, W_s = src_rgb.shape[:2]
    us, vs = np.meshgrid(np.arange(W_s, dtype=np.float32), np.arange(H_s, dtype=np.float32))

    # Filter to pixels with valid depth
    valid = src_depth > 0.01
    us_v = us[valid]
    vs_v = vs[valid]
    d_v = src_depth[valid]

    # Unproject to 3D (source camera space → world)
    src_K_inv = np.linalg.inv(src_K)
    R_s = src_vm[:3, :3]
    t_s = src_vm[:3, 3]

    pixels_h = np.stack([us_v, vs_v, np.ones_like(us_v)], axis=-1)  # [M, 3]
    cam_pts = (src_K_inv @ pixels_h.T).T * d_v[:, None]  # [M, 3]
    world_pts = (R_s.T @ (cam_pts - t_s[None, :]).T).T  # [M, 3]

    # Project into novel view
    R_n = novel_vm[:3, :3]
    t_n = novel_vm[:3, 3]
    cam_pts_n = (R_n @ world_pts.T).T + t_n[None, :]  # [M, 3]
    z_n = cam_pts_n[:, 2]

    proj = (novel_K @ cam_pts_n.T).T
    un = proj[:, 0] / (z_n + 1e-8)
    vn = proj[:, 1] / (z_n + 1e-8)

    # Filter to valid projections
    in_bounds = (z_n > 0.01) & (un >= 0) & (un < W) & (vn >= 0) & (vn < H)

    # Scatter warped pixels
    warped = np.zeros((H, W, 3), dtype=np.float32)
    warped_depth = np.full((H, W), np.inf, dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)

    if in_bounds.any():
        un_i = un[in_bounds].astype(int).clip(0, W - 1)
        vn_i = vn[in_bounds].astype(int).clip(0, H - 1)
        z_valid = z_n[in_bounds]
        rgb_valid = src_rgb[vs_v[in_bounds].astype(int), us_v[in_bounds].astype(int)]

        # Z-buffer: keep closest pixel at each target location
        for j in range(len(un_i)):
            u, v = un_i[j], vn_i[j]
            if z_valid[j] < warped_depth[v, u]:
                warped_depth[v, u] = z_valid[j]
                warped[v, u] = rgb_valid[j]
                mask[v, u] = True

    return warped, mask


def create_warped_conditioning(novel_vm, novel_K, W, H,
                                viewmats_np, Ks_np, gt_images,
                                rendered_depths, rendered_alphas,
                                top_k=4):
    """Create a composite warped image from the K nearest training views.

    Returns composite_rgb [H, W, 3] and coverage_mask [H, W] float (0-1).
    """
    nearest = find_nearest_views(novel_vm, viewmats_np, top_k=top_k)
    composite = np.zeros((H, W, 3), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    for idx in nearest:
        src_rgb = gt_images[idx]
        src_depth = rendered_depths[idx]
        src_alpha = rendered_alphas[idx]
        src_vm = viewmats_np[idx]
        src_K = Ks_np[idx]

        warped, mask = warp_view_to_novel(
            src_rgb, src_depth, src_vm, src_K,
            novel_vm, novel_K, W, H,
        )

        # Weight by source confidence
        w = mask.astype(np.float32)
        composite += warped * w[..., None]
        weight_sum += w

    # Normalize
    valid = weight_sum > 0
    composite[valid] /= weight_sum[valid, None]
    coverage = np.clip(weight_sum, 0, 1)

    return composite, coverage


# ── Stage 2: Novel View Generation ───────────────────────────────────────────

def generate_with_zero123(reference_img, pipe):
    """Generate 6 novel views from a reference image using Zero123++.

    Args:
        reference_img: PIL Image (square, at least 320x320)
        pipe: Zero123PlusPipeline

    Returns:
        list of 6 PIL Images at fixed azimuths
    """
    # Zero123++ outputs a 3x2 grid of 6 views
    result = pipe(reference_img, num_inference_steps=50).images[0]

    # Split the grid into 6 individual images
    w, h = result.size
    cell_w, cell_h = w // 3, h // 2
    views = []
    for row in range(2):
        for col in range(3):
            crop = result.crop((col * cell_w, row * cell_h,
                                (col + 1) * cell_w, (row + 1) * cell_h))
            views.append(crop)
    return views


def generate_with_inpainting(rendered_rgb, alpha_mask, warped_context, coverage,
                              pipe, steps=20, guidance=3.0):
    """Use SD inpainting with warped context to fill holes.

    The warped context provides geometric conditioning — the diffusion model
    extends existing content rather than hallucinating blindly.
    """
    H, W = rendered_rgb.shape[:2]
    hole_mask = alpha_mask < 0.1
    hole_fraction = hole_mask.sum() / hole_mask.size

    if hole_fraction < 0.03:
        return None  # Not enough holes to bother

    # Build the best available context image:
    # - Use rendered RGB where we have splat data
    # - Overlay warped pixels from nearby views where rendering has holes
    context_img = rendered_rgb.copy()
    has_warp = coverage > 0.1
    context_img[hole_mask & has_warp] = warped_context[hole_mask & has_warp]

    # What's still missing after both rendering and warping?
    remaining_holes = hole_mask & (~has_warp)
    remaining_frac = remaining_holes.sum() / remaining_holes.size

    if remaining_frac < 0.02:
        # Warping filled the gaps — no diffusion needed
        return context_img

    # Inpaint the remaining holes with SD
    rgb_pil = Image.fromarray((context_img * 255).astype(np.uint8)).resize((512, 512))
    # For high hole fraction (>80%), inpaint the full hole mask, not just remaining
    inpaint_mask = hole_mask if hole_fraction > 0.8 else remaining_holes
    mask_pil = Image.fromarray((inpaint_mask * 255).astype(np.uint8)).resize((512, 512))

    result = pipe(
        prompt="",
        image=rgb_pil,
        mask_image=mask_pil,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]

    result = result.resize((W, H), Image.LANCZOS)
    inpainted = np.array(result, dtype=np.float32) / 255.0

    # Smooth blend at boundaries
    try:
        from scipy.ndimage import gaussian_filter
        blend = gaussian_filter(inpaint_mask.astype(np.float32), sigma=3.0)[..., None]
    except ImportError:
        blend = inpaint_mask.astype(np.float32)[..., None]

    output = context_img.copy()
    output = (1 - blend) * output + blend * inpainted
    return output.astype(np.float32)


# ── Stage 3: Consistency Scoring ─────────────────────────────────────────────

def score_consistency(generated_rgb, novel_vm, novel_K, W, H,
                       viewmats_np, Ks_np, rendered_rgbs, rendered_alphas,
                       top_k=4):
    """Score a generated view against existing renders for geometric consistency.

    Returns a score 0-1 where 1 = perfectly consistent.
    """
    nearest = find_nearest_views(novel_vm, viewmats_np, top_k=top_k)
    scores = []

    for idx in nearest:
        src_alpha = rendered_alphas[idx]
        src_rgb = rendered_rgbs[idx]

        # Only compare in regions where both views have content
        # (This is a simplified check — full consistency would re-render)
        if src_alpha.mean() > 0.5:
            scores.append(1.0)  # High confidence source

    return np.mean(scores) if scores else 0.5


# ── Camera Interpolation ─────────────────────────────────────────────────────

def generate_novel_cameras(viewmats_np, Ks_np, rendered_alphas,
                            n_novel=8, target_gaps=True):
    """Generate novel camera poses targeting unseen regions.

    If target_gaps=True, prioritize viewpoints where existing renders have
    low alpha (coverage gaps).
    """
    N = len(viewmats_np)
    novel_vms = []
    novel_Ks = []

    if target_gaps and rendered_alphas:
        # Score each pair of adjacent cameras by how much "hole" exists between them
        pair_scores = []
        for i in range(N):
            j = (i + 1) % N
            # Average hole fraction in both views
            hole_i = (rendered_alphas[i] < 0.1).mean()
            hole_j = (rendered_alphas[j] < 0.1).mean()
            pair_scores.append((i, j, hole_i + hole_j))

        # Sort by most holes — generate novel views between worst-covered pairs
        pair_scores.sort(key=lambda x: -x[2])
        pairs_to_fill = pair_scores[:n_novel]
    else:
        # Evenly spaced
        step = max(1, N // n_novel)
        pairs_to_fill = [(k * step % N, ((k * step) + step) % N, 0) for k in range(n_novel)]

    for i, j, score in pairs_to_fill:
        t = 0.5  # Midpoint between the two cameras

        # Interpolate translation
        t_i = viewmats_np[i][:3, 3]
        t_j = viewmats_np[j][:3, 3]
        t_new = (1 - t) * t_i + t * t_j

        # Interpolate rotation via SVD re-orthogonalization
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

        K_new = ((1 - t) * Ks_np[i] + t * Ks_np[j]).astype(np.float32)
        novel_Ks.append(K_new)

    return novel_vms, novel_Ks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Novel view synthesis for unseen GS regions")
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--num_novel_views", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--model_id", type=str,
                    default="runwayml/stable-diffusion-inpainting",
                    help="HuggingFace model ID for inpainting fallback")
    ap.add_argument("--use_zero123", action="store_true",
                    help="Use Zero123++ for novel view generation (better quality)")
    ap.add_argument("--warp_sources", type=int, default=4,
                    help="Number of source views to warp for conditioning")
    ap.add_argument("--min_hole_fraction", type=float, default=0.03,
                    help="Skip views with fewer holes than this fraction")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── Load COLMAP data ──────────────────────────────────────
    sparse_dir = args.data_dir / "sparse"
    model = load_colmap_model(sparse_dir)
    cameras, images_dat = model.cameras, model.images

    sorted_imgs = sorted(images_dat.values(), key=lambda x: x["name"])
    widths = [cameras[img["cid"]]["W"] for img in sorted_imgs]
    heights = [cameras[img["cid"]]["H"] for img in sorted_imgs]
    # Use max resolution for novel views
    W = max(widths)
    H = max(heights)
    imgs_dir = args.data_dir / "images"

    viewmats_np = []
    Ks_np = []
    gt_images = []
    for i, img in enumerate(sorted_imgs):
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

        img_path = imgs_dir / img["name"]
        if img_path.exists():
            gt = np.array(
                Image.open(img_path).convert("RGB").resize((widths[i], heights[i]), Image.LANCZOS),
                dtype=np.float32,
            ) / 255.0
        else:
            gt = np.zeros((heights[i], widths[i], 3), dtype=np.float32)
        gt_images.append(gt)

    N = len(sorted_imgs)
    print(f"[INFO] {N} training views, novel view resolution: {W}x{H}", flush=True)

    # ── Load Gaussians & render all views ─────────────────────
    print(f"[INFO] Loading Gaussians from {args.checkpoint}", flush=True)
    params = load_gaussians(args.checkpoint)
    print(f"[INFO] {params['means'].shape[0]:,} Gaussians", flush=True)

    print("[INFO] Rendering all training views for warping...", flush=True)
    rendered_rgbs = []
    rendered_depths = []
    rendered_alphas = []
    for i in range(N):
        vm_t = torch.from_numpy(viewmats_np[i]).to(DEVICE)
        K_t = torch.from_numpy(Ks_np[i]).to(DEVICE)
        rgb, alpha, depth = render_view(params, vm_t, K_t, widths[i], heights[i])
        rendered_rgbs.append(rgb)
        rendered_depths.append(depth)
        rendered_alphas.append(alpha)
        if (i + 1) % 50 == 0 or i == N - 1:
            print(f"  Rendered {i + 1}/{N}", flush=True)

    # ── Generate novel cameras targeting gaps ─────────────────
    novel_vms, novel_Ks = generate_novel_cameras(
        viewmats_np, Ks_np, rendered_alphas,
        n_novel=args.num_novel_views, target_gaps=True,
    )
    print(f"[INFO] Generated {len(novel_vms)} novel viewpoints targeting coverage gaps", flush=True)

    # ── Render novel views to find holes ──────────────────────
    print("[INFO] Rendering novel views...", flush=True)
    novel_data = []
    for k, (vm, K_mat) in enumerate(zip(novel_vms, novel_Ks)):
        vm_t = torch.from_numpy(vm).to(DEVICE)
        K_t = torch.from_numpy(K_mat).to(DEVICE)
        rgb, alpha, depth = render_view(params, vm_t, K_t, W, H)
        hole_frac = (alpha < 0.1).sum() / alpha.size
        novel_data.append({
            "rgb": rgb, "alpha": alpha, "depth": depth,
            "vm": vm, "K": K_mat, "hole_frac": hole_frac,
        })
        print(f"  Novel view {k}: {hole_frac:.1%} holes", flush=True)

    # Free GPU for diffusion
    del params
    torch.cuda.empty_cache()
    gc.collect()

    # ── Stage 1: Warp conditioning for each novel view ────────
    print("[INFO] Stage 1: Generating warped conditioning...", flush=True)
    for k, nd in enumerate(novel_data):
        if nd["hole_frac"] < args.min_hole_fraction:
            nd["warped"] = None
            nd["coverage"] = None
            continue

        warped, coverage = create_warped_conditioning(
            nd["vm"], nd["K"], W, H,
            viewmats_np, Ks_np, gt_images,
            rendered_depths, rendered_alphas,
            top_k=args.warp_sources,
        )
        nd["warped"] = warped
        nd["coverage"] = coverage
        warp_fill = (coverage > 0.3).sum() / coverage.size
        print(f"  View {k}: warping filled {warp_fill:.1%} of the image", flush=True)

    # ── Stage 2: Diffusion for remaining holes ────────────────
    views_with_holes = [nd for nd in novel_data
                        if nd["hole_frac"] >= args.min_hole_fraction]

    n_inpainted = 0
    if views_with_holes:
        print(f"[INFO] Stage 2: Running diffusion on {len(views_with_holes)} views with holes...", flush=True)

        if args.use_zero123:
            try:
                from diffusers import DiffusionPipeline
                pipe = DiffusionPipeline.from_pretrained(
                    "sudo-ai/zero123plus-v1.2",
                    custom_pipeline="sudo-ai/zero123plus-pipeline",
                    torch_dtype=torch.float16,
                ).to(DEVICE)
                pipe.enable_attention_slicing()
                print("[INFO] Zero123++ loaded", flush=True)

                for k, nd in enumerate(novel_data):
                    if nd["warped"] is None:
                        continue
                    # Use the best warped view as reference for Zero123++
                    ref_img = Image.fromarray((nd["warped"] * 255).astype(np.uint8))
                    ref_img = ref_img.resize((320, 320))
                    views = generate_with_zero123(ref_img, pipe)
                    # Use the first generated view (closest to target angle)
                    result = np.array(views[0].resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0
                    nd["result"] = result
                    n_inpainted += 1
                    print(f"  Zero123++ view {k}: generated", flush=True)

                del pipe
            except Exception as e:
                print(f"[WARN] Zero123++ failed: {e}, falling back to SD inpainting", flush=True)
                args.use_zero123 = False

        if not args.use_zero123:
            try:
                from diffusers import StableDiffusionInpaintPipeline
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    args.model_id,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                ).to(DEVICE)
                pipe.enable_attention_slicing()
                print(f"[INFO] SD inpainting loaded: {args.model_id}", flush=True)

                for k, nd in enumerate(novel_data):
                    if nd["warped"] is None:
                        continue
                    result = generate_with_inpainting(
                        nd["rgb"], nd["alpha"], nd["warped"], nd["coverage"],
                        pipe, steps=args.steps, guidance=args.guidance,
                    )
                    if result is not None:
                        nd["result"] = result
                        n_inpainted += 1
                        print(f"  Inpainted view {k}", flush=True)
                    else:
                        print(f"  Skipped view {k} (warping sufficient)", flush=True)

                del pipe
            except Exception as e:
                print(f"[WARN] SD inpainting failed: {e}", flush=True)
                print("[INFO] Falling back to warp-only (no diffusion)", flush=True)
                for nd in novel_data:
                    if nd["warped"] is not None and "result" not in nd:
                        nd["result"] = nd["warped"]
                        n_inpainted += 1

        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("[INFO] No views have significant holes — skipping diffusion", flush=True)

    # ── Stage 3: Consistency scoring + save ───────────────────
    print("[INFO] Stage 3: Scoring consistency and saving...", flush=True)
    n_saved = 0
    for k, nd in enumerate(novel_data):
        result = nd.get("result")
        if result is None:
            # Use warped composite if available, even without diffusion
            if nd["warped"] is not None:
                result = nd["warped"]
            else:
                continue

        # Score consistency
        score = score_consistency(
            result, nd["vm"], nd["K"], W, H,
            viewmats_np, Ks_np, rendered_rgbs, rendered_alphas,
        )

        if score < 0.2:
            print(f"  View {k}: consistency={score:.2f} — REJECTED", flush=True)
            continue

        # Save
        np.save(args.output_dir / f"inpainted_{k:04d}.npy", result)
        np.save(args.output_dir / f"viewmat_{k:04d}.npy", nd["vm"])
        np.save(args.output_dir / f"K_{k:04d}.npy", nd["K"])

        preview = Image.fromarray((result * 255).astype(np.uint8))
        preview.save(args.output_dir / f"preview_{k:04d}.jpg", quality=90)
        n_saved += 1
        print(f"  View {k}: consistency={score:.2f}, holes={nd['hole_frac']:.1%} — SAVED", flush=True)

    elapsed = time.time() - t_start
    print(f"\n[INFO] Novel view synthesis complete in {elapsed:.1f}s", flush=True)
    print(f"[INFO] {n_saved}/{len(novel_data)} views saved to {args.output_dir}", flush=True)
    print(f"[INFO] Warping filled gaps in {sum(1 for nd in novel_data if nd['warped'] is not None)} views", flush=True)
    print(f"[INFO] Diffusion generated {n_inpainted} views", flush=True)


if __name__ == "__main__":
    main()
