#!/usr/bin/env python3
"""
generate_novel_views.py — Multi-model novel view synthesis.

Supports: Zero123++, Wonder3D, Era3D, SD-Inpainting (fallback)
Each model generates novel views from reference images to fill coverage gaps.

Usage:
  python generate_novel_views.py \
    --input_dir <frames_dir> \
    --output_dir <novel_views_dir> \
    --model zero123pp \
    --num_refs 4 \
    --device cuda
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ── Model Loaders ─────────────────────────────────────────────────────────────

def load_zero123pp(device="cuda"):
    """Zero123++ v1.2: 6 views at fixed azimuths from a single reference."""
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,
    ).to(device)
    pipe.enable_attention_slicing()
    return pipe


def load_wonder3d(device="cuda"):
    """Wonder3D: 6 views + 6 normal maps at 256x256."""
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        "flamehaze1115/wonder3d-v1.0",
        custom_pipeline="flamehaze1115/wonder3d-pipeline",
        torch_dtype=torch.float16,
    ).to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()
    return pipe


def load_era3d(device="cuda"):
    """Era3D: 6 high-res views + normals at 512x512."""
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        "pengHTYX/MacLab-Era3D-512-6view",
        custom_pipeline="pengHTYX/Era3D-512-6view-pipeline",
        torch_dtype=torch.float16,
    ).to(device)
    pipe.enable_attention_slicing()
    return pipe


def load_sd_inpainting(device="cuda"):
    """SD Inpainting fallback: generic inpainting, no 3D awareness."""
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.enable_attention_slicing()
    return pipe


MODEL_LOADERS = {
    "zero123pp": ("Zero123++ v1.2", load_zero123pp),
    "wonder3d": ("Wonder3D v1.0", load_wonder3d),
    "era3d": ("Era3D 512x512", load_era3d),
    "sd_inpaint": ("SD Inpainting (fallback)", load_sd_inpainting),
}


# ── View Generators ───────────────────────────────────────────────────────────

def generate_zero123pp(pipe, ref_img, output_size=800):
    """Generate 6 novel views using Zero123++."""
    ref_small = ref_img.resize((320, 320))
    grid = pipe(ref_small, num_inference_steps=50).images[0]

    gw, gh = grid.size
    cw, ch = gw // 3, gh // 2
    views = []
    for row in range(2):
        for col in range(3):
            crop = grid.crop((col * cw, row * ch, (col + 1) * cw, (row + 1) * ch))
            crop = crop.resize((output_size, output_size), Image.LANCZOS)
            views.append(crop)
    return views, grid


def generate_wonder3d(pipe, ref_img, output_size=800):
    """Generate 6 color views + 6 normal maps using Wonder3D."""
    ref_small = ref_img.resize((256, 256))
    try:
        result = pipe(ref_small, num_inference_steps=50)
        # Wonder3D outputs color + normal images
        color_views = []
        normal_views = []
        if hasattr(result, 'images'):
            imgs = result.images
            # First 6 are color, next 6 are normals (if available)
            for i, img in enumerate(imgs):
                resized = img.resize((output_size, output_size), Image.LANCZOS)
                if i < 6:
                    color_views.append(resized)
                else:
                    normal_views.append(resized)
        return color_views if color_views else [ref_img], None
    except Exception as e:
        print(f"[WARN] Wonder3D generation failed: {e}", flush=True)
        return [ref_img], None


def generate_era3d(pipe, ref_img, output_size=800):
    """Generate 6 high-res views using Era3D."""
    ref_small = ref_img.resize((512, 512))
    try:
        result = pipe(ref_small, num_inference_steps=50)
        views = []
        if hasattr(result, 'images'):
            for img in result.images[:6]:
                views.append(img.resize((output_size, output_size), Image.LANCZOS))
        return views if views else [ref_img], None
    except Exception as e:
        print(f"[WARN] Era3D generation failed: {e}", flush=True)
        return [ref_img], None


def generate_sd_inpaint(pipe, ref_img, output_size=800):
    """Generate a single novel view using SD inpainting (full hallucination)."""
    ref_resized = ref_img.resize((512, 512))
    mask = Image.fromarray(np.full((512, 512), 255, dtype=np.uint8))

    prompts = [
        "same object from behind, 3D render, studio lighting",
        "same object from above, top view, 3D render",
        "same object from the left side, 3D render",
        "same object from below, bottom view, 3D render",
    ]
    views = []
    for prompt in prompts:
        result = pipe(
            prompt=prompt,
            image=ref_resized,
            mask_image=mask,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]
        views.append(result.resize((output_size, output_size), Image.LANCZOS))
    return views, None


GENERATORS = {
    "zero123pp": generate_zero123pp,
    "wonder3d": generate_wonder3d,
    "era3d": generate_era3d,
    "sd_inpaint": generate_sd_inpaint,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate novel views using multiple models")
    ap.add_argument("--input_dir", required=True, type=Path,
                    help="Directory with reference frames (JPG/PNG)")
    ap.add_argument("--output_dir", required=True, type=Path,
                    help="Output directory for novel views")
    ap.add_argument("--model", default="zero123pp",
                    choices=list(MODEL_LOADERS.keys()),
                    help="Which model to use")
    ap.add_argument("--num_refs", type=int, default=4,
                    help="Number of reference frames to generate from")
    ap.add_argument("--output_size", type=int, default=800,
                    help="Output image size (square)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Find reference frames
    frames = sorted(args.input_dir.glob("*.jpg")) + sorted(args.input_dir.glob("*.png"))
    if not frames:
        print("[ERROR] No images found", flush=True)
        sys.exit(1)

    # Pick evenly-spaced references
    step = max(1, len(frames) // args.num_refs)
    ref_indices = [i * step for i in range(args.num_refs)]
    ref_indices = [min(idx, len(frames) - 1) for idx in ref_indices]

    model_name, loader_fn = MODEL_LOADERS[args.model]
    generator_fn = GENERATORS[args.model]

    print(f"[INFO] Model: {model_name}", flush=True)
    print(f"[INFO] Frames: {len(frames)}, using {len(ref_indices)} references", flush=True)
    print(f"[INFO] Output size: {args.output_size}x{args.output_size}", flush=True)

    # Load model
    print(f"[INFO] Loading {model_name}...", flush=True)
    try:
        pipe = loader_fn(args.device)
        print(f"[INFO] Model loaded on {args.device}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load {model_name}: {e}", flush=True)
        print(f"[INFO] Falling back to SD Inpainting", flush=True)
        args.model = "sd_inpaint"
        model_name = "SD Inpainting (fallback)"
        pipe = load_sd_inpainting(args.device)
        generator_fn = generate_sd_inpaint

    # Generate views
    total_views = 0
    for ref_idx in ref_indices:
        ref_path = frames[ref_idx]
        ref_img = Image.open(ref_path).convert("RGB")

        print(f"[NOVEL] Generating from {ref_path.name}...", flush=True)
        try:
            views, grid = generator_fn(pipe, ref_img, args.output_size)
        except Exception as e:
            print(f"[WARN] Failed for {ref_path.name}: {e}", flush=True)
            continue

        # Save grid if available
        if grid is not None:
            grid.save(args.output_dir / f"grid_{ref_path.stem}.jpg", quality=95)

        # Save individual views
        for v, view in enumerate(views):
            name = f"novel_{ref_path.stem}_v{v}.jpg"
            view.save(args.output_dir / name, quality=95)
            total_views += 1

        print(f"  Saved {len(views)} views", flush=True)

    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - t_start
    print(f"\n[INFO] Complete: {total_views} novel views in {elapsed:.1f}s", flush=True)
    print(f"[INFO] Model: {model_name}", flush=True)
    print(f"[INFO] Output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
