#!/usr/bin/env python3
"""
generate_masks.py — Generate object masks using GroundingDINO + SAM2 (keyword mode)
                    or SAM2 point prompting (point mode).

Masks are binary PNGs: white = keep, black = mask out (COLMAP convention).

Keyword mode:
  python generate_masks.py \
    --input_dir <frames_dir> --output_dir <masks_dir> \
    --keywords "person.tripod.camera" --precision 0.3

Point mode:
  python generate_masks.py \
    --input_dir <frames_dir> --output_dir <masks_dir> \
    --points_json '{"frame":"0001.jpg","points":[[x1,y1],[x2,y2]],"labels":[1,1]}'

Requires: pip install groundingdino-py segment-anything-2
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# SAM2 model management
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).resolve().parent.parent / "tools" / "models"
_SAM2_MODEL_ID = "facebook/sam2-hiera-large"

# Lazy-loaded singletons
_grounding_model = None
_sam_predictor = None


def _load_grounding_dino():
    """Load GroundingDINO model (lazy, cached)."""
    global _grounding_model
    if _grounding_model is not None:
        return _grounding_model

    try:
        from groundingdino.util.inference import load_model
        import torch

        model_paths = [
            Path(__file__).parent.parent / "models" / "groundingdino_swint_ogc.pth",
            Path.home() / ".cache" / "groundingdino" / "groundingdino_swint_ogc.pth",
        ]
        config_paths = [
            Path(__file__).parent.parent / "models" / "GroundingDINO_SwinT_OGC.py",
        ]

        model_path = next((p for p in model_paths if p.exists()), None)
        config_path = next((p for p in config_paths if p.exists()), None)

        if not model_path or not config_path:
            print("[INFO] Downloading GroundingDINO model...", flush=True)
            try:
                import torch.hub
                _grounding_model = torch.hub.load(
                    "IDEA-Research/GroundingDINO",
                    "groundingdino_swint_ogc",
                    pretrained=True,
                )
                return _grounding_model
            except Exception as e:
                print(f"[ERROR] Cannot load GroundingDINO: {e}", flush=True)
                return None

        _grounding_model = load_model(str(config_path), str(model_path))
        return _grounding_model

    except ImportError:
        print("[ERROR] groundingdino not installed. Run: pip install groundingdino-py", flush=True)
        return None


def _load_sam2():
    """Load SAM2 Hiera-Large predictor (lazy, cached). Auto-downloads from HuggingFace."""
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor

    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] Loading SAM2 Hiera-Large on {device}...", flush=True)
        _sam_predictor = SAM2ImagePredictor.from_pretrained(
            _SAM2_MODEL_ID,
            device=device,
        )
        print(f"[INFO] SAM2 loaded on {device}", flush=True)
        return _sam_predictor

    except ImportError:
        print(
            "[ERROR] sam2 not installed. Run: pip install SAM-2@git+https://github.com/facebookresearch/sam2.git",
            flush=True,
        )
        return None
    except Exception as e:
        print(f"[ERROR] SAM2 loading failed: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Keyword mode: GroundingDINO detection → SAM2 segmentation
# ---------------------------------------------------------------------------

def detect_objects(image_np, keywords, precision=0.3):
    """Detect objects using GroundingDINO text prompts.

    Returns (boxes_xyxy, scores, labels).
    """
    import torch
    from groundingdino.util.inference import predict

    model = _load_grounding_dino()
    if model is None:
        return np.zeros((0, 4)), np.zeros(0), []

    prompt = keywords.replace(".", " . ")

    from torchvision import transforms
    pil_img = Image.fromarray(image_np)
    transform = transforms.Compose([
        transforms.Resize(800, max_size=1333),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(pil_img)

    boxes, logits, phrases = predict(
        model=model, image=img_tensor, caption=prompt,
        box_threshold=precision, text_threshold=precision,
    )

    H, W = image_np.shape[:2]
    raw = boxes.cpu().numpy().copy()
    boxes_xyxy = np.zeros_like(raw)
    boxes_xyxy[:, 0] = (raw[:, 0] - raw[:, 2] / 2) * W
    boxes_xyxy[:, 1] = (raw[:, 1] - raw[:, 3] / 2) * H
    boxes_xyxy[:, 2] = (raw[:, 0] + raw[:, 2] / 2) * W
    boxes_xyxy[:, 3] = (raw[:, 1] + raw[:, 3] / 2) * H

    return boxes_xyxy, logits.cpu().numpy(), phrases


def segment_boxes(image_np, boxes):
    """Segment detected bounding boxes with SAM2. Returns combined bool mask."""
    predictor = _load_sam2()
    if predictor is None or len(boxes) == 0:
        return np.zeros(image_np.shape[:2], dtype=bool)

    import torch

    predictor.set_image(image_np)
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=predictor.device)
    masks, scores, _ = predictor.predict(box=boxes_tensor, multimask_output=False)

    combined = np.zeros(image_np.shape[:2], dtype=bool)
    for mask in masks:
        if mask.ndim == 3:
            mask = mask[0]
        combined |= mask.astype(bool)
    return combined


# ---------------------------------------------------------------------------
# Point mode: SAM2 point-prompted segmentation
# ---------------------------------------------------------------------------

def segment_points(image_np, points, labels):
    """Segment from user-clicked points using SAM2.

    Args:
        image_np: [H, W, 3] uint8
        points: [[x, y], ...] click coordinates
        labels: [1, 1, 0, ...] — 1=foreground, 0=background

    Returns:
        mask: [H, W] bool
    """
    predictor = _load_sam2()
    if predictor is None:
        return np.zeros(image_np.shape[:2], dtype=bool)

    import torch

    predictor.set_image(image_np)

    pts = np.array(points, dtype=np.float32)
    lbls = np.array(labels, dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=lbls,
        multimask_output=True,
    )
    # Pick the highest-scoring mask
    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    if mask.ndim == 3:
        mask = mask[0]
    return mask.astype(bool)


def _bbox_from_mask(mask, margin_pct=0.1):
    """Compute xyxy bounding box from a binary mask with margin."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    H, W = mask.shape
    mx = int((x2 - x1) * margin_pct)
    my = int((y2 - y1) * margin_pct)
    return [
        max(0, x1 - mx), max(0, y1 - my),
        min(W, x2 + mx), min(H, y2 + my),
    ]


def propagate_mask_to_frame(image_np, ref_bbox):
    """Use SAM2 box prompt (from reference mask bbox) on a new frame.

    Args:
        image_np: [H, W, 3] target frame
        ref_bbox: [x1, y1, x2, y2] bounding box from reference mask

    Returns:
        mask: [H, W] bool
    """
    predictor = _load_sam2()
    if predictor is None or ref_bbox is None:
        return np.zeros(image_np.shape[:2], dtype=bool)

    import torch

    predictor.set_image(image_np)
    box_tensor = torch.tensor([ref_bbox], dtype=torch.float32, device=predictor.device)
    masks, scores, _ = predictor.predict(box=box_tensor, multimask_output=False)

    mask = masks[0]
    if mask.ndim == 3:
        mask = mask[0]
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Post-processing & output
# ---------------------------------------------------------------------------

def apply_mask_ops(mask, expand=0, feather=0, invert=False):
    """Apply expand, feather, and invert operations to a binary mask."""
    if expand > 0:
        from scipy.ndimage import binary_dilation
        struct = np.ones((expand * 2 + 1, expand * 2 + 1))
        mask = binary_dilation(mask, structure=struct)

    mask_uint8 = (mask.astype(np.uint8) * 255)

    if feather > 0:
        mask_pil = Image.fromarray(mask_uint8)
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather))
        mask_uint8 = np.array(mask_pil)

    if invert:
        mask_uint8 = 255 - mask_uint8

    return mask_uint8


def generate_output(image_np, mask_uint8, mode="mask"):
    """Generate output based on mode."""
    if mode == "mask":
        # COLMAP: White=keep, Black=masked
        return Image.fromarray(255 - mask_uint8)
    elif mode == "transparent":
        rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = image_np
        rgba[:, :, 3] = 255 - mask_uint8
        return Image.fromarray(rgba)
    elif mode == "combined":
        output = image_np.copy()
        mask_bool = mask_uint8 > 127
        output[mask_bool, 0] = np.clip(output[mask_bool, 0].astype(int) + 100, 0, 255).astype(np.uint8)
        output[mask_bool, 1] = (output[mask_bool, 1] * 0.5).astype(np.uint8)
        output[mask_bool, 2] = (output[mask_bool, 2] * 0.5).astype(np.uint8)
        return Image.fromarray(output)
    return Image.fromarray(255 - mask_uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_keyword_mode(args, frames):
    """Run keyword-based GroundingDINO + SAM2 masking."""
    N = len(frames)
    print(f"[INFO] Keyword mode: {N} frames, keywords={args.keywords}", flush=True)
    print(
        f"[INFO] precision={args.precision}, expand={args.expand}, "
        f"feather={args.feather}, invert={args.invert}",
        flush=True,
    )

    t_start = time.time()
    n_detections_total = 0

    for i, frame_path in enumerate(frames):
        img = np.array(Image.open(frame_path).convert("RGB"))
        boxes, scores, labels = detect_objects(img, args.keywords, args.precision)
        n_detections = len(boxes)
        n_detections_total += n_detections

        if n_detections > 0:
            mask = segment_boxes(img, boxes)
        else:
            mask = np.zeros(img.shape[:2], dtype=bool)

        mask_uint8 = apply_mask_ops(mask, args.expand, args.feather, args.invert)
        output = generate_output(img, mask_uint8, args.mode)

        out_name = frame_path.name + ".png"
        output.save(args.output_dir / out_name)
        print(f"[MASK] {i + 1}/{N} {frame_path.name} — {n_detections} detections", flush=True)

    elapsed = time.time() - t_start
    print(
        f"[INFO] Masking complete: {N} masks generated, "
        f"{n_detections_total} total detections in {elapsed:.1f}s",
        flush=True,
    )


def run_point_mode(args, frames):
    """Run point-prompted SAM2 masking."""
    points_data = json.loads(args.points_json)
    ref_frame_name = points_data["frame"]
    points = points_data["points"]     # [[x,y], ...]
    labels = points_data["labels"]     # [1, 1, 0, ...]

    N = len(frames)
    print(f"[INFO] Point mode: {N} frames, {len(points)} points on {ref_frame_name}", flush=True)

    t_start = time.time()

    # Find reference frame
    ref_path = args.input_dir / ref_frame_name
    if not ref_path.exists():
        print(f"[ERROR] Reference frame not found: {ref_frame_name}", flush=True)
        sys.exit(1)

    # Segment reference frame with point prompts
    ref_img = np.array(Image.open(ref_path).convert("RGB"))
    ref_mask = segment_points(ref_img, points, labels)
    ref_bbox = _bbox_from_mask(ref_mask)

    if ref_bbox is None:
        print("[WARN] No mask produced from points — generating empty masks", flush=True)

    for i, frame_path in enumerate(frames):
        img = np.array(Image.open(frame_path).convert("RGB"))

        if frame_path.name == ref_frame_name:
            mask = ref_mask
        elif ref_bbox is not None:
            mask = propagate_mask_to_frame(img, ref_bbox)
        else:
            mask = np.zeros(img.shape[:2], dtype=bool)

        mask_uint8 = apply_mask_ops(mask, args.expand, args.feather, args.invert)
        output = generate_output(img, mask_uint8, args.mode)

        out_name = frame_path.name + ".png"
        output.save(args.output_dir / out_name)
        print(f"[MASK] {i + 1}/{N} {frame_path.name}", flush=True)

    elapsed = time.time() - t_start
    print(f"[INFO] Masking complete: {N} masks generated in {elapsed:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Generate object masks with SAM2")
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    # Keyword mode args
    ap.add_argument("--keywords", type=str, default=None,
                    help="Dot-separated keywords, e.g. 'person.tripod.camera'")
    ap.add_argument("--mode", default="mask", choices=["mask", "transparent", "combined"])
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--precision", type=float, default=0.3)
    ap.add_argument("--expand", type=int, default=0)
    ap.add_argument("--feather", type=int, default=0)
    # Point mode args
    ap.add_argument("--points_json", type=str, default=None,
                    help='JSON: {"frame":"0001.jpg","points":[[x,y],...],"labels":[1,...]}')
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(args.input_dir.glob("*.jpg")) + sorted(args.input_dir.glob("*.png"))
    # Exclude any mask PNGs that might be in the same dir
    frames = [f for f in frames if not f.name.endswith(".jpg.png") and not f.name.endswith(".png.png")]
    if not frames:
        print("[ERROR] No images found in input directory", flush=True)
        sys.exit(1)

    if args.points_json:
        run_point_mode(args, frames)
    elif args.keywords:
        run_keyword_mode(args, frames)
    else:
        print("[ERROR] Must provide either --keywords or --points_json", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
