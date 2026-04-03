#!/usr/bin/env python3
"""
generate_masks.py — Generate object masks using GroundingDINO + SAM2.

Detects objects matching keyword prompts in each frame, then generates
precise segmentation masks using SAM2. Masks are saved as binary PNGs
(white = keep, black = mask out by default).

Requires: pip install groundingdino-py segment-anything-2

Usage:
  python generate_masks.py \
    --input_dir <frames_dir> \
    --output_dir <masks_dir> \
    --keywords "person.tripod.camera" \
    --mode mask \
    --precision 0.3
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# Lazy imports for heavy ML dependencies
_grounding_model = None
_sam_predictor = None


def _load_grounding_dino():
    """Load GroundingDINO model (lazy, cached)."""
    global _grounding_model
    if _grounding_model is not None:
        return _grounding_model

    try:
        from groundingdino.util.inference import load_model, predict
        import torch

        # Try to find model weights
        model_paths = [
            Path(__file__).parent.parent / "models" / "groundingdino_swint_ogc.pth",
            Path.home() / ".cache" / "groundingdino" / "groundingdino_swint_ogc.pth",
        ]
        config_paths = [
            Path(__file__).parent.parent / "models" / "GroundingDINO_SwinT_OGC.py",
        ]

        model_path = None
        config_path = None
        for p in model_paths:
            if p.exists():
                model_path = p
                break
        for p in config_paths:
            if p.exists():
                config_path = p
                break

        if not model_path or not config_path:
            # Try torch.hub download
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
                print("[INFO] Place model files in backend/models/ or install via pip", flush=True)
                return None

        _grounding_model = load_model(str(config_path), str(model_path))
        return _grounding_model

    except ImportError:
        print("[ERROR] groundingdino not installed. Run: pip install groundingdino-py", flush=True)
        return None


def _load_sam2():
    """Load SAM2 predictor (lazy, cached)."""
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor

    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use from_pretrained (auto-downloads correct checkpoint from HuggingFace)
        _sam_predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-small",
            device=device,
        )
        print(f"[INFO] SAM2 loaded on {device}", flush=True)
        return _sam_predictor

    except ImportError:
        print("[ERROR] sam2 not installed. Run: pip install SAM-2@git+https://github.com/facebookresearch/sam2.git", flush=True)
        return None
    except Exception as e:
        print(f"[ERROR] SAM2 loading failed: {e}", flush=True)
        return None


def detect_objects(image_np, keywords, precision=0.3):
    """Detect objects in image using GroundingDINO.

    Args:
        image_np: [H, W, 3] uint8 numpy array
        keywords: dot-separated keywords string
        precision: detection confidence threshold

    Returns:
        boxes: [N, 4] xyxy bounding boxes
        scores: [N] confidence scores
        labels: [N] detected class labels
    """
    import torch
    from groundingdino.util.inference import predict
    from groundingdino.util.utils import get_phrases_from_posmap

    model = _load_grounding_dino()
    if model is None:
        return np.zeros((0, 4)), np.zeros(0), []

    # Convert keywords: "person.tripod" → "person . tripod"
    prompt = keywords.replace(".", " . ")

    # GroundingDINO expects PIL image
    from torchvision import transforms
    pil_img = Image.fromarray(image_np)

    transform = transforms.Compose([
        transforms.Resize(800, max_size=1333),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(pil_img)

    boxes, logits, phrases = predict(
        model=model,
        image=img_tensor,
        caption=prompt,
        box_threshold=precision,
        text_threshold=precision,
    )

    H, W = image_np.shape[:2]
    # GroundingDINO returns cxcywh normalized [0,1] — convert to xyxy pixel coords
    raw = boxes.cpu().numpy().copy()
    boxes_xyxy = np.zeros_like(raw)
    boxes_xyxy[:, 0] = (raw[:, 0] - raw[:, 2] / 2) * W  # x1
    boxes_xyxy[:, 1] = (raw[:, 1] - raw[:, 3] / 2) * H  # y1
    boxes_xyxy[:, 2] = (raw[:, 0] + raw[:, 2] / 2) * W  # x2
    boxes_xyxy[:, 3] = (raw[:, 1] + raw[:, 3] / 2) * H  # y2

    return boxes_xyxy, logits.cpu().numpy(), phrases


def segment_boxes(image_np, boxes):
    """Generate precise masks for detected bounding boxes using SAM2.

    Args:
        image_np: [H, W, 3] uint8 numpy array
        boxes: [N, 4] xyxy bounding boxes

    Returns:
        combined_mask: [H, W] bool — True where any object is detected
    """
    predictor = _load_sam2()
    if predictor is None or len(boxes) == 0:
        return np.zeros(image_np.shape[:2], dtype=bool)

    import torch

    predictor.set_image(image_np)

    # SAM2 expects boxes as [N, 4] tensor in xyxy format
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=predictor.device)
    masks, scores, _ = predictor.predict(
        box=boxes_tensor,
        multimask_output=False,
    )

    # Combine all masks into one
    combined = np.zeros(image_np.shape[:2], dtype=bool)
    for mask in masks:
        if mask.ndim == 3:
            mask = mask[0]  # Take first mask if multimask
        combined |= mask.astype(bool)

    return combined


def apply_mask_ops(mask, expand=0, feather=0, invert=False):
    """Apply expand, feather, and invert operations to a binary mask."""
    # Expand mask
    if expand > 0:
        from scipy.ndimage import binary_dilation
        struct = np.ones((expand * 2 + 1, expand * 2 + 1))
        mask = binary_dilation(mask, structure=struct)

    # Convert to uint8 for PIL operations
    mask_uint8 = (mask.astype(np.uint8) * 255)

    # Feather edges
    if feather > 0:
        mask_pil = Image.fromarray(mask_uint8)
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather))
        mask_uint8 = np.array(mask_pil)

    # Invert
    if invert:
        mask_uint8 = 255 - mask_uint8

    return mask_uint8


def generate_output(image_np, mask_uint8, mode="mask"):
    """Generate output based on mode.

    Args:
        image_np: [H, W, 3] uint8 original image
        mask_uint8: [H, W] uint8 mask (255 = masked region)
        mode: "mask" | "transparent" | "combined"

    Returns:
        output: PIL Image
    """
    if mode == "mask":
        # COLMAP convention: White (255) = valid/keep, Black (0) = masked/ignored
        return Image.fromarray(255 - mask_uint8)

    elif mode == "transparent":
        # Original image with masked areas transparent
        rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = image_np
        rgba[:, :, 3] = 255 - mask_uint8  # alpha: 0 where masked
        return Image.fromarray(rgba)

    elif mode == "combined":
        # Original image with red overlay on masked areas
        output = image_np.copy()
        mask_bool = mask_uint8 > 127
        output[mask_bool, 0] = np.clip(output[mask_bool, 0].astype(int) + 100, 0, 255).astype(np.uint8)
        output[mask_bool, 1] = (output[mask_bool, 1] * 0.5).astype(np.uint8)
        output[mask_bool, 2] = (output[mask_bool, 2] * 0.5).astype(np.uint8)
        return Image.fromarray(output)

    return Image.fromarray(255 - mask_uint8)


def main():
    ap = argparse.ArgumentParser(description="Generate object masks with GroundingDINO + SAM2")
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--keywords", required=True, type=str,
                    help="Dot-separated keywords, e.g. 'person.tripod.camera'")
    ap.add_argument("--mode", default="mask", choices=["mask", "transparent", "combined"])
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--precision", type=float, default=0.3)
    ap.add_argument("--expand", type=int, default=0)
    ap.add_argument("--feather", type=int, default=0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(args.input_dir.glob("*.jpg")) + sorted(args.input_dir.glob("*.png"))
    if not frames:
        print("[ERROR] No images found in input directory", flush=True)
        sys.exit(1)

    N = len(frames)
    print(f"[INFO] Processing {N} frames with keywords: {args.keywords}", flush=True)
    print(f"[INFO] Mode: {args.mode}, precision: {args.precision}, "
          f"expand: {args.expand}, feather: {args.feather}, invert: {args.invert}", flush=True)

    t_start = time.time()
    n_detections_total = 0

    for i, frame_path in enumerate(frames):
        img = np.array(Image.open(frame_path).convert("RGB"))

        # Detect objects
        boxes, scores, labels = detect_objects(img, args.keywords, args.precision)
        n_detections = len(boxes)
        n_detections_total += n_detections

        if n_detections > 0:
            # Segment with SAM2
            mask = segment_boxes(img, boxes)
        else:
            # No detections — empty mask
            mask = np.zeros(img.shape[:2], dtype=bool)

        # Apply post-processing
        mask_uint8 = apply_mask_ops(mask, args.expand, args.feather, args.invert)

        # Generate output
        output = generate_output(img, mask_uint8, args.mode)

        # Save — COLMAP expects mask filenames as <image_name>.png (e.g. 0001.jpg.png)
        out_name = frame_path.name + ".png"
        output.save(args.output_dir / out_name)

        print(f"[MASK] {i + 1}/{N} {frame_path.name} — {n_detections} detections", flush=True)

    elapsed = time.time() - t_start
    print(f"[INFO] Masking complete: {N} masks generated, "
          f"{n_detections_total} total detections in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
