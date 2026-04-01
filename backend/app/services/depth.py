"""
depth.py — Monocular depth estimation using MiDAS (DPT-Hybrid).

Pre-computes per-frame depth maps that can be used as additional
training supervision to reduce floaters and improve geometry.
Uses MiDAS via torch hub (small download, ~100MB model weights).
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Model cache to avoid re-downloading
_model = None
_transform = None


def _load_model(device: str = "cuda"):
    """Load MiDAS DPT-Hybrid model via torch hub (~100MB weights)."""
    global _model, _transform
    if _model is not None:
        return _model, _transform

    logger.info("Loading MiDAS depth model...")
    _model = torch.hub.load("intel-isl/MiDAS", "DPT_Hybrid", trust_repo=True)
    _model = _model.to(device).eval()

    midas_transforms = torch.hub.load("intel-isl/MiDAS", "transforms", trust_repo=True)
    _transform = midas_transforms.dpt_transform

    logger.info("MiDAS depth model loaded successfully")
    return _model, _transform


def estimate_depth_single(image_path: Path, device: str = "cuda") -> np.ndarray:
    """Estimate depth for a single image. Returns normalized float32 depth map."""
    import cv2
    model, transform = _load_model(device)

    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # Normalize to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    return depth.astype(np.float32)


def estimate_depths(
    frames_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    log_callback=None,
) -> int:
    """
    Run depth estimation on all JPEGs in frames_dir.
    Saves normalized depth maps as .npy files to output_dir.

    Args:
        log_callback: optional callable(str) for progress messages (e.g. WS relay)

    Returns the number of frames processed.
    """
    def _log(msg: str):
        logger.info(msg)
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.glob("*.jpg"))

    if not frames:
        _log("[DEPTH] No JPEG frames found — skipping")
        return 0

    count = 0
    for frame_path in frames:
        out_path = output_dir / f"{frame_path.stem}.npy"
        if out_path.exists():
            count += 1
            continue  # Skip already processed

        try:
            depth = estimate_depth_single(frame_path, device)
            np.save(out_path, depth)
            count += 1
            if count % 10 == 0 or count == 1:
                _log(f"[DEPTH] Processed {count}/{len(frames)} frames")
        except Exception as e:
            logger.warning("Depth estimation failed for %s: %s", frame_path.name, e)

    _log(f"[DEPTH] Done: {count}/{len(frames)} depth maps saved")
    return count
