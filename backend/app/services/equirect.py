"""
equirect.py — Equirectangular 360° video detection and perspective crop extraction.

Equirectangular images have a 2:1 aspect ratio (width ≈ 2 × height).
For each frame we generate 10 perspective views: 8 horizontal (every 45°) +
2 vertical (up 60° and down 60°). COLMAP then treats these as normal
perspective images.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Perspective views to extract: (theta_deg, phi_deg)
_VIEWS: List[Tuple[float, float]] = [
    (0, 0), (45, 0), (90, 0), (135, 0),
    (180, 0), (225, 0), (270, 0), (315, 0),
    (0, 60), (0, -60),
]


def is_equirectangular(width: int, height: int, tolerance: float = 0.05) -> bool:
    """Return True if width/height is within tolerance of 2.0."""
    if height == 0:
        return False
    ratio = width / height
    return abs(ratio - 2.0) <= tolerance


def equirect_to_perspective(
    img: np.ndarray,
    fov_deg: float = 90,
    theta_deg: float = 0,
    phi_deg: float = 0,
    width: int = 800,
    height: int = 800,
) -> np.ndarray:
    """
    Extract a perspective view from an equirectangular image.

    theta = horizontal rotation (yaw) in degrees
    phi   = vertical rotation (pitch) in degrees
    """
    f = width / (2 * np.tan(np.radians(fov_deg / 2)))
    cx, cy = width / 2, height / 2
    th = np.radians(theta_deg)
    ph = np.radians(phi_deg)

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - cx) / f
    y = (v - cy) / f
    z = np.ones_like(x)

    cos_th, sin_th = np.cos(th), np.sin(th)
    cos_ph, sin_ph = np.cos(ph), np.sin(ph)

    xr =  cos_th * x - sin_th * z
    yr =  sin_ph * sin_th * x + cos_ph * y + sin_ph * cos_th * z
    zr = -cos_ph * sin_th * x + sin_ph * y + cos_ph * cos_th * z

    lon = np.arctan2(xr, zr)
    lat = np.arcsin(np.clip(yr / np.sqrt(xr**2 + yr**2 + zr**2 + 1e-8), -1, 1))

    src_h, src_w = img.shape[:2]
    map_x = ((lon / np.pi + 1) / 2 * src_w).astype(np.float32)
    map_y = ((-lat / (np.pi / 2) + 1) / 2 * src_h).astype(np.float32)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def extract_perspective_crops(
    frame_path: Path,
    output_dir: Path,
    frame_index: int,
    crop_size: int = 800,
    fov_deg: float = 90,
) -> List[Path]:
    """
    Given one equirectangular frame, produce 10 perspective JPEG crops.
    Returns list of output paths.
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        log.warning("Could not read frame: %s", frame_path)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for view_idx, (theta, phi) in enumerate(_VIEWS):
        crop = equirect_to_perspective(img, fov_deg=fov_deg, theta_deg=theta,
                                       phi_deg=phi, width=crop_size, height=crop_size)
        name = f"frame_{frame_index:04d}_v{view_idx:02d}_t{int(theta):03d}_p{int(phi):+04d}.jpg"
        out_path = output_dir / name
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(out_path)

    return paths


def process_equirect_frames(
    frames_dir: Path,
    output_dir: Path,
    crop_size: int = 800,
) -> int:
    """
    Process all equirectangular frames in frames_dir, writing perspective
    crops to output_dir.  Returns the number of crops produced.
    """
    frames = sorted(frames_dir.glob("*.jpg"))
    total = 0
    for idx, frame in enumerate(frames):
        crops = extract_perspective_crops(frame, output_dir, idx, crop_size)
        total += len(crops)
    log.info("Equirect processing: %d frames → %d perspective crops", len(frames), total)
    return total
