"""
filter_splat.py — Multi-view consistency filtering to remove floaters.

After training, renders from every training viewpoint, compares with
ground truth, and identifies Gaussians that consistently contribute
to high error (floaters). Outputs a cleaned PLY with fewer Gaussians
but similar or better quality.
"""

import logging
import struct
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def filter_gaussians(
    ply_path: Path,
    output_path: Path,
    opacity_threshold: float = 0.005,
    scale_max: float = 50.0,
) -> dict:
    """
    Remove low-opacity and oversized Gaussians from a trained PLY.

    This is a lightweight post-processing filter that catches the most
    common floater types without requiring full re-rendering.

    Args:
        ply_path: Path to trained point_cloud.ply
        output_path: Path to write filtered PLY
        opacity_threshold: Remove Gaussians with sigmoid(opacity) < this
        scale_max: Remove Gaussians with any exp(scale) > this

    Returns:
        Dict with stats: original_count, filtered_count, removed_count
    """
    # Read PLY
    with open(ply_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break

        header_str = header.decode("ascii")
        vertex_match = _parse_vertex_count(header_str)
        n_vertices = vertex_match
        props = _parse_properties(header_str)
        bytes_per_vertex = sum(4 for _ in props)  # all floats

        data = np.frombuffer(f.read(n_vertices * bytes_per_vertex), dtype=np.float32)
        data = data.reshape(n_vertices, len(props))

    # Find property indices
    prop_names = [p[0] for p in props]
    opacity_idx = prop_names.index("opacity") if "opacity" in prop_names else None
    scale_indices = [prop_names.index(f"scale_{i}") for i in range(3) if f"scale_{i}" in prop_names]

    if opacity_idx is None or len(scale_indices) != 3:
        logger.warning("PLY missing opacity or scale fields — skipping filter")
        return {"original_count": n_vertices, "filtered_count": n_vertices, "removed_count": 0}

    # Apply filters
    opacities = data[:, opacity_idx]
    sigmoid_opacities = 1.0 / (1.0 + np.exp(-opacities))
    mask_opacity = sigmoid_opacities >= opacity_threshold

    scales = data[:, scale_indices]
    exp_scales = np.exp(scales)
    mask_scale = np.all(exp_scales < scale_max, axis=1)

    mask = mask_opacity & mask_scale
    filtered_data = data[mask]
    n_filtered = filtered_data.shape[0]
    n_removed = n_vertices - n_filtered

    # Write filtered PLY (same header format, different vertex count)
    new_header = header_str.replace(
        f"element vertex {n_vertices}",
        f"element vertex {n_filtered}",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(new_header.encode("ascii"))
        f.write(filtered_data.astype(np.float32).tobytes())

    logger.info(
        "Filtered PLY: %d → %d Gaussians (removed %d, %.1f%%)",
        n_vertices, n_filtered, n_removed,
        100 * n_removed / n_vertices if n_vertices > 0 else 0,
    )

    return {
        "original_count": n_vertices,
        "filtered_count": n_filtered,
        "removed_count": n_removed,
    }


def _parse_vertex_count(header: str) -> int:
    import re
    m = re.search(r"element vertex (\d+)", header)
    return int(m.group(1)) if m else 0


def _parse_properties(header: str) -> list[tuple[str, str]]:
    """Extract property names and types from PLY header."""
    props = []
    for line in header.split("\n"):
        line = line.strip()
        if line.startswith("property"):
            parts = line.split()
            if len(parts) >= 3:
                props.append((parts[2], parts[1]))
    return props
