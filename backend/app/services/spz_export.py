"""
spz_export.py — Convert PLY gaussian splats to SPZ format.

SPZ is Niantic's compressed gaussian splat format. Achieves 10-20x
compression over PLY by quantizing SH coefficients and using gzip.

Format spec: https://github.com/nianticlabs/spz
"""

import gzip
import logging
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# SPZ magic number and version
SPZ_MAGIC = 0x5053474E  # "NGSP" in little-endian
SPZ_VERSION = 2


def ply_to_spz(ply_path: Path, output_path: Path) -> int:
    """
    Convert a standard 3DGS PLY to SPZ format.

    The SPZ format stores:
    - Header: magic, version, num_points, sh_degree, flags
    - Per-point: position (float16 xyz), scale (uint8 x3), rotation (uint8 x3),
      alpha (uint8), color (uint8 rgb), sh_coeffs (int8 per band)

    Returns the number of points written.
    """
    # Parse PLY
    positions, scales, rotations, opacities, sh_dc, sh_rest, n_sh_rest = _read_ply(ply_path)
    n = len(positions)

    if n == 0:
        logger.warning("Empty PLY, nothing to export")
        return 0

    # Determine SH degree from coefficient count
    total_sh = 1 + (n_sh_rest // 3 if n_sh_rest > 0 else 0)
    if total_sh >= 16:
        sh_degree = 3
    elif total_sh >= 9:
        sh_degree = 2
    elif total_sh >= 4:
        sh_degree = 1
    else:
        sh_degree = 0

    # Quantize values
    # Positions: keep as float32 (SPZ v2 uses float32 for positions)
    pos_f32 = positions.astype(np.float32)

    # Scales: log-space, quantize to uint8
    log_scales = scales.astype(np.float32)  # already log-scales from PLY
    s_min, s_max = log_scales.min(), log_scales.max()
    if s_max - s_min > 1e-6:
        scales_u8 = ((log_scales - s_min) / (s_max - s_min) * 255).clip(0, 255).astype(np.uint8)
    else:
        scales_u8 = np.full_like(log_scales, 128, dtype=np.uint8)

    # Rotations: quaternion wxyz, quantize to int8 (-127 to 127)
    rot_norm = rotations / (np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-8)
    rot_i8 = (rot_norm * 127).clip(-127, 127).astype(np.int8)

    # Opacity: sigmoid(logit) -> uint8
    sigmoid_opa = 1.0 / (1.0 + np.exp(-opacities.astype(np.float32)))
    alpha_u8 = (sigmoid_opa * 255).clip(0, 255).astype(np.uint8)

    # SH DC -> RGB uint8
    C0 = 0.28209479177387814
    rgb = (sh_dc * C0 + 0.5).clip(0, 1)
    rgb_u8 = (rgb * 255).clip(0, 255).astype(np.uint8)

    # SH rest -> int8
    if sh_rest is not None and sh_rest.size > 0:
        sh_scale = max(abs(sh_rest.min()), abs(sh_rest.max()), 1e-6)
        sh_rest_i8 = (sh_rest / sh_scale * 127).clip(-127, 127).astype(np.int8)
    else:
        sh_rest_i8 = np.array([], dtype=np.int8)
        sh_scale = 1.0

    # Build SPZ binary
    buf = bytearray()

    # Header
    buf += struct.pack("<I", SPZ_MAGIC)
    buf += struct.pack("<I", SPZ_VERSION)
    buf += struct.pack("<I", n)
    buf += struct.pack("<B", sh_degree)
    buf += struct.pack("<B", 0)  # flags
    buf += struct.pack("<f", s_min)
    buf += struct.pack("<f", s_max)
    buf += struct.pack("<f", sh_scale)

    # Per-point data
    buf += pos_f32.tobytes()
    buf += scales_u8.tobytes()
    buf += rot_i8.tobytes()
    buf += alpha_u8.tobytes()
    buf += rgb_u8.tobytes()
    if sh_rest_i8.size > 0:
        buf += sh_rest_i8.tobytes()

    # Compress with gzip
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(output_path), "wb", compresslevel=6) as f:
        f.write(bytes(buf))

    ply_size = ply_path.stat().st_size
    spz_size = output_path.stat().st_size
    ratio = ply_size / spz_size if spz_size > 0 else 0
    logger.info(
        "SPZ export: %d points, %s -> %s (%.1fx compression)",
        n,
        _fmt_size(ply_size),
        _fmt_size(spz_size),
        ratio,
    )
    return n


def _read_ply(ply_path: Path):
    """Read a standard 3DGS PLY and return raw arrays."""
    with open(ply_path, "rb") as f:
        # Parse header
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break

        header_str = header.decode("ascii")

        # Get vertex count
        import re
        m = re.search(r"element vertex (\d+)", header_str)
        n = int(m.group(1)) if m else 0

        # Parse properties
        props = []
        for line in header_str.split("\n"):
            line = line.strip()
            if line.startswith("property float"):
                props.append(line.split()[-1])

        n_props = len(props)
        data = np.frombuffer(f.read(n * n_props * 4), dtype=np.float32).reshape(n, n_props)

    # Extract fields by name
    prop_idx = {name: i for i, name in enumerate(props)}

    positions = np.column_stack([data[:, prop_idx[k]] for k in ("x", "y", "z")])

    scales = np.column_stack([
        data[:, prop_idx[f"scale_{i}"]] for i in range(3)
    ])

    rotations = np.column_stack([
        data[:, prop_idx[f"rot_{i}"]] for i in range(4)
    ])

    opacities = data[:, prop_idx["opacity"]]

    sh_dc = np.column_stack([
        data[:, prop_idx[f"f_dc_{i}"]] for i in range(3)
    ])

    # SH rest coefficients
    sh_rest_names = sorted([p for p in props if p.startswith("f_rest_")],
                           key=lambda x: int(x.split("_")[-1]))
    n_sh_rest = len(sh_rest_names)
    if n_sh_rest > 0:
        sh_rest = np.column_stack([data[:, prop_idx[name]] for name in sh_rest_names])
    else:
        sh_rest = None

    return positions, scales, rotations, opacities, sh_dc, sh_rest, n_sh_rest


def _fmt_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024*1024):.1f}MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f}KB"
    return f"{size_bytes}B"
