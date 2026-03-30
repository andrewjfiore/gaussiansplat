"""
compress.py — Convert .ply splat files to .ksplat compressed format.

.ksplat is the compressed binary format used by mkkellogg/GaussianSplats3D.
Compression typically achieves 70-90% size reduction, critical for Quest
streaming over wifi.

Requires: npm install -g ksplat-encoder
"""

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def compress_splat(ply_path: Path, output_path: Path) -> Path:
    """
    Convert .ply to .ksplat format using the ksplat-encoder npm package.
    Falls back to returning the original .ply if compression fails.
    """
    try:
        result = subprocess.run(
            ["npx", "ksplat-encoder", str(ply_path), str(output_path)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0 and output_path.exists():
            ply_size  = ply_path.stat().st_size
            ksplat_sz = output_path.stat().st_size
            reduction = 100 * (1 - ksplat_sz / ply_size) if ply_size else 0
            log.info(
                "Compressed: %s → %s (%.0f%% reduction, %dMB → %dMB)",
                ply_path.name, output_path.name, reduction,
                ply_size // 1_000_000, ksplat_sz // 1_000_000,
            )
            return output_path
        else:
            log.warning("ksplat compression failed (rc=%d): %s", result.returncode, result.stderr[:200])
    except FileNotFoundError:
        log.warning("npx/ksplat-encoder not found; using raw .ply")
    except subprocess.TimeoutExpired:
        log.warning("ksplat compression timed out; using raw .ply")
    except Exception as e:
        log.warning("ksplat compression error: %s", e)

    return ply_path


def get_ksplat_path(ply_path: Path) -> Path:
    """Return the expected .ksplat path for a given .ply."""
    return ply_path.with_suffix(".ksplat")


def ensure_ksplat(ply_path: Path) -> Path:
    """
    Return .ksplat path if it exists (or can be created), else ply_path.
    Compresses on demand if .ksplat is missing.
    """
    ksplat_path = get_ksplat_path(ply_path)
    if ksplat_path.exists():
        return ksplat_path
    if not ply_path.exists():
        return ply_path
    return compress_splat(ply_path, ksplat_path)
