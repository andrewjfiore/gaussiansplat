"""
disk.py — Disk space checks before pipeline operations.
"""

import shutil
from pathlib import Path


# Minimum free space (MB) required for each pipeline step
SPACE_REQUIREMENTS_MB = {
    "extract-frames": 500,
    "sfm": 1000,
    "train": 2000,
}


def check_disk_space(path: Path, required_mb: int) -> tuple[bool, int]:
    """
    Check if the filesystem containing path has at least required_mb free.

    Returns:
        (ok, free_mb) tuple.
    """
    usage = shutil.disk_usage(str(path))
    free_mb = usage.free // (1024 * 1024)
    return free_mb >= required_mb, free_mb
