"""
masking.py — Object masking service using SAM2 + GroundingDINO (built-in)
             or external AutoMasker CLI executable.

Supports two modes:
  - Keyword mode: GroundingDINO detects objects by text → SAM2 segments
  - Point mode: user clicks on a reference frame → SAM2 segments + propagates

Masks are binary images (white = keep, black = mask out) saved alongside
extracted frames. COLMAP and training scripts use them to ignore masked
regions during reconstruction.
"""

import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

from ..config import settings

logger = logging.getLogger(__name__)

# Built-in masking script location
_MASK_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_masks.py"


def find_automasker_exe() -> Optional[Path]:
    """Find external AutoMasker executable."""
    if settings.automasker_exe:
        p = Path(settings.automasker_exe)
        if p.exists():
            return p

    tools_mask = settings.tools_dir / "automasker"
    if tools_mask.exists():
        for name in ["AutoMasker.exe", "automasker.exe", "AutoMasker"]:
            exe = tools_mask / name
            if exe.exists():
                return exe
        for exe in tools_mask.rglob("AutoMasker*"):
            if exe.is_file() and exe.suffix in (".exe", ""):
                return exe

    which = shutil.which("AutoMasker") or shutil.which("automasker")
    if which:
        return Path(which)

    return None


def build_external_mask_cmd(
    input_dir: Path,
    output_dir: Path,
    keywords: str,
    mode: str = "mask",
    invert: bool = False,
    precision: float = 0.3,
    expand: int = 0,
    feather: int = 0,
) -> list[str]:
    """Build CLI command for external AutoMasker executable."""
    exe = find_automasker_exe()
    if not exe:
        raise FileNotFoundError(
            "AutoMasker executable not found. Place it in tools/automasker/ "
            "or set AUTOMASKER_EXE in settings."
        )

    cmd = [
        str(exe),
        "--input", str(input_dir),
        "--output", str(output_dir),
        "--keywords", keywords,
        "--mode", mode,
        "--precision", str(precision),
    ]
    if invert:
        cmd.append("--invert")
    if expand > 0:
        cmd += ["--expand", str(expand)]
    if feather > 0:
        cmd += ["--feather", str(feather)]
    return cmd


def build_builtin_mask_cmd(
    input_dir: Path,
    output_dir: Path,
    keywords: str,
    mode: str = "mask",
    invert: bool = False,
    precision: float = 0.3,
    expand: int = 0,
    feather: int = 0,
) -> list[str]:
    """Build command for built-in keyword-based masking (GroundingDINO + SAM2)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(_MASK_SCRIPT),
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
        "--keywords", keywords,
        "--mode", mode,
        "--precision", str(precision),
    ]
    if invert:
        cmd.append("--invert")
    if expand > 0:
        cmd += ["--expand", str(expand)]
    if feather > 0:
        cmd += ["--feather", str(feather)]
    return cmd


def build_point_mask_cmd(
    input_dir: Path,
    output_dir: Path,
    reference_frame: str,
    points: list[list[float]],
    point_labels: list[int],
    mode: str = "mask",
    invert: bool = False,
    expand: int = 0,
    feather: int = 0,
) -> list[str]:
    """Build command for point-prompted SAM2 masking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    points_data = json.dumps({
        "frame": reference_frame,
        "points": points,
        "labels": point_labels,
    })
    cmd = [
        sys.executable, str(_MASK_SCRIPT),
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
        "--points_json", points_data,
        "--mode", mode,
    ]
    if invert:
        cmd.append("--invert")
    if expand > 0:
        cmd += ["--expand", str(expand)]
    if feather > 0:
        cmd += ["--feather", str(feather)]
    return cmd


def parse_mask_line(line: str) -> Optional[dict]:
    """Parse masking progress output."""
    m = re.match(r"\[MASK\]\s*(\d+)/(\d+)", line)
    if m:
        current, total = int(m.group(1)), int(m.group(2))
        return {"percent": (current / total) * 100 if total > 0 else 0}

    if "complete" in line.lower() and "mask" in line.lower():
        return {"percent": 100}

    return None
