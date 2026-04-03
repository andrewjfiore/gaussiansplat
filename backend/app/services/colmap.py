import re
from pathlib import Path
from typing import Optional

from ..config import settings


def _colmap() -> str:
    """Return the colmap executable path. Uses settings.colmap_bin which
    already handles env vars, recursive search, and system PATH."""
    return str(settings.colmap_bin)


def build_feature_extractor_cmd(db_path: Path, image_path: Path,
                                single_camera: bool = True,
                                mask_path: Path | None = None) -> list[str]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _colmap(), "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_path),
    ]
    if single_camera:
        cmd += ["--ImageReader.single_camera", "1"]
    if mask_path and mask_path.exists():
        cmd += ["--ImageReader.mask_path", str(mask_path)]
    return cmd


def build_matcher_cmd(db_path: Path, matcher_type: str = "sequential_matcher") -> list[str]:
    return [
        _colmap(), matcher_type,
        "--database_path", str(db_path),
    ]


def build_mapper_cmd(db_path: Path, image_path: Path, output_path: Path) -> list[str]:
    output_path.mkdir(parents=True, exist_ok=True)
    return [
        _colmap(), "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_path),
        "--output_path", str(output_path),
    ]


def build_undistorter_cmd(image_path: Path, input_path: Path, output_path: Path) -> list[str]:
    output_path.mkdir(parents=True, exist_ok=True)
    return [
        _colmap(), "image_undistorter",
        "--image_path", str(image_path),
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--output_type", "COLMAP",
    ]


def build_patch_match_stereo_cmd(workspace_path: Path) -> list[str]:
    """Dense stereo matching on undistorted images."""
    return [
        _colmap(), "patch_match_stereo",
        "--workspace_path", str(workspace_path),
    ]


def build_stereo_fusion_cmd(workspace_path: Path, output_ply: Path) -> list[str]:
    """Fuse stereo depth maps into a dense point cloud."""
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    return [
        _colmap(), "stereo_fusion",
        "--workspace_path", str(workspace_path),
        "--output_path", str(output_ply),
    ]


def parse_colmap_line(line: str) -> Optional[dict]:
    # Match progress indicators from COLMAP output
    if "Registering image" in line:
        m = re.search(r"#(\d+)", line)
        if m:
            return {"percent": -1}
    # Feature extraction progress
    if "Processed file" in line or "processed file" in line:
        return {"percent": -1}
    # Matching progress
    if "Matching block" in line or "Verified" in line:
        return {"percent": -1}
    return None
