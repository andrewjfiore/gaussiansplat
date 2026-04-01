import json
import re
from pathlib import Path
from typing import Optional

from ..config import settings


def build_extract_cmd(video_path: Path, output_dir: Path, fps: float = 2.0,
                      start_time: Optional[float] = None,
                      video_prefix: str = "") -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = str(settings.ffmpeg_bin)

    cmd = [
        ffmpeg,
        "-y",                          # overwrite without prompting (prevents stdin hang)
        "-hwaccel", "auto",            # use GPU decoding if available
    ]
    if start_time is not None:
        cmd += ["-ss", f"{start_time:.2f}"]
    cmd += [
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-vsync", "vfr",              # drop duplicate frames
        "-q:v", "2",                   # high quality JPEG
        "-threads", "0",              # use all CPU cores
        "-progress", "pipe:1",        # structured progress to stdout
        str(output_dir / f"{video_prefix}%04d.jpg"),
    ]
    return cmd


def generate_frame_manifest(frames_dir: Path, fps: float,
                            video_sources: list[dict] | None = None) -> dict:
    """Generate a manifest.json mapping frames to normalized timestamps.

    Args:
        frames_dir: Directory containing extracted .jpg frames.
        fps: Extraction FPS.
        video_sources: Optional list of {"video_id": int, "filename": str, "prefix": str}.
                       When provided, frames are grouped by prefix for per-video timestamps.
    """
    frames = sorted(frames_dir.glob("*.jpg"))
    n = len(frames)

    # Detect video_id from filename prefix (e.g., "v0_0001.jpg" → video_id=0)
    def _video_id(name: str) -> int:
        if video_sources:
            for vs in video_sources:
                if name.startswith(vs.get("prefix", "")):
                    return vs["video_id"]
        return 0

    # Count frames per video for per-video t_norm
    from collections import Counter
    vid_counts: Counter = Counter()
    vid_indices: dict[int, int] = {}  # video_id → running frame index within that video

    manifest = {
        "fps": fps,
        "n_frames": n,
        "n_videos": len(video_sources) if video_sources else 1,
        "duration": n / fps if fps > 0 else 0,
        "frames": [],
    }
    if video_sources:
        manifest["video_sources"] = video_sources

    # First pass: count frames per video
    frame_vids = []
    for f in frames:
        vid = _video_id(f.name)
        vid_counts[vid] += 1
        frame_vids.append(vid)

    # Second pass: build frame entries with per-video t_norm
    vid_cur: dict[int, int] = {}
    for i, f in enumerate(frames):
        vid = frame_vids[i]
        vid_cur.setdefault(vid, 0)
        vid_frame_idx = vid_cur[vid]
        vid_total = vid_counts[vid]

        t_sec = vid_frame_idx / fps if fps > 0 else 0
        t_norm = vid_frame_idx / max(vid_total - 1, 1)

        manifest["frames"].append({
            "index": i,
            "file": f.name,
            "video_id": vid,
            "t_sec": round(t_sec, 4),
            "t_norm": round(t_norm, 6),
        })
        vid_cur[vid] = vid_frame_idx + 1

    manifest_path = frames_dir / "manifest.json"
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)

    return manifest


def parse_ffmpeg_line(line: str) -> Optional[dict]:
    # -progress pipe:1 outputs "frame=N" lines
    m = re.search(r"frame=\s*(\d+)", line)
    if m:
        return {"percent": -1}  # We don't know total ahead of time
    # Also detect "out_time=" for progress
    if "out_time=" in line:
        return {"percent": -1}
    # Detect completion
    if line.strip() == "progress=end":
        return {"percent": 100}
    return None
