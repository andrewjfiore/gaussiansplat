import re
from pathlib import Path
from typing import Optional

from ..config import settings


def build_extract_cmd(video_path: Path, output_dir: Path, fps: float = 2.0) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = str(settings.ffmpeg_bin)

    return [
        ffmpeg,
        "-y",                          # overwrite without prompting (prevents stdin hang)
        "-hwaccel", "auto",            # use GPU decoding if available
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-vsync", "vfr",              # drop duplicate frames
        "-q:v", "2",                   # high quality JPEG
        "-threads", "0",              # use all CPU cores
        "-progress", "pipe:1",        # structured progress to stdout
        str(output_dir / "%04d.jpg"),
    ]


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
