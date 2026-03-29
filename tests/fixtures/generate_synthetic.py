#!/usr/bin/env python3
"""
Generate a tiny synthetic test video for fast pipeline unit tests.

Usage:
    python tests/fixtures/generate_synthetic.py

Output:
    tests/fixtures/small_video.mp4  — 5-second 320×240 testsrc2 video

This video is used in unit tests that exercise frame extraction, command
building, and the API upload endpoint without needing real footage.
"""
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).parent / "small_video.mp4"


def generate(out: Path = OUT) -> Path:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "testsrc2=size=320x240:rate=10",
        "-t", "5",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr, file=sys.stderr)
        sys.exit(1)
    size_kb = out.stat().st_size // 1024
    print(f"Generated: {out}  ({size_kb} KB)")
    return out


if __name__ == "__main__":
    generate()
