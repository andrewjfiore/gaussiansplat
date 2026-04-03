#!/usr/bin/env python3
"""
prune_splat.py — Post-training Gaussian Splat cleanup.

Removes floaters, oversized blobs, and positional outliers from a trained PLY.
All operations are CPU-only and take seconds.

Usage:
  python prune_splat.py \
    --input point_cloud.ply \
    --output point_cloud_clean.ply \
    --min_opacity 0.1 \
    --max_scale_mult 8 \
    --position_percentile 99
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def load_ply(path):
    """Read a 3DGS PLY file, return header lines + data array + property names."""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if b"end_header" in line:
                break

        header_str = b"".join(header_lines).decode("utf-8", errors="replace")
        n_verts = 0
        props = []
        for h in header_str.split("\n"):
            if h.startswith("element vertex"):
                n_verts = int(h.split()[-1])
            if h.startswith("property float"):
                props.append(h.split()[-1])

        n_floats = len(props)
        raw = f.read(n_verts * n_floats * 4)
        data = np.frombuffer(raw, dtype=np.float32).copy().reshape(n_verts, n_floats)

    return header_lines, data, {name: i for i, name in enumerate(props)}


def save_ply(path, header_lines, data, n_orig):
    """Write a pruned PLY with corrected vertex count in header."""
    n_kept = len(data)
    with open(path, "wb") as f:
        for line in header_lines:
            s = line.decode("utf-8", errors="replace")
            if s.startswith("element vertex"):
                f.write(f"element vertex {n_kept}\n".encode())
            else:
                f.write(line)
        f.write(data.tobytes())


def main():
    ap = argparse.ArgumentParser(description="Prune floaters from a trained Gaussian Splat PLY")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--min_opacity", type=float, default=0.1,
                    help="Remove Gaussians with sigmoid(opacity) below this (0-1)")
    ap.add_argument("--max_scale_mult", type=float, default=8.0,
                    help="Remove Gaussians with max scale > N * median scale")
    ap.add_argument("--position_percentile", type=float, default=99.0,
                    help="Keep only Gaussians within this distance percentile from center")
    ap.add_argument("--bbox", type=str, default=None,
                    help="Crop to bounding box: 'xmin,ymin,zmin,xmax,ymax,zmax'")
    args = ap.parse_args()

    t0 = time.time()
    header_lines, data, prop_idx = load_ply(args.input)
    n_orig = len(data)
    print(f"[INFO] Input: {n_orig:,} Gaussians, {args.input.stat().st_size/1024/1024:.0f} MB", flush=True)

    # ── Opacity pruning ──
    opacity = 1.0 / (1.0 + np.exp(-data[:, prop_idx["opacity"]]))
    keep_opacity = opacity >= args.min_opacity
    n_opacity = (~keep_opacity).sum()

    # ── Scale pruning ──
    scales = np.stack([np.exp(data[:, prop_idx[f"scale_{i}"]]) for i in range(3)], axis=1)
    max_scale = scales.max(axis=1)
    med_scale = np.median(max_scale)
    keep_scale = max_scale <= med_scale * args.max_scale_mult
    n_scale = (~keep_scale).sum()

    # ── Position pruning ──
    positions = data[:, :3]
    center = positions.mean(axis=0)
    dists = np.linalg.norm(positions - center, axis=1)
    dist_thresh = np.percentile(dists, args.position_percentile)
    keep_pos = dists <= dist_thresh
    n_pos = (~keep_pos).sum()

    # ── Bounding box crop ──
    if args.bbox:
        coords = [float(x) for x in args.bbox.split(",")]
        xmin, ymin, zmin, xmax, ymax, zmax = coords
        keep_bbox = (
            (positions[:, 0] >= xmin) & (positions[:, 0] <= xmax) &
            (positions[:, 1] >= ymin) & (positions[:, 1] <= ymax) &
            (positions[:, 2] >= zmin) & (positions[:, 2] <= zmax)
        )
        n_bbox = (~keep_bbox).sum()
    else:
        keep_bbox = np.ones(n_orig, dtype=bool)
        n_bbox = 0

    # ── Combined mask ──
    keep = keep_opacity & keep_scale & keep_pos & keep_bbox
    filtered = data[keep]
    n_kept = len(filtered)
    n_pruned = n_orig - n_kept

    print(f"[PRUNE] opacity < {args.min_opacity}: {n_opacity:,}", flush=True)
    print(f"[PRUNE] scale > {args.max_scale_mult}x median: {n_scale:,}", flush=True)
    print(f"[PRUNE] position > {args.position_percentile}th pctl: {n_pos:,}", flush=True)
    if n_bbox > 0:
        print(f"[PRUNE] outside bbox: {n_bbox:,}", flush=True)
    print(f"[INFO] Total pruned: {n_pruned:,} ({n_pruned/n_orig*100:.1f}%)", flush=True)
    print(f"[INFO] Kept: {n_kept:,} ({n_kept/n_orig*100:.1f}%)", flush=True)

    save_ply(args.output, header_lines, filtered, n_orig)

    elapsed = time.time() - t0
    out_mb = args.output.stat().st_size / 1024 / 1024
    print(f"[INFO] Output: {args.output} ({out_mb:.0f} MB) in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
