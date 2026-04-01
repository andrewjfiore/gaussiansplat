#!/usr/bin/env python3
"""Interior walkthrough pipeline v2 — equirect crops + exhaustive matching + proper training"""
import sys, os, subprocess, time, shutil
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["EGL_PLATFORM"] = "surfaceless"
os.chdir(os.path.expanduser("~/Projects/gaussiansplat"))

PROJECT = Path("data/projects/interior-walkthru")
FRAMES_RAW = PROJECT / "frames"         # raw equirect frames (already extracted)
FRAMES_CROP = PROJECT / "frames_crop"   # perspective crops for COLMAP
COLMAP_DIR = PROJECT / "colmap_v2"
SPARSE = COLMAP_DIR / "sparse"
DENSE = COLMAP_DIR / "dense"
OUTPUT = PROJECT / "output_v2"

# ── Step 1: Convert equirect frames to perspective crops ─────────────────
print("=== Interior Pipeline v2 ===")
print(f"[1/5] Converting {len(list(FRAMES_RAW.glob('*.jpg')))} equirect frames to perspective crops...")

sys.path.insert(0, "backend/app/services")
from equirect import extract_perspective_crops

if FRAMES_CROP.exists():
    shutil.rmtree(FRAMES_CROP)
FRAMES_CROP.mkdir(parents=True)

frames = sorted(FRAMES_RAW.glob("*.jpg"))
# Subsample: take every 3rd frame (182 frames × 10 crops = 1820 images)
frames = frames[::3]
total_crops = 0
for idx, frame in enumerate(frames):
    crops = extract_perspective_crops(frame, FRAMES_CROP, idx, crop_size=800, fov_deg=90)
    total_crops += len(crops)
    if idx % 20 == 0:
        print(f"  {idx}/{len(frames)} frames → {total_crops} crops", flush=True)

print(f"  Done: {len(frames)} frames → {total_crops} perspective crops")

# ── Step 2: COLMAP SfM ──────────────────────────────────────────────────
print(f"\n[2/5] COLMAP Feature Extraction (CPU)...")
if COLMAP_DIR.exists():
    shutil.rmtree(COLMAP_DIR)
COLMAP_DIR.mkdir(parents=True)
SPARSE.mkdir()
DB = str(COLMAP_DIR / "database.db")

subprocess.run([
    "colmap", "feature_extractor",
    "--database_path", DB,
    "--image_path", str(FRAMES_CROP),
    "--ImageReader.single_camera", "1",
    "--ImageReader.camera_model", "PINHOLE",
    "--SiftExtraction.use_gpu", "0",
    "--SiftExtraction.max_num_features", "4096",
], check=True)

print(f"\n[3/5] COLMAP Exhaustive Matching (CPU)...")
subprocess.run([
    "colmap", "exhaustive_matcher",
    "--database_path", DB,
    "--SiftMatching.use_gpu", "0",
    "--ExhaustiveMatching.block_size", "50",
], check=True)

print(f"\n[4/5] COLMAP Mapping...")
subprocess.run([
    "colmap", "mapper",
    "--database_path", DB,
    "--image_path", str(FRAMES_CROP),
    "--output_path", str(SPARSE),
], check=True)

# Find sparse model
sparse_model = SPARSE / "0"
if not sparse_model.exists():
    subdirs = sorted(d for d in SPARSE.iterdir() if d.is_dir())
    sparse_model = subdirs[0] if subdirs else None
if not sparse_model or not sparse_model.exists():
    print("ERROR: No sparse model")
    sys.exit(1)

# Model stats
subprocess.run(["colmap", "model_analyzer", "--path", str(sparse_model)])

print(f"\nUndistortion...")
DENSE.mkdir(parents=True, exist_ok=True)
subprocess.run([
    "colmap", "image_undistorter",
    "--image_path", str(FRAMES_CROP),
    "--input_path", str(sparse_model),
    "--output_path", str(DENSE),
    "--output_type", "COLMAP",
], check=True)

reg_imgs = len(list((DENSE / "images").glob("*.jpg")))
print(f"Registered images: {reg_imgs}")

# ── Step 5: Training ────────────────────────────────────────────────────
print(f"\n[5/5] Training Scaffold-GS (15000 steps)...")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Use the venv python
venv_python = Path.home() / "Projects/gaussiansplat/.venv/bin/python"
subprocess.run([
    str(venv_python), "backend/scripts/train_scaffold.py",
    "--data_dir", str(DENSE),
    "--result_dir", str(OUTPUT),
    "--max_steps", "15000",
    "--voxel_size", "0.001",
], check=True)

ply = OUTPUT / "point_cloud.ply"
if ply.exists():
    size = ply.stat().st_size / 1024 / 1024
    print(f"\nSUCCESS: {ply} ({size:.1f} MB)")
else:
    print("\nWARNING: No PLY output")
