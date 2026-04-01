#!/bin/bash
set -e
export QT_QPA_PLATFORM=offscreen
export EGL_PLATFORM=surfaceless
cd ~/Projects/gaussiansplat

PD="data/projects/interior-walkthru"
DB="$PD/colmap/database.db"
FRAMES="$PD/frames"
SPARSE="$PD/colmap/sparse"
DENSE="$PD/colmap/dense"

echo "=== Interior Pipeline: Matching -> Training ==="
echo "$(date)"

echo ""
echo "[1/5] Sequential matching (CPU)..."
colmap sequential_matcher \
    --database_path "$DB" \
    --SequentialMatching.overlap 15 \
    --SiftMatching.use_gpu 0 2>&1 | grep -E "^(I|W|E|Matching)" | tail -20
echo "  Done"

MATCH_COUNT=$(python3 -c "import sqlite3; c=sqlite3.connect('$DB'); print(c.execute('SELECT count(*) FROM matches').fetchone()[0])")
echo "  Match pairs: $MATCH_COUNT"

echo ""
echo "[2/5] Incremental mapping..."
rm -rf "$SPARSE"
mkdir -p "$SPARSE"
colmap mapper \
    --database_path "$DB" \
    --image_path "$FRAMES" \
    --output_path "$SPARSE" 2>&1 | grep -E "^(I|W|E)" | tail -30

SPARSE_MODEL="$SPARSE/0"
if [ ! -d "$SPARSE_MODEL" ]; then
    SPARSE_MODEL=$(ls -d "$SPARSE/"*/ 2>/dev/null | head -1)
fi
if [ -z "$SPARSE_MODEL" ] || [ ! -d "$SPARSE_MODEL" ]; then
    echo "ERROR: No sparse model found"
    exit 1
fi
echo "  Sparse model: $SPARSE_MODEL"

echo ""
echo "[3/5] Undistortion..."
rm -rf "$DENSE"
colmap image_undistorter \
    --image_path "$FRAMES" \
    --input_path "$SPARSE_MODEL" \
    --output_path "$DENSE" \
    --output_type COLMAP 2>&1 | tail -5

REG=$(ls "$DENSE/images/"*.jpg 2>/dev/null | wc -l)
echo "  Registered images: $REG"

echo ""
echo "[4/5] Dense reconstruction..."
echo "  Patch match stereo..."
colmap patch_match_stereo \
    --workspace_path "$DENSE" \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.gpu_index 0 2>&1 | tail -5

echo "  Stereo fusion..."
colmap stereo_fusion \
    --workspace_path "$DENSE" \
    --output_path "$DENSE/fused.ply" 2>&1 | tail -5

echo ""
echo "[5/5] Training Scaffold-GS (15000 steps, SH=2)..."
python3 backend/scripts/train_scaffold.py \
    --data_dir "$DENSE" \
    --result_dir "$PD/output" \
    --max_steps 15000 \
    --sh_degree 2 \
    --voxel_size 0.001 2>&1 | tail -30

echo ""
echo "=== Pipeline Complete ==="
echo "$(date)"
if [ -f "$PD/output/point_cloud.ply" ]; then
    PLY_SIZE=$(du -h "$PD/output/point_cloud.ply" | cut -f1)
    echo "Output: $PD/output/point_cloud.ply ($PLY_SIZE)"
    echo "SUCCESS"
else
    echo "WARNING: No point_cloud.ply found"
fi
