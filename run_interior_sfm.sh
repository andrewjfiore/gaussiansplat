#!/bin/bash
set -e
export QT_QPA_PLATFORM=offscreen
export EGL_PLATFORM=surfaceless
cd ~/Projects/gaussiansplat

PROJECT_DIR="data/projects/interior-walkthru"
COLMAP_DB="$PROJECT_DIR/colmap/database.db"
FRAMES="$PROJECT_DIR/frames"
SPARSE="$PROJECT_DIR/colmap/sparse"
DENSE="$PROJECT_DIR/colmap/dense"

echo "=== Interior Pipeline: SfM + Training ==="
echo "$(date)"

# Wait for feature extraction to finish (check if process is still running)
while pgrep -f "feature_extractor.*interior-walkthru" > /dev/null 2>&1; do
    COUNT=$(python3 -c "import sqlite3; c=sqlite3.connect('$COLMAP_DB'); print(c.execute('SELECT count(*) FROM images').fetchone()[0])")
    echo "  Feature extraction in progress: $COUNT/546 images..."
    sleep 30
done

COUNT=$(python3 -c "import sqlite3; c=sqlite3.connect('$COLMAP_DB'); print(c.execute('SELECT count(*) FROM images').fetchone()[0])")
echo "Feature extraction complete: $COUNT images"

# Sequential matching (CPU mode)
echo ""
echo "[SfM 2/4] Sequential matching..."
colmap sequential_matcher \
    --database_path "$COLMAP_DB" \
    --SequentialMatching.overlap 15 \
    --SiftMatching.use_gpu 0 2>&1 | tail -10
echo "  Matching done"

# Mapping
echo ""
echo "[SfM 3/4] Incremental mapping..."
rm -rf "$SPARSE"
mkdir -p "$SPARSE"
colmap mapper \
    --database_path "$COLMAP_DB" \
    --image_path "$FRAMES" \
    --output_path "$SPARSE" 2>&1 | tail -20

SPARSE_MODEL="$SPARSE/0"
if [ ! -d "$SPARSE_MODEL" ]; then
    SPARSE_MODEL=$(ls -d "$SPARSE/"*/ 2>/dev/null | head -1)
fi
if [ -z "$SPARSE_MODEL" ] || [ ! -d "$SPARSE_MODEL" ]; then
    echo "ERROR: No sparse model found"
    exit 1
fi
echo "  Sparse model: $SPARSE_MODEL"

# Undistortion
echo ""
echo "[SfM 4/4] Undistortion..."
rm -rf "$DENSE"
colmap image_undistorter \
    --image_path "$FRAMES" \
    --input_path "$SPARSE_MODEL" \
    --output_path "$DENSE" \
    --output_type COLMAP 2>&1 | tail -5

REGISTERED=$(ls "$DENSE/images/"*.jpg 2>/dev/null | wc -l)
echo "  Undistorted $REGISTERED images"

# Dense reconstruction
echo ""
echo "[Dense] Patch match stereo..."
colmap patch_match_stereo \
    --workspace_path "$DENSE" \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.gpu_index 0 2>&1 | tail -10

echo "[Dense] Stereo fusion..."
colmap stereo_fusion \
    --workspace_path "$DENSE" \
    --output_path "$DENSE/fused.ply" 2>&1 | tail -5

# Training
echo ""
echo "[Train] Scaffold-GS — 15000 steps, SH degree 2..."
python3 backend/scripts/train_scaffold.py \
    --data_dir "$DENSE" \
    --result_dir "$PROJECT_DIR/output" \
    --max_steps 15000 \
    --sh_degree 2 \
    --voxel_size 0.001 2>&1 | tail -30

echo ""
echo "=== Pipeline Complete ==="
echo "$(date)"
if [ -f "$PROJECT_DIR/output/point_cloud.ply" ]; then
    PLY_SIZE=$(du -h "$PROJECT_DIR/output/point_cloud.ply" | cut -f1)
    echo "Output: $PROJECT_DIR/output/point_cloud.ply ($PLY_SIZE)"
    echo "SUCCESS"
else
    echo "WARNING: No point_cloud.ply found"
fi
