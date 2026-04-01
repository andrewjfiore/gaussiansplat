#!/bin/bash
set -e
cd ~/Projects/gaussiansplat

PROJECT_ID="interior-walkthru"
PROJECT_DIR="data/projects/$PROJECT_ID"
VIDEO="360_walkthrough_5m34s.mp4"
START_TIME=335  # 5m 35s
FPS=2
MATCHER="sequential_matcher"
SH_DEGREE=2
MAX_STEPS=15000

echo "=== Interior Walkthrough Pipeline ==="
echo "Start time: ${START_TIME}s (5m 35s)"
echo "FPS: $FPS | Matcher: $MATCHER | SH: $SH_DEGREE | Steps: $MAX_STEPS"
echo ""

# Setup project directories
mkdir -p "$PROJECT_DIR"/{input,frames,colmap/sparse,colmap/dense,output}

# Copy video if not already there
if [ ! -f "$PROJECT_DIR/input/$VIDEO" ]; then
    cp "$VIDEO" "$PROJECT_DIR/input/"
    echo "[1/5] Video copied to project"
else
    echo "[1/5] Video already in project"
fi

# ── Step 1: Frame extraction with start time offset ──
echo ""
echo "[2/5] Extracting frames (start=${START_TIME}s, fps=${FPS})..."
rm -f "$PROJECT_DIR/frames/"*.jpg 2>/dev/null
ffmpeg -y -hwaccel auto -ss $START_TIME -i "$PROJECT_DIR/input/$VIDEO" \
    -vf "fps=$FPS" -vsync vfr -q:v 2 -threads 0 \
    "$PROJECT_DIR/frames/%04d.jpg" 2>&1 | tail -3

FRAME_COUNT=$(ls "$PROJECT_DIR/frames/"*.jpg 2>/dev/null | wc -l)
echo "Extracted $FRAME_COUNT frames"

if [ "$FRAME_COUNT" -lt 10 ]; then
    echo "ERROR: Too few frames ($FRAME_COUNT). Aborting."
    exit 1
fi

# ── Step 2: COLMAP Feature Extraction ──
echo ""
echo "[3/5] Running COLMAP SfM..."
COLMAP_DB="$PROJECT_DIR/colmap/database.db"
rm -f "$COLMAP_DB"

echo "  Feature extraction..."
colmap feature_extractor \
    --database_path "$COLMAP_DB" \
    --image_path "$PROJECT_DIR/frames" \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1 2>&1 | tail -5

echo "  Sequential matching..."
colmap sequential_matcher \
    --database_path "$COLMAP_DB" \
    --SequentialMatching.overlap 10 \
    --SiftMatching.use_gpu 1 2>&1 | tail -5

echo "  Mapping..."
colmap mapper \
    --database_path "$COLMAP_DB" \
    --image_path "$PROJECT_DIR/frames" \
    --output_path "$PROJECT_DIR/colmap/sparse" 2>&1 | tail -10

# Find sparse model
SPARSE_MODEL="$PROJECT_DIR/colmap/sparse/0"
if [ ! -d "$SPARSE_MODEL" ]; then
    SPARSE_MODEL=$(ls -d "$PROJECT_DIR/colmap/sparse/"*/ 2>/dev/null | head -1)
fi

if [ -z "$SPARSE_MODEL" ] || [ ! -d "$SPARSE_MODEL" ]; then
    echo "ERROR: No sparse model found. SfM failed."
    exit 1
fi
echo "  Sparse model: $SPARSE_MODEL"

echo "  Undistortion..."
colmap image_undistorter \
    --image_path "$PROJECT_DIR/frames" \
    --input_path "$SPARSE_MODEL" \
    --output_path "$PROJECT_DIR/colmap/dense" \
    --output_type COLMAP 2>&1 | tail -5

# Dense reconstruction
echo "  Dense stereo matching..."
colmap patch_match_stereo \
    --workspace_path "$PROJECT_DIR/colmap/dense" \
    --PatchMatchStereo.geom_consistency true 2>&1 | tail -5

echo "  Stereo fusion..."
colmap stereo_fusion \
    --workspace_path "$PROJECT_DIR/colmap/dense" \
    --output_path "$PROJECT_DIR/colmap/dense/fused.ply" 2>&1 | tail -5

SFM_IMAGES=$(ls "$PROJECT_DIR/colmap/dense/images/"*.jpg 2>/dev/null | wc -l)
echo "  SfM registered $SFM_IMAGES images"

# ── Step 3: Scaffold-GS Training ──
echo ""
echo "[4/5] Training Scaffold-GS (${MAX_STEPS} steps, SH degree ${SH_DEGREE})..."
python3 backend/scripts/train_scaffold.py \
    --data_dir "$PROJECT_DIR/colmap/dense" \
    --result_dir "$PROJECT_DIR/output" \
    --max_steps $MAX_STEPS \
    --sh_degree $SH_DEGREE \
    --voxel_size 0.001 2>&1 | tail -20

# ── Done ──
echo ""
echo "[5/5] Pipeline complete!"
if [ -f "$PROJECT_DIR/output/point_cloud.ply" ]; then
    PLY_SIZE=$(du -h "$PROJECT_DIR/output/point_cloud.ply" | cut -f1)
    echo "Output: $PROJECT_DIR/output/point_cloud.ply ($PLY_SIZE)"
    echo "SUCCESS"
else
    echo "WARNING: No point_cloud.ply found"
fi
