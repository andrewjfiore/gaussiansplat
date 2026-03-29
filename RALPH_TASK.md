# Ralph Task: GaussianSplat Studio — End-to-End Pipeline + Testing

## System Context
- **Machine**: Ubuntu desktop, RTX 3060 12GB, CUDA 13.1, 136GB free disk
- **Repo**: ~/Projects/gaussiansplat (already has frontend + backend scaffolding)
- **Goal**: Working end-to-end video-to-3DGS pipeline with automated testing

## Phase 1: Environment Setup (DO THIS FIRST)
1. Install COLMAP: `sudo apt install -y colmap`
2. Install ffmpeg if missing: `sudo apt install -y ffmpeg`
3. Create Python venv and install deps:
   ```
   cd ~/Projects/gaussiansplat
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install gsplat
   ```
4. Install frontend deps: `cd frontend && npm install`
5. **Verify CUDA**: `python3 -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('CUDA OK:', torch.cuda.get_device_name(0))"`
6. **Verify COLMAP**: `colmap -h`
7. **Verify gsplat**: `python3 -c "import gsplat; print('gsplat OK')"`

## Phase 2: Download Test Videos from Pexels
1. Get a Pexels API key or use direct download URLs
2. Download 2-3 suitable videos for 3D reconstruction:
   - Drone footage of buildings/landscapes (orbit or flyover)
   - Walkaround footage of objects/statues
   - 15-60 seconds each, 1080p minimum
3. Save to `~/Projects/gaussiansplat/test_data/videos/`
4. Name them descriptively: `drone_building.mp4`, `walkaround_statue.mp4`, etc.

**Pexels video search tips**: Search for "drone orbit building", "360 object", "turntable product" — you need multi-view coverage, not flat pans. Download via their API or direct links.

If Pexels API doesn't work, use `yt-dlp` or `curl` to grab Creative Commons videos from alternative sources. The key requirement is multi-angle coverage of a 3D scene.

## Phase 3: Fix/Complete the Pipeline
The existing code has a frontend + backend scaffold. Make the full pipeline work:

1. **Video → Frames** (ffmpeg service): Extract frames at configurable FPS
2. **Frames → COLMAP SfM**: Feature extraction, matching, sparse reconstruction, undistortion
3. **COLMAP → Gaussian Splat Training**: Using gsplat library
4. **Training → PLY output**: Export trained splat as .ply file
5. **PLY → Web Viewer**: In-browser 3D viewer (already scaffolded with GaussianSplats3D)

### Key technical details:
- COLMAP should use `exhaustive_matcher` for small datasets, `sequential_matcher` for video
- Frame extraction: ~2-3 FPS is usually enough (not every frame)
- Training: Start with 7000 iterations for testing, 30000 for production
- The backend should stream real-time logs via WebSocket (already scaffolded)

## Phase 4: Build the E2E Testing System (CRITICAL)
Create a comprehensive automated testing system:

### Structure:
```
tests/
├── conftest.py              # Shared fixtures, test data paths
├── test_pipeline_unit.py    # Unit tests for each pipeline stage
├── test_pipeline_e2e.py     # Full end-to-end pipeline test
├── test_api.py              # API endpoint tests
├── test_frontend.py         # Frontend build/lint tests
├── fixtures/
│   └── small_video.mp4      # Tiny test video (generate synthetically if needed)
└── logs/
    └── .gitkeep             # Test run logs go here
```

### Testing requirements:
1. **Verbose logging**: Every test step logs to both console and `tests/logs/test_run_TIMESTAMP.log`
2. **Stage isolation**: Can test each pipeline stage independently
3. **Synthetic test data**: Generate a small synthetic video (colored cubes rotating) for fast CI tests
4. **Real data tests**: Run against downloaded Pexels videos (slower, marked with `@pytest.mark.slow`)
5. **Error scenarios**: Test with corrupt video, missing COLMAP, insufficient GPU memory
6. **Metrics tracking**: Log timing for each stage, GPU memory usage, output quality metrics
7. **Exit codes**: Clear pass/fail with detailed error messages

### Test runner script:
```bash
#!/bin/bash
# tests/run_tests.sh
# Runs the full test suite with verbose logging
set -euo pipefail

LOGDIR="tests/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGDIR/test_run_$TIMESTAMP.log"

mkdir -p "$LOGDIR"

echo "=== GaussianSplat Test Suite ===" | tee "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a "$LOGFILE"

# Unit tests
echo "--- Unit Tests ---" | tee -a "$LOGFILE"
python -m pytest tests/test_pipeline_unit.py -v --tb=long 2>&1 | tee -a "$LOGFILE"

# API tests
echo "--- API Tests ---" | tee -a "$LOGFILE"
python -m pytest tests/test_api.py -v --tb=long 2>&1 | tee -a "$LOGFILE"

# E2E tests (slow)
echo "--- E2E Pipeline Tests ---" | tee -a "$LOGFILE"
python -m pytest tests/test_pipeline_e2e.py -v --tb=long -s 2>&1 | tee -a "$LOGFILE"

echo "=== Complete: $(date) ===" | tee -a "$LOGFILE"
echo "Log: $LOGFILE"
```

## Phase 5: Run It
1. Run the full pipeline on at least one Pexels video
2. Verify the .ply output exists and is valid
3. Run all tests and fix any failures
4. Commit everything with clear commit messages

## Rules
- **Commit often** with descriptive messages
- **Log everything** — verbose output, timestamps, GPU stats
- **If something fails, fix it** — don't skip
- **Test on real data** — synthetic tests are for CI, but prove it works on real video
- **Push to GitHub** when each major phase completes

## Completion Signal
When completely finished, run:
```
openclaw system event --text "Done: GaussianSplat pipeline complete — environment setup, Pexels videos downloaded, pipeline working, E2E tests passing" --mode now
```
