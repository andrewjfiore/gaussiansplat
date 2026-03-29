#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_tests.sh — Master test runner for GaussianSplat Studio
#
# Usage:
#   ./tests/run_tests.sh               # Run all fast tests (no GPU required)
#   ./tests/run_tests.sh --slow        # Include @pytest.mark.slow tests
#   ./tests/run_tests.sh --unit        # Unit tests only
#   ./tests/run_tests.sh --api         # API tests only
#   ./tests/run_tests.sh --e2e         # E2E tests only (quick subset)
#   ./tests/run_tests.sh --frontend    # Frontend tests only
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_ROOT/.venv"
LOGS_DIR="$SCRIPT_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOGS_DIR/test_run_${TIMESTAMP}.log"

# ── Argument parsing ──────────────────────────────────────────────────────────
RUN_SLOW=false
PYTEST_ARGS=()
FILTER=""

for arg in "$@"; do
    case "$arg" in
        --slow)       RUN_SLOW=true ;;
        --unit)       FILTER="tests/test_pipeline_unit.py" ;;
        --api)        FILTER="tests/test_api.py" ;;
        --e2e)        FILTER="tests/test_pipeline_e2e.py" ;;
        --frontend)   FILTER="tests/test_frontend.py" ;;
        --help|-h)
            sed -n '3,14p' "$0"
            exit 0
            ;;
        *)
            PYTEST_ARGS+=("$arg")
            ;;
    esac
done

if $RUN_SLOW; then
    PYTEST_ARGS+=("--run-slow")
    echo "[run_tests.sh] Slow tests ENABLED (this may take hours)"
fi

# ── Logging setup ─────────────────────────────────────────────────────────────
mkdir -p "$LOGS_DIR"

tee_log() {
    # Write to stdout AND the log file
    tee -a "$LOG_FILE"
}

exec > >(tee_log) 2>&1

echo "════════════════════════════════════════════════════════════════════"
echo " GaussianSplat Studio — Test Suite"
echo " Timestamp : $TIMESTAMP"
echo " Log file  : $LOG_FILE"
echo " Project   : $PROJECT_ROOT"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# ── Environment ───────────────────────────────────────────────────────────────
echo "── Environment ──────────────────────────────────────────────────────"

# Activate venv
if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    echo "venv     : $VENV (activated)"
else
    echo "WARNING: .venv not found at $VENV — using system Python"
fi

echo "Python   : $(python --version 2>&1)"
echo "Pip      : $(pip --version 2>&1 | awk '{print $1, $2}')"
echo "Platform : $(uname -srm)"
echo ""

# ── GPU info ──────────────────────────────────────────────────────────────────
echo "── GPU Information ──────────────────────────────────────────────────"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free \
               --format=csv,noheader 2>/dev/null || echo "(nvidia-smi query failed)"
    echo ""
    echo "nvidia-smi driver version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true
else
    echo "nvidia-smi not found — no GPU detected"
fi
echo ""

# Check PyTorch CUDA
python -c "
import sys
try:
    import torch
    print(f'PyTorch   : {torch.__version__}')
    print(f'CUDA avail: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU       : {torch.cuda.get_device_name(0)}')
        vram = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        print(f'VRAM      : {vram} MB')
        free = torch.cuda.mem_get_info()[0] // (1024**2)
        print(f'VRAM free : {free} MB')
except ImportError:
    print('PyTorch not installed')
" 2>/dev/null || echo "(PyTorch check failed)"
echo ""

# ── Dependency versions ───────────────────────────────────────────────────────
echo "── Key Dependencies ─────────────────────────────────────────────────"
python -c "
deps = ['fastapi', 'uvicorn', 'aiosqlite', 'httpx', 'pydantic', 'gsplat']
for dep in deps:
    try:
        import importlib
        m = importlib.import_module(dep.replace('-', '_'))
        ver = getattr(m, '__version__', '?')
        print(f'  {dep:<20} {ver}')
    except ImportError:
        print(f'  {dep:<20} NOT INSTALLED')
" 2>/dev/null
echo ""

# ── FFmpeg & COLMAP ───────────────────────────────────────────────────────────
echo "── External Tools ───────────────────────────────────────────────────"
ffmpeg -version 2>&1 | head -1 || echo "ffmpeg: NOT FOUND"
colmap -h 2>&1 | head -1 || colmap help 2>&1 | head -1 || echo "colmap: NOT FOUND"
echo ""

# ── Generate synthetic test video if missing ──────────────────────────────────
SYNTHETIC="$SCRIPT_DIR/fixtures/small_video.mp4"
if [[ ! -f "$SYNTHETIC" || ! -s "$SYNTHETIC" ]]; then
    echo "── Generating synthetic test video ──────────────────────────────────"
    python "$SCRIPT_DIR/fixtures/generate_synthetic.py"
    echo ""
fi

# ── Run pytest ────────────────────────────────────────────────────────────────
echo "── Running Tests ────────────────────────────────────────────────────"
cd "$PROJECT_ROOT"

PYTEST_CMD=(
    python -m pytest
    -v
    --tb=short
    --log-cli-level=INFO
    "--log-file=$LOG_FILE"
    "--log-file-level=DEBUG"
    -p no:warnings
)

# Markers: skip slow tests unless explicitly requested
if ! $RUN_SLOW; then
    PYTEST_CMD+=("-m" "not slow")
fi

# Add any extra args
PYTEST_CMD+=("${PYTEST_ARGS[@]}")

# Add test filter or default to all tests
if [[ -n "$FILTER" ]]; then
    PYTEST_CMD+=("$FILTER")
else
    PYTEST_CMD+=("tests/")
fi

echo "Command: ${PYTEST_CMD[*]}"
echo "────────────────────────────────────────────────────────────────────"
echo ""

START_TIME=$(date +%s)
set +e
"${PYTEST_CMD[@]}"
PYTEST_EXIT=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " Test run completed in ${ELAPSED}s"
echo " Exit code : $PYTEST_EXIT"
echo " Full log  : $LOG_FILE"
echo "════════════════════════════════════════════════════════════════════"

exit $PYTEST_EXIT
