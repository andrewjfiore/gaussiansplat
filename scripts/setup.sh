#!/bin/bash
# GaussianSplat Studio — Linux/macOS setup script
# Creates .venv at project root, detects CUDA, installs all dependencies.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Colour helpers ──────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
warn() { echo -e "${YELLOW}  ! $*${NC}"; }
err()  { echo -e "${RED}  ✗ $*${NC}"; }

echo
echo "=== GaussianSplat Studio Setup ==="
echo

# ── 1. Python ───────────────────────────────────────────────────────────────
echo "[1/7] Checking Python..."
PYTHON=$(command -v python3 || command -v python || true)
if [ -z "$PYTHON" ]; then
    err "Python 3.10+ not found. Install it and re-run."
    exit 1
fi
PYVER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ok "Python $PYVER found at $PYTHON"

# ── 2. Virtual environment ───────────────────────────────────────────────────
echo "[2/7] Setting up virtual environment at .venv ..."
if [ ! -d ".venv" ]; then
    "$PYTHON" -m venv .venv
    ok "Created .venv"
else
    ok ".venv already exists"
fi
source .venv/bin/activate

# ── 3. CUDA detection ────────────────────────────────────────────────────────
echo "[3/7] Detecting GPU / CUDA..."
CUDA_TORCH_URL="https://download.pytorch.org/whl/cpu"
CUDA_DETECTED="no"
CUDA_VER=""

if command -v nvidia-smi &>/dev/null; then
    # Extract CUDA version from nvidia-smi header, e.g. "CUDA Version: 12.4"
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || true)
    if [ -n "$CUDA_VER" ]; then
        CUDA_DETECTED="yes"
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        # Map driver CUDA version to nearest torch wheel tag.
        # CUDA drivers are backwards-compatible: CUDA 13.x drivers can run cu126 wheels.
        if   [ "$CUDA_MAJOR" -ge 13 ]; then                              TORCH_CUDA="cu126"
        elif [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 6 ]; then TORCH_CUDA="cu126"
        elif [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then TORCH_CUDA="cu124"
        elif [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then TORCH_CUDA="cu121"
        elif [ "$CUDA_MAJOR" -ge 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then TORCH_CUDA="cu118"
        else TORCH_CUDA="cu118"; fi
        CUDA_TORCH_URL="https://download.pytorch.org/whl/$TORCH_CUDA"
        ok "NVIDIA GPU detected — CUDA $CUDA_VER → will install torch+$TORCH_CUDA"
    else
        warn "nvidia-smi found but couldn't read CUDA version; falling back to CPU torch"
    fi
else
    warn "nvidia-smi not found — installing CPU-only PyTorch (training will be unavailable)"
fi

# ── 4. Install Python dependencies ──────────────────────────────────────────
echo "[4/7] Installing Python dependencies..."

pip install --upgrade pip --quiet

# Base FastAPI deps
pip install -r backend/requirements.txt --quiet
ok "Base dependencies installed"

if [ "$CUDA_DETECTED" = "yes" ]; then
    # Install CUDA torch first (separate step so the index-url applies only here)
    pip install torch torchvision torchaudio --extra-index-url "$CUDA_TORCH_URL" --quiet
    ok "PyTorch (CUDA) installed from $CUDA_TORCH_URL"
    # Install gsplat (requires torch already present)
    pip install gsplat --quiet
    ok "gsplat installed"
    # Additional ML utilities
    pip install numpy Pillow tqdm --quiet
    ok "numpy / Pillow / tqdm installed"
else
    # CPU fallback — still install torch so imports don't fail
    pip install torch torchvision torchaudio --quiet
    warn "PyTorch installed (CPU-only) — GPU training will not be available"
    pip install numpy Pillow tqdm --quiet
fi

# ── 5. COLMAP ────────────────────────────────────────────────────────────────
echo "[5/7] Checking COLMAP..."
if command -v colmap &>/dev/null; then
    ok "COLMAP already available: $(command -v colmap)"
else
    warn "COLMAP not found — attempting install..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y colmap 2>/dev/null && ok "COLMAP installed via apt" || warn "apt install colmap failed — install manually"
        else
            warn "Non-apt Linux — install COLMAP manually: https://colmap.github.io/install.html"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &>/dev/null; then
            brew install colmap && ok "COLMAP installed via brew" || warn "brew install colmap failed"
        else
            warn "Homebrew not found — install from https://colmap.github.io/install.html"
        fi
    else
        warn "Unknown OS — install COLMAP manually: https://colmap.github.io/install.html"
    fi
fi

# ── 6. ffmpeg ────────────────────────────────────────────────────────────────
echo "[6/7] Checking ffmpeg..."
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg already available: $(command -v ffmpeg)"
else
    warn "ffmpeg not found — attempting install..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y ffmpeg 2>/dev/null && ok "ffmpeg installed via apt" || warn "apt install ffmpeg failed — install manually"
        else
            warn "Non-apt Linux — install ffmpeg manually: https://ffmpeg.org/download.html"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &>/dev/null; then
            brew install ffmpeg && ok "ffmpeg installed via brew" || warn "brew install ffmpeg failed"
        else
            warn "Homebrew not found — install from https://ffmpeg.org/download.html"
        fi
    else
        warn "Unknown OS — install ffmpeg manually"
    fi
fi

# ── 7. Node.js / frontend ────────────────────────────────────────────────────
echo "[7/7] Installing frontend dependencies..."
if ! command -v node &>/dev/null; then
    err "Node.js not found. Install from https://nodejs.org/ and re-run."
    exit 1
fi
ok "Node.js $(node --version) found"
(cd frontend && npm install --silent)
ok "Frontend dependencies installed"

# ── Verify CUDA ───────────────────────────────────────────────────────────────
if [ "$CUDA_DETECTED" = "yes" ]; then
    echo
    echo "Verifying CUDA install..."
    python -c "
import torch
if torch.cuda.is_available():
    d = torch.cuda.get_device_properties(0)
    print(f'  ✓ torch.cuda OK — {d.name} ({d.total_memory // 1024**3} GB)')
else:
    print('  ! torch installed but CUDA not available — check driver/wheel mismatch')
" || warn "CUDA verification failed (non-fatal)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "=== Setup Summary ==="
echo "  Venv        : $PROJECT_ROOT/.venv"
echo "  Python      : $PYVER"
if [ "$CUDA_DETECTED" = "yes" ]; then
    echo "  PyTorch     : CUDA ($TORCH_CUDA)"
else
    echo "  PyTorch     : CPU-only"
fi
echo "  COLMAP      : $(command -v colmap 2>/dev/null || echo 'NOT FOUND — install manually')"
echo "  ffmpeg      : $(command -v ffmpeg 2>/dev/null || echo 'NOT FOUND — install manually')"
echo "  gsplat      : $(python -c 'import gsplat; print(gsplat.__version__)' 2>/dev/null || echo 'not installed')"
echo
echo "Run 'python scripts/check_cuda.py' for a full environment report."
echo "Run './scripts/start.sh' to launch the app."
echo
