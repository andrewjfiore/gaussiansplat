# GaussianSplat Studio

All-in-one web app for creating 3D Gaussian Splats from video. Upload a video (or use a sample), and the app walks you through the full pipeline: frame extraction, COLMAP photogrammetry, splat training, and an in-browser 3D viewer.

## Quick Start (Linux / macOS)

```bash
chmod +x scripts/*.sh
./scripts/setup.sh     # auto-detects CUDA, installs everything
./scripts/start.sh
```

Then open http://localhost:3000

## Quick Start (Windows)

```
scripts\setup.bat
scripts\start.bat
```

## Requirements

- **Python 3.10+**
- **Node.js 18+**
- **NVIDIA GPU with CUDA** (recommended — CPU fallback installs but training is unusable without a GPU)
- **COLMAP** — `setup.sh` tries `apt install colmap` (Linux) or `brew install colmap` (macOS) automatically
- **ffmpeg** — `setup.sh` tries `apt install ffmpeg` / `brew install ffmpeg` automatically

`setup.sh` / `setup.bat` handle all Python deps including CUDA-enabled PyTorch and gsplat.

## Environment check

After setup, verify everything is working:

```bash
source .venv/bin/activate
python scripts/check_cuda.py
```

Sample output on a machine with an RTX 4060:

```
=== GaussianSplat Studio — Environment Check ===

  Python version : 3.11.9
  PyTorch version: 2.6.0+cu124
  CUDA available : yes
  CUDA version   : 12.4
  GPU name       : NVIDIA GeForce RTX 4060 Laptop GPU
  GPU memory     : 8.0 GB
  gsplat version : 1.5.3
  COLMAP         : yes  (/usr/bin/colmap)
  ffmpeg         : yes  (/usr/bin/ffmpeg)

  All checks passed — ready to train!
```

## Pipeline

1. **Video to Frames** — Extracts frames using ffmpeg
2. **Structure from Motion** — COLMAP feature extraction, matching, mapping, undistortion
3. **Gaussian Splat Training** — gsplat trains a 3D Gaussian Splat model
4. **3D Viewer** — View the result in-browser with GaussianSplats3D (WebGL)

## Architecture

- **Frontend**: Next.js 15 + React + TypeScript + Tailwind CSS
- **Backend**: Python FastAPI + SQLite
- **Real-time**: WebSocket for live log streaming and training metrics
- **Viewer**: GaussianSplats3D (Three.js-based WebGL viewer)

## Python dependency files

| File | Purpose |
|------|---------|
| `backend/requirements.txt` | FastAPI server + test deps |
| `backend/requirements-cuda.txt` | PyTorch (CUDA), gsplat, ML utils |

To install CUDA deps manually (e.g. to change the CUDA version):

```bash
# Change cu124 to match your driver — cu118, cu121, cu124, cu126
pip install torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu124
pip install gsplat
```

## Troubleshooting

**`torch.cuda.is_available()` returns `False` after setup**

- The CUDA wheel must match your driver's max CUDA version.  Run `nvidia-smi` and check the top-right "CUDA Version" field.  Then reinstall:
  ```bash
  pip install torch --extra-index-url https://download.pytorch.org/whl/cu124
  ```
- Make sure you activated the right venv: `source .venv/bin/activate` (project root, not `backend/.venv`).

**gsplat install fails with "No matching distribution"**

- gsplat requires a CUDA-enabled torch to be installed first.  Install torch with the correct `--extra-index-url` before running `pip install gsplat`.

**COLMAP not found**

- Linux: `sudo apt install colmap`
- macOS: `brew install colmap`
- Or download a binary from https://github.com/colmap/colmap/releases and place it in `tools/colmap/`.

**ffmpeg not found**

- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

## Development

Backend:
```bash
source .venv/bin/activate
uvicorn backend.app.main:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm run dev
```
