# GaussianSplat Studio

All-in-one web app for creating 3D Gaussian Splats from video. Upload a video (or use a sample), and the app walks you through the full pipeline: frame extraction, COLMAP photogrammetry, splat training, and an in-browser 3D viewer.

## Quick Start (Windows)

```
scripts\setup.bat
scripts\start.bat
```

Then open http://localhost:3000

## Quick Start (Linux)

```
chmod +x scripts/*.sh
./scripts/setup.sh
./scripts/start.sh
```

## Requirements

- **Python 3.10+**
- **Node.js 18+**
- **NVIDIA GPU with CUDA** (required for training)
- **ffmpeg** (auto-downloaded on Windows, or `apt install ffmpeg` on Linux)
- **COLMAP** (auto-downloaded on Windows, or `apt install colmap` on Linux)

For training, you also need PyTorch with CUDA and gsplat:
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install gsplat
```

## Pipeline

1. **Video to Frames** - Extracts frames using ffmpeg
2. **Structure from Motion** - COLMAP feature extraction, matching, mapping, undistortion
3. **Gaussian Splat Training** - gsplat trains a 3D Gaussian Splat model
4. **3D Viewer** - View the result in-browser with GaussianSplats3D (WebGL)

## Architecture

- **Frontend**: Next.js 15 + React + TypeScript + Tailwind CSS
- **Backend**: Python FastAPI + SQLite
- **Real-time**: WebSocket for live log streaming and training metrics
- **Viewer**: GaussianSplats3D (Three.js-based WebGL viewer)

## Development

Backend:
```
cd backend
../.venv/Scripts/activate  # or source ../.venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Frontend:
```
cd frontend
npm run dev
```
