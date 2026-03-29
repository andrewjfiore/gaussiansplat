@echo off
setlocal EnableDelayedExpansion
:: GaussianSplat Studio — Windows setup script
:: Creates .venv at project root, detects CUDA, installs all dependencies.

echo.
echo === GaussianSplat Studio Setup ===
echo.

:: Resolve project root (one level up from scripts\)
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
pushd "%PROJECT_ROOT%"

:: ── 1. Python ──────────────────────────────────────────────────────────────
echo [1/7] Checking Python...
python --version 2>nul
if errorlevel 1 (
    echo   ERROR: Python 3.10+ not found. Install from https://www.python.org/downloads/
    exit /b 1
)
echo   OK: Python found
echo.

:: ── 2. Virtual environment ──────────────────────────────────────────────────
echo [2/7] Setting up virtual environment at .venv ...
if not exist ".venv" (
    python -m venv .venv
    echo   OK: Created .venv
) else (
    echo   OK: .venv already exists
)
call .venv\Scripts\activate.bat
echo.

:: ── 3. CUDA detection ───────────────────────────────────────────────────────
echo [3/7] Detecting GPU / CUDA...
set "CUDA_DETECTED=no"
set "TORCH_CUDA=cpu"
set "CUDA_TORCH_URL=https://download.pytorch.org/whl/cpu"

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    :: Parse CUDA version from nvidia-smi output
    for /f "tokens=*" %%A in ('nvidia-smi 2^>nul ^| findstr /i "CUDA Version"') do (
        set "SMILINE=%%A"
    )
    :: Extract the version number after "CUDA Version: "
    for /f "tokens=3 delims=: " %%V in ("!SMILINE!") do (
        set "CUDA_VER=%%V"
    )
    if defined CUDA_VER (
        set "CUDA_DETECTED=yes"
        :: Default to cu124; refine based on major version detected in version string
        set "TORCH_CUDA=cu124"
        echo !CUDA_VER! | findstr /b "12.6" >nul && set "TORCH_CUDA=cu126"
        echo !CUDA_VER! | findstr /b "12.4" >nul && set "TORCH_CUDA=cu124"
        echo !CUDA_VER! | findstr /b "12.1" >nul && set "TORCH_CUDA=cu121"
        echo !CUDA_VER! | findstr /b "12.0" >nul && set "TORCH_CUDA=cu121"
        echo !CUDA_VER! | findstr /b "11." >nul && set "TORCH_CUDA=cu118"
        set "CUDA_TORCH_URL=https://download.pytorch.org/whl/!TORCH_CUDA!"
        echo   OK: NVIDIA GPU detected -- CUDA !CUDA_VER! -- will install torch+!TORCH_CUDA!
    ) else (
        echo   WARNING: nvidia-smi found but couldn't read CUDA version; falling back to CPU torch
    )
) else (
    echo   WARNING: nvidia-smi not found -- installing CPU-only PyTorch
)
echo.

:: ── 4. Python dependencies ──────────────────────────────────────────────────
echo [4/7] Installing Python dependencies...
pip install --upgrade pip --quiet

pip install -r backend\requirements.txt --quiet
if errorlevel 1 ( echo   ERROR: failed to install base deps & exit /b 1 )
echo   OK: Base dependencies installed

if "!CUDA_DETECTED!"=="yes" (
    pip install torch torchvision torchaudio --extra-index-url "!CUDA_TORCH_URL!" --quiet
    if errorlevel 1 ( echo   WARNING: CUDA torch install failed )
    echo   OK: PyTorch (CUDA) installed
    pip install gsplat --quiet
    if errorlevel 1 ( echo   WARNING: gsplat install failed )
    echo   OK: gsplat installed
    pip install numpy Pillow tqdm --quiet
    echo   OK: numpy / Pillow / tqdm installed
) else (
    pip install torch torchvision torchaudio --quiet
    echo   WARNING: PyTorch installed (CPU-only)
    pip install numpy Pillow tqdm --quiet
)
echo.

:: ── 5. COLMAP ────────────────────────────────────────────────────────────────
echo [5/7] Checking COLMAP...
where colmap >nul 2>&1
if not errorlevel 1 (
    echo   OK: COLMAP found in PATH
) else (
    if exist "tools\colmap\COLMAP.bat" (
        echo   OK: COLMAP found in tools\colmap\
    ) else (
        echo   WARNING: COLMAP not found.
        echo     Download from https://github.com/colmap/colmap/releases
        echo     and place in tools\colmap\ or add to PATH.
    )
)
echo.

:: ── 6. ffmpeg ─────────────────────────────────────────────────────────────
echo [6/7] Checking ffmpeg...
where ffmpeg >nul 2>&1
if not errorlevel 1 (
    echo   OK: ffmpeg found in PATH
) else (
    if exist "tools\ffmpeg\ffmpeg.exe" (
        echo   OK: ffmpeg found in tools\ffmpeg\
    ) else (
        echo   WARNING: ffmpeg not found.
        echo     Download from https://ffmpeg.org/download.html
        echo     and place in tools\ffmpeg\ or add to PATH.
    )
)
echo.

:: ── 7. Node.js / frontend ──────────────────────────────────────────────────
echo [7/7] Installing frontend dependencies...
node --version 2>nul
if errorlevel 1 (
    echo   ERROR: Node.js not found. Install from https://nodejs.org/
    exit /b 1
)
echo   OK: Node.js found
cd frontend
call npm install --silent
cd ..
echo   OK: Frontend dependencies installed
echo.

:: ── Verify CUDA ──────────────────────────────────────────────────────────────
if "!CUDA_DETECTED!"=="yes" (
    echo Verifying CUDA install...
    python -c "import torch; d=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; print(f'  OK: {d.name} ({d.total_memory//1024**3} GB)' if d else '  WARNING: CUDA not available after install')"
    echo.
)

:: ── Summary ──────────────────────────────────────────────────────────────────
echo === Setup Summary ===
echo   Venv   : %CD%\.venv
if "!CUDA_DETECTED!"=="yes" (
    echo   PyTorch: CUDA (!TORCH_CUDA!)
    for /f %%G in ('python -c "import gsplat; print(gsplat.__version__)" 2^>nul') do echo   gsplat : %%G
) else (
    echo   PyTorch: CPU-only
    echo   gsplat : not installed
)
echo.
echo Run 'python scripts\check_cuda.py' for a full environment report.
echo Run 'scripts\start.bat' to launch the app.
echo.

popd
endlocal
