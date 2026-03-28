@echo off
echo === GaussianSplat Studio Setup ===
echo.

echo [1/4] Checking Python...
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://www.python.org/downloads/
    exit /b 1
)

echo [2/4] Setting up Python virtual environment...
if not exist "backend\.venv" (
    python -m venv backend\.venv
)
call backend\.venv\Scripts\activate.bat
pip install -r backend\requirements.txt

echo [3/4] Checking Node.js...
node --version 2>nul
if errorlevel 1 (
    echo ERROR: Node.js not found. Install from https://nodejs.org/
    exit /b 1
)

echo [4/4] Installing frontend dependencies...
cd frontend
call npm install
cd ..

echo.
echo === Setup complete! ===
echo Run 'scripts\start.bat' to launch the app.
