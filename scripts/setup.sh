#!/bin/bash
set -e

echo "=== GaussianSplat Studio Setup ==="
echo

echo "[1/4] Checking Python..."
python3 --version || { echo "ERROR: Python3 not found. Install it first."; exit 1; }

echo "[2/4] Setting up Python virtual environment..."
if [ ! -d "backend/.venv" ]; then
    python3 -m venv backend/.venv
fi
source backend/.venv/bin/activate
pip install -r backend/requirements.txt

echo "[3/4] Checking Node.js..."
node --version || { echo "ERROR: Node.js not found. Install from https://nodejs.org/"; exit 1; }

echo "[4/4] Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo
echo "=== Setup complete! ==="
echo "Run './scripts/start.sh' to launch the app."
