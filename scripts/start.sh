#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

VENV="$PROJECT_ROOT/.venv"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: .venv not found. Run './scripts/setup.sh' first."
    exit 1
fi

echo "=== Starting GaussianSplat Studio ==="

# Start backend
source "$VENV/bin/activate"
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 2

# Start frontend
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
