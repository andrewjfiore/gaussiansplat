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

LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')

echo
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
if [ -n "$LAN_IP" ]; then
    echo
    echo "LAN access (Quest 3):  http://$LAN_IP:3000"
    echo "QR / instructions:     http://$LAN_IP:3000/connect"
    echo
    echo "If Quest browser blocks HTTP, use HTTPS instead:"
    echo "  Stop this script, then run:"
    echo "    cd frontend && npm run dev:https"
fi
echo
echo "Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
