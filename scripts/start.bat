@echo off
echo === Starting GaussianSplat Studio ===
echo.

:: Start backend
start "GS-Backend" cmd /k "cd /d %~dp0.. && call .venv\Scripts\activate.bat && python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"

:: Wait a moment for backend to start
timeout /t 3 /nobreak > nul

:: Start frontend
start "GS-Frontend" cmd /k "cd /d %~dp0..\frontend && npm run dev"

echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Close the terminal windows to stop the servers.
