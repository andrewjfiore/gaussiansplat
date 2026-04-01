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
echo For Quest 3 access, open: http://YOUR_LAN_IP:3000/connect
echo (Run 'ipconfig' to find your LAN IP under Wi-Fi adapter)
echo.
echo If Quest browser blocks HTTP, use HTTPS:
echo   Stop servers, then run in frontend\: npm run dev:https
echo.
echo Close the terminal windows to stop the servers.
