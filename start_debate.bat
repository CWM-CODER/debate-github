@echo off
echo Starting Clash of Minds AI Debate System...

echo.
echo [1/2] Starting FastAPI Backend via WSL (Linux)...
start cmd /k "wsl bash /mnt/d/Debate/start_backend_wsl.sh"

echo [2/2] Starting Frontend HTTP Server on port 8080...
start cmd /k "cd /d d:\Debate\frontend && python -m http.server 8080"

echo.
echo Servers are starting up in new windows.
echo Please wait 30-60 seconds for the AI models to load.
echo.
echo Opening browser to http://localhost:8080...
timeout /t 10 > nul
start http://localhost:8080

echo Done! You can close this window.
