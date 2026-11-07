@echo off
title AquaGuard AI - Autonomous Inspection System
echo ============================================================
echo   AquaGuard AI - Underwater Infrastructure Inspection
echo ============================================================
echo.
echo Launching unified server (FastAPI Backend + React UI)...
echo URL: http://localhost:8000
echo Docs: http://localhost:8000/docs
echo.
timeout /t 2 >nul
start "" http://localhost:8000
.\aquaguard_env\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
pause
