@echo off
title AquaGuard AI - Autonomous Inspection System
echo ============================================================
echo   AquaGuard AI - Underwater Infrastructure Inspection
echo ============================================================
echo.

set PYTHON_CMD=python

if exist "aquaguard_env\Scripts\python.exe" (
    set PYTHON_CMD=aquaguard_env\Scripts\python.exe
) else if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    set PYTHON_CMD=venv\Scripts\python.exe
)

echo Using Python interpreter: %PYTHON_CMD%
echo Launching unified server (FastAPI Backend + React UI)...
echo URL: http://localhost:8000
echo Docs: http://localhost:8000/docs
echo.
timeout /t 2 >nul
start "" http://localhost:8000
%PYTHON_CMD% -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
pause
