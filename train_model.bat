@echo off
title AquaGuard AI - YOLO26 Training Pipeline
echo ============================================================
echo   AquaGuard AI - YOLO26 Underwater Crack Detector Training
echo ============================================================
echo.
echo Dataset: 11,460 train images, 368 valid images (1 class: crack)
echo Architecture: YOLO26 Nano (yolo26n.pt)
echo Output Path:  models\best.pt
echo.
set /p EPOCHS="Enter number of epochs [press Enter for 10]: "
if "%EPOCHS%"=="" set EPOCHS=10

set /p BATCH="Enter batch size [press Enter for 16]: "
if "%BATCH%"=="" set BATCH=16

echo.
echo Starting training: %EPOCHS% epochs, batch size %BATCH%...
.\aquaguard_env\Scripts\python.exe backend\training\train.py --model yolo26n.pt --epochs %EPOCHS% --batch %BATCH% --device auto
echo.
echo Training complete. Checking weights...
dir models\best.pt
pause
