@echo off
REM Plan C Safe Start - UTF-8 Encoding Fix
REM ==========================================

REM Set UTF-8 encoding for Windows console
chcp 65001 >nul 2>&1

echo ========================================
echo Plan C Training - Safe Start
echo ========================================
echo.
echo [Fix] Console encoding set to UTF-8
echo [Fix] Using clean config without Unicode chars
echo.

REM Check Python environment
python --version
echo.

REM Start training with clean config
echo [Starting] Training with clean configuration...
echo Config: configs/llvip/stage2_2_planC_dualmodality_macl_clean.py
echo.

python tools/train.py configs/llvip/stage2_2_planC_dualmodality_macl_clean.py

echo.
echo ========================================
echo Training finished or interrupted
echo ========================================
pause
