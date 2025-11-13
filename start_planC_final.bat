@echo off
REM Plan C Final Safe Start
REM ========================

echo ========================================
echo Plan C Training - Final Safe Start
echo ========================================
echo.

REM Set UTF-8 encoding
chcp 65001 >nul 2>&1

REM Use correct Python interpreter
set PYTHON=C:\Users\Xinyu\.conda\envs\py311\python.exe

echo [INFO] Using Python: %PYTHON%
%PYTHON% --version
echo.

echo [INFO] Config: configs/llvip/stage2_2_planC_dualmodality_macl_clean.py
echo [INFO] Key settings:
echo   - num_workers: 0 (fixed deadlock)
echo   - batch_size: 2
echo   - return_modality_pair: True
echo   - lambda1: 0.01 (conservative MACL)
echo.

echo [START] Beginning training...
echo ========================================
echo.

%PYTHON% tools/train.py configs/llvip/stage2_2_planC_dualmodality_macl_clean.py

echo.
echo ========================================
echo Training finished or interrupted
echo ========================================
echo.
pause
