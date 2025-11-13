@echo off
chcp 65001 >nul
echo ============================================
echo Plan C Training - Retry Attempt
echo ============================================
echo Time: %date% %time%
echo.

REM 激活conda环境
echo [1/5] Activating conda environment...
call conda activate mmdet
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    pause
    exit /b 1
)
echo OK - Environment activated
echo.

REM 验证CUDA可用性
echo [2/5] Checking CUDA availability...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
if errorlevel 1 (
    echo ERROR: CUDA check failed
    pause
    exit /b 1
)
echo.

REM 验证MMEngine和配置
echo [3/5] Verifying MMEngine and config...
python -c "from mmengine import Config; cfg = Config.fromfile('configs/llvip/stage2_2_planC_dualmodality_macl_clean.py'); print(f'Config loaded: {cfg.work_dir}')"
if errorlevel 1 (
    echo ERROR: Config verification failed
    pause
    exit /b 1
)
echo OK - Config is valid
echo.

REM 清理之前的日志
echo [4/5] Cleaning old logs...
if exist work_dirs\stage2_2_planC_dualmodality_macl\*.log.json (
    del /q work_dirs\stage2_2_planC_dualmodality_macl\*.log.json
    echo Old logs cleaned
) else (
    echo No old logs to clean
)
echo.

REM 启动训练
echo [5/5] Starting training...
echo Command: python tools/train.py configs/llvip/stage2_2_planC_dualmodality_macl_clean.py
echo Working Directory: %cd%
echo.
echo Training Output:
echo ============================================

python tools/train.py configs/llvip/stage2_2_planC_dualmodality_macl_clean.py 2>&1 | tee training_output.log

echo.
echo ============================================
echo Training finished or interrupted
echo Time: %date% %time%
echo.
echo Check training_output.log for full output
pause
