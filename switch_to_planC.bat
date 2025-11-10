@echo off
REM Emergency Switch to Plan C - Freeze Domain Alignment
echo ============================================
echo EMERGENCY: Switching to Plan C
echo Freeze Domain Alignment Strategy
echo ============================================
echo.

echo [WARNING] This will STOP current training and switch to Plan C
echo.
echo Plan C Strategy:
echo - Freeze domain alignment (domain_weight=0.0)
echo - Focus on detection performance optimization
echo - Train 6 additional epochs from current checkpoint
echo.

set /p confirm="Are you sure you want to switch to Plan C? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Operation cancelled.
    pause
    exit /b
)

echo.
echo [1] Finding latest checkpoint...
set workdir=work_dirs\stage2_kaist_full_conservative

REM Find best checkpoint or latest epoch
if exist "%workdir%\best_pascal_voc_mAP_*.pth" (
    for /f "tokens=*" %%f in ('dir /B /O-D "%workdir%\best_pascal_voc_mAP_*.pth" 2^>nul') do (
        set "checkpoint=%%f"
        goto :found_checkpoint
    )
)

REM If no best checkpoint, use latest epoch
for /f "tokens=*" %%f in ('dir /B /O-D "%workdir%\epoch_*.pth" 2^>nul') do (
    set "checkpoint=%%f"
    goto :found_checkpoint
)

echo [ERROR] No checkpoint found in %workdir%
pause
exit /b 1

:found_checkpoint
echo Found checkpoint: %checkpoint%
echo.

REM Update config file with correct checkpoint path
echo [2] Updating Plan C config with checkpoint path...
powershell -Command "(Get-Content configs\llvip\stage2_planC_freeze_domain.py) -replace \"load_from = '.*'\", \"load_from = '%workdir%\%checkpoint%'\" | Set-Content configs\llvip\stage2_planC_freeze_domain.py"

echo.
echo [3] Stopping current training process (if any)...
taskkill /IM python.exe /F 2>nul
timeout /t 3 /nobreak >nul

echo.
echo [4] Starting Plan C training...
call C:\Users\Xinyu\.conda\envs\py311\Scripts\activate.bat
cd /d C:\Users\Xinyu\mmdetection

REM Create log directory
if not exist "logs" mkdir logs

REM Generate timestamp
set timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set timestamp=%timestamp: =0%

echo.
echo Starting Plan C training from checkpoint: %checkpoint%
echo Log file: logs\stage2_planC_%timestamp%.log
echo.

start "Stage2 Plan C Training" cmd /c "python -u tools\train.py configs\llvip\stage2_planC_freeze_domain.py --work-dir work_dirs\stage2_planC_freeze_domain 2>&1 | tee logs\stage2_planC_%timestamp%.log & pause"

echo.
echo ============================================
echo Plan C training started in new window
echo Monitor with: monitor_training.bat
echo ============================================
pause
