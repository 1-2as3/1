@echo off
REM Epoch Checkpoint Analyzer for Decision Making
echo ============================================
echo Stage2 Training Decision Point Analyzer
echo ============================================
echo.

set workdir=work_dirs\stage2_kaist_full_conservative

REM Extract mAP from checkpoint files
echo [EPOCH PERFORMANCE SUMMARY]
echo ============================================
echo.

for /f "tokens=*" %%f in ('dir /B /O:N %workdir%\epoch_*.pth 2^>nul') do (
    set "filename=%%f"
    setlocal enabledelayedexpansion
    set "epoch=!filename:~6,1!"
    if "!filename:~7,1!" neq "." set "epoch=!epoch!!filename:~7,1!"
    
    REM Search for mAP in log file
    for /f "tokens=*" %%m in ('powershell -Command "Get-Content -Path (Get-ChildItem logs\stage2_conservative_*.log ^| Sort-Object LastWriteTime -Descending ^| Select-Object -First 1).FullName ^| Select-String 'Epoch\(val\) \[!epoch!\].*mAP:' ^| Select-Object -Last 1"') do (
        echo Epoch !epoch!: %%m
    )
    endlocal
)

echo.
echo ============================================
echo [DECISION GUIDELINES]
echo ============================================
echo.
echo ** EPOCH 4 Decision Point (domain_weight reaches 0.08) **
echo   - If mAP >= 0.60: Continue training [GREEN]
echo   - If mAP <  0.55: STOP and switch to Plan C [RED]
echo   - If mAP 0.55-0.60: Borderline, monitor closely [YELLOW]
echo.
echo ** EPOCH 8 Mid-point Assessment **
echo   - If mAP >= 0.65: On track for target [GREEN]
echo   - If mAP 0.60-0.65: Acceptable progress [YELLOW]
echo   - If mAP <  0.60: Consider intervention [RED]
echo.
echo ** EPOCH 12 Final Target **
echo   - Best case: mAP 0.70-0.72 [EXCELLENT]
echo   - Acceptable: mAP 0.68-0.70 [GOOD]
echo   - Needs work: mAP <  0.68 [RETRY NEEDED]
echo.
echo ============================================

REM Show current best checkpoint
echo [BEST CHECKPOINT]
echo ----------------------------------------
if exist "%workdir%\best_pascal_voc_mAP_*.pth" (
    dir /O-D /B %workdir%\best_pascal_voc_mAP_*.pth | findstr /V ".py"
) else (
    echo No best checkpoint found yet
)
echo.

REM Extract domain_weight progression from log
echo [DOMAIN WEIGHT WARMUP PROGRESSION]
echo ----------------------------------------
powershell -Command "Get-Content -Path (Get-ChildItem logs\stage2_conservative_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName | Select-String 'DomainWeightWarmupHook'"
echo.

echo ============================================
pause
