@echo off
REM Live Monitor for Plan A (Pure Detection) Training
REM Usage: Run this in a separate terminal while training is running
REM Press Ctrl+C to exit

:loop
cls
echo ================================================
echo Plan A - Pure Detection Training (LIVE Monitor)
echo ================================================
echo Time: %date% %time%
echo.

REM Find the latest log file
for /f "delims=" %%a in ('powershell -Command "Get-ChildItem 'work_dirs\stage2_1_pure_detection\*\*.log' | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do set LATEST_LOG=%%a

if defined LATEST_LOG (
    echo [LOG FILE]
    echo %LATEST_LOG%
    echo.
    
    echo [CURRENT PROGRESS]
    echo ------------------------------------------------
    powershell -Command "Get-Content '%LATEST_LOG%' | Select-String 'Epoch\(train\)' | Select-Object -Last 1"
    echo.
    
    echo [LOSS TREND - Last 10 batches]
    echo ------------------------------------------------
    powershell -Command "Get-Content '%LATEST_LOG%' | Select-String 'loss_total:' | Select-Object -Last 10 | ForEach-Object { if ($_ -match 'Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/\s*(\d+)\].*loss_total:\s*([\d.]+)') { $epoch=$matches[1]; $batch=$matches[2]; $total=$matches[3]; $loss=$matches[4]; $pct=[math]::Round(($batch/$total)*100, 1); Write-Host \"  Epoch $epoch [$batch/$total = ${pct}%%] loss_total=$loss\" } }"
    echo.
    
    echo [LATEST VALIDATION RESULTS]
    echo ------------------------------------------------
    powershell -Command "Get-Content '%LATEST_LOG%' | Select-String 'pascal_voc' | Select-Object -Last 5"
    echo.
    
    echo [GPU STATUS]
    echo ------------------------------------------------
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    echo.
    
    echo [CHECKPOINT FILES]
    echo ------------------------------------------------
    dir /O-D /B work_dirs\stage2_1_pure_detection\*.pth 2>nul | findstr /V ".py"
    
) else (
    echo [ERROR] No log file found in work_dirs\stage2_1_pure_detection\
)

echo.
echo ================================================
echo [Baseline] Stage1 epoch_21: mAP=0.6288
echo [Failed]   Stage2.1 epoch_3: mAP=0.5265
echo [Target]   Recovery: mAP >= 0.63
echo ================================================
echo.
echo Refreshing in 30 seconds... (Press Ctrl+C to exit)
timeout /t 30 /nobreak >nul
goto loop
