@echo off
REM Recovery Training Monitor - Tracks both Plan A (pure) and Plan B (progressive)
echo ============================================
echo Stage2.1 Recovery Training Monitor
echo ============================================
echo.

REM Check training processes
echo [1] Active Training Processes:
echo ----------------------------------------
tasklist /FI "IMAGENAME eq python.exe" /FO TABLE | findstr python.exe
if %errorlevel% == 0 (
    echo [OK] Training is running
) else (
    echo [WARNING] No Python training found!
)
echo.

REM Show Plan A (Pure Detection) status
echo [2] Plan A - Pure Detection Status:
echo ----------------------------------------
if exist "work_dirs\stage2_1_pure_detection" (
    echo Latest checkpoint:
    dir /O-D /B work_dirs\stage2_1_pure_detection\*.pth 2>nul | findstr /V ".py"
    echo.
    echo Current training progress:
    powershell -Command "$log = Get-ChildItem 'work_dirs\stage2_1_pure_detection\*\*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($log) { Write-Host \"Log: $($log.Directory.Name)\"; Get-Content $log.FullName | Select-String 'Epoch\(train\)' | Select-Object -Last 1 }"
    echo.
    echo Latest validation mAP:
    powershell -Command "$log = Get-ChildItem 'work_dirs\stage2_1_pure_detection\*\*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($log) { Get-Content $log.FullName | Select-String 'pascal_voc/mAP' | Select-Object -Last 1 }"
    echo.
    echo Loss trend (last 5 entries):
    powershell -Command "$log = Get-ChildItem 'work_dirs\stage2_1_pure_detection\*\*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($log) { Get-Content $log.FullName | Select-String 'loss_total:' | Select-Object -Last 5 | ForEach-Object { $_ -replace '.*\[(\d+)\]\[.*loss_total:\s*([\d.]+).*', 'Batch $1: loss=$2' } }"
) else (
    echo [Not started yet]
)
echo.

REM Show Plan B (Progressive MACL) status
echo [3] Plan B - Progressive MACL Status:
echo ----------------------------------------
if exist "work_dirs\stage2_1_progressive_macl" (
    echo Latest checkpoint:
    dir /O-D /B work_dirs\stage2_1_progressive_macl\*.pth 2>nul | findstr /V ".py"
    echo.
    echo Latest mAP:
    powershell -Command "$log = Get-ChildItem 'work_dirs\stage2_1_progressive_macl\*\*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($log) { Get-Content $log.FullName | Select-String 'pascal_voc/mAP' | Select-Object -Last 1 }"
    echo.
    echo Lambda1 warmup progress:
    powershell -Command "$log = Get-ChildItem 'work_dirs\stage2_1_progressive_macl\*\*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($log) { Get-Content $log.FullName | Select-String 'LambdaWarmupHook' | Select-Object -Last 3 }"
) else (
    echo [Not started yet]
)
echo.

REM GPU Status
echo [4] GPU Memory & Temperature:
echo ----------------------------------------
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo.

REM Early Stop Warnings
echo [5] Early Stop Alerts:
echo ----------------------------------------
powershell -Command "$logs = Get-ChildItem 'work_dirs\stage2_1_*\*\*.log' -ErrorAction SilentlyContinue; foreach ($log in $logs) { $warnings = Get-Content $log.FullName | Select-String 'EarlyStopHook|bad_epoch' | Select-Object -Last 1; if ($warnings) { Write-Host \"$($log.Directory.Name): $warnings\" } }"
echo.

echo ============================================
echo [Baseline: Stage1 epoch_21 mAP=0.6288]
echo [Failed: Stage2.1 original epoch_3 mAP=0.5265]
echo [Target: Recovery mAP >= 0.63]
echo ============================================
echo Press any key to refresh...
pause >nul
cls
goto :eof
