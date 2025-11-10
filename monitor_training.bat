@echo off
REM Quick status check for Stage2 training
echo ============================================
echo Stage2 Training Status Monitor
echo ============================================
echo.

REM Check if training process is running
echo [1] Checking Python training processes...
tasklist /FI "IMAGENAME eq python.exe" /FO TABLE | findstr python.exe
if %errorlevel% == 0 (
    echo [OK] Training process is running
) else (
    echo [WARNING] No Python training process found!
)
echo.

REM Show latest log entries
echo [2] Latest training log (last 20 lines):
echo ----------------------------------------
powershell -Command "Get-Content -Path (Get-ChildItem logs\stage2_conservative_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName -Tail 20"
echo.

REM Show GPU status
echo [3] GPU Memory Usage:
echo ----------------------------------------
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
echo.

REM Show checkpoint files
echo [4] Latest checkpoints:
echo ----------------------------------------
dir /O-D /B work_dirs\stage2_kaist_full_conservative\*.pth 2>nul | findstr /V /C:"stage2_kaist_full_conservative.py"
echo.

REM Show current epoch from log
echo [5] Training Progress:
echo ----------------------------------------
powershell -Command "Get-Content -Path (Get-ChildItem logs\stage2_conservative_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName | Select-String 'Epoch\(val\)' | Select-Object -Last 1"
echo.

echo ============================================
echo Press any key to refresh, Ctrl+C to exit
pause >nul
cls
goto :eof
