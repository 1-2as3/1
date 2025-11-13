@echo off
REM Plan C Linux Package Creator
REM ============================================================================
REM This script creates a compressed archive for easy transfer to Linux
REM ============================================================================

echo ================================================
echo   Creating Plan C Linux Deployment Package
echo ================================================
echo.

REM Check if 7-Zip is available
where 7z >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: 7-Zip not found. Will create uncompressed zip instead.
    echo          Install 7-Zip for better compression: https://www.7-zip.org/
    echo.
    set USE_POWERSHELL=1
) else (
    set USE_POWERSHELL=0
)

REM Set package name with timestamp
set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set PACKAGE_NAME=planC_linux_package_%TIMESTAMP%

echo Creating package: %PACKAGE_NAME%
echo.

REM Create temporary directory
if exist temp_package rmdir /s /q temp_package
mkdir temp_package\%PACKAGE_NAME%

REM Copy files
echo [1/5] Copying configuration files...
copy config_planC_linux.py temp_package\%PACKAGE_NAME%\ >nul
copy requirements_linux.txt temp_package\%PACKAGE_NAME%\ >nul
copy README_DEPLOYMENT.md temp_package\%PACKAGE_NAME%\ >nul

echo [2/5] Copying training scripts...
copy train_planC.sh temp_package\%PACKAGE_NAME%\ >nul
copy setup_planC.sh temp_package\%PACKAGE_NAME%\ >nul
copy test_dual_modality.py temp_package\%PACKAGE_NAME%\ >nul

echo [3/5] Creating quick reference...
copy QUICK_REFERENCE.txt temp_package\%PACKAGE_NAME%\ >nul 2>nul || echo No QUICK_REFERENCE.txt found, skipping...

echo [4/5] Compressing package...
cd temp_package

if %USE_POWERSHELL%==1 (
    powershell -Command "Compress-Archive -Path '%PACKAGE_NAME%' -DestinationPath '../%PACKAGE_NAME%.zip' -Force"
    echo Created: %PACKAGE_NAME%.zip
) else (
    7z a -ttar -so %PACKAGE_NAME% | 7z a -si ../%PACKAGE_NAME%.tar.gz >nul
    echo Created: %PACKAGE_NAME%.tar.gz
)

cd ..

echo [5/5] Cleaning up...
rmdir /s /q temp_package

echo.
echo ================================================
echo   Package Created Successfully!
echo ================================================
echo.

if %USE_POWERSHELL%==1 (
    echo Package: %PACKAGE_NAME%.zip
    for %%A in (%PACKAGE_NAME%.zip) do echo Size: %%~zA bytes
) else (
    echo Package: %PACKAGE_NAME%.tar.gz
    for %%A in (%PACKAGE_NAME%.tar.gz) do echo Size: %%~zA bytes
)

echo.
echo Contents:
echo   - config_planC_linux.py      [Training configuration]
echo   - train_planC.sh             [Automated training script]
echo   - setup_planC.sh             [Auto-setup script]
echo   - test_dual_modality.py      [Smoke test]
echo   - requirements_linux.txt     [Python dependencies]
echo   - README_DEPLOYMENT.md       [Full deployment guide]
echo   - QUICK_REFERENCE.txt        [Quick command reference]
echo.
echo Transfer Instructions:
echo   1. Upload package via 向日葵 file transfer
echo   2. Extract on Linux: tar -xzf %PACKAGE_NAME%.tar.gz
echo   3. Run: bash %PACKAGE_NAME%/setup_planC.sh
echo.
echo ================================================
pause
