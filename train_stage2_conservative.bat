@echo off
REM Stage2 Conservative Training Script
REM This script runs in the background and logs all output

echo ============================================
echo Stage2 KAIST Full Training - Conservative
echo Start Time: %date% %time%
echo ============================================

REM Activate conda environment
call C:\Users\Xinyu\.conda\envs\py311\Scripts\activate.bat

REM Change to working directory
cd /d C:\Users\Xinyu\mmdetection

REM Create log directory
if not exist "logs" mkdir logs

REM Generate timestamp for log filename
set timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set timestamp=%timestamp: =0%

REM Run training with output redirection
echo Starting training at %date% %time% > logs\stage2_conservative_%timestamp%.log 
python -u tools\train.py ^
    configs\llvip\stage2_kaist_full_conservative.py ^
    --work-dir work_dirs\stage2_kaist_full_conservative ^
    2>&1 | tee -a logs\stage2_conservative_%timestamp%.log

echo Training finished at %date% %time%
echo Check logs in: logs\stage2_conservative_%timestamp%.log
pause
