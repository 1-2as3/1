@echo off
REM Plan C 训练 + TensorBoard 一键启动脚本
REM =========================================

echo ========================================
echo Plan C 训练环境启动
echo ========================================
echo.

REM 检查TensorBoard是否已安装
python -c "import tensorboard" 2>nul
if errorlevel 1 (
    echo [警告] TensorBoard未安装,正在安装...
    pip install tensorboard
)

echo [1/3] 启动TensorBoard服务器...
start "TensorBoard" cmd /k "tensorboard --logdir=work_dirs --port=6006 & echo. & echo TensorBoard运行中: http://localhost:6006 & echo 关闭此窗口将停止TensorBoard & echo."

timeout /t 3 /nobreak >nul

echo [2/3] 等待TensorBoard启动...
timeout /t 5 /nobreak >nul

echo [3/3] 启动训练...
echo.
echo ========================================
echo 监控指南:
echo   - TensorBoard: http://localhost:6006
echo   - 日志位置: work_dirs\stage2_2_planC_dualmodality_macl\
echo   - 关键指标: loss_macl, mAP, grad_norm
echo ========================================
echo.

REM 启动训练
python tools\train.py configs\llvip\stage2_2_planC_dualmodality_macl.py

pause
