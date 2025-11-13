@echo off
REM work_dirs 清理脚本
REM 生成时间: 2025-11-12 19:31:05

echo ========================================
echo work_dirs 清理脚本
echo ========================================
echo.
echo 将删除以下目录:
echo   - stage2_kaist_finetune_sanity (2046.0 MB)
echo   - stage2_1_planB_macl_rescue (1603.4 MB)
echo   - stage2_test_validation (0.0 MB)
echo   - stage2_final_validation (0.0 MB)
echo   - stage2_2_kaist_contrastive (0.0 MB)
echo   - _archive_20251110 (0.0 MB)
echo   - stage2_1_emergency (0.0 MB)
echo.
pause
echo.
echo 开始清理...

echo 删除: stage2_kaist_finetune_sanity
rd /s /q "work_dirs\stage2_kaist_finetune_sanity" 2>nul

echo 删除: stage2_1_planB_macl_rescue
rd /s /q "work_dirs\stage2_1_planB_macl_rescue" 2>nul

echo 删除: stage2_test_validation
rd /s /q "work_dirs\stage2_test_validation" 2>nul

echo 删除: stage2_final_validation
rd /s /q "work_dirs\stage2_final_validation" 2>nul

echo 删除: stage2_2_kaist_contrastive
rd /s /q "work_dirs\stage2_2_kaist_contrastive" 2>nul

echo 删除: _archive_20251110
rd /s /q "work_dirs\_archive_20251110" 2>nul

echo 删除: stage2_1_emergency
rd /s /q "work_dirs\stage2_1_emergency" 2>nul

echo.
echo 清理完成! 释放空间约 3649.5 MB
echo.
pause