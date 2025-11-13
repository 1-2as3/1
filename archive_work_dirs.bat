@echo off
REM work_dirs 归档脚本

set ARCHIVE_DIR=work_dirs_archive_20251112
mkdir %ARCHIVE_DIR% 2>nul
echo 归档到: %ARCHIVE_DIR%
echo.
echo 移动: stage2_kaist_full_v1
move "work_dirs\stage2_kaist_full_v1" %ARCHIVE_DIR%\ 2>nul

echo 移动: stage2_kaist_full_conservative_remote
move "work_dirs\stage2_kaist_full_conservative_remote" %ARCHIVE_DIR%\ 2>nul

echo 移动: stage2_kaist_full_conservative_test
move "work_dirs\stage2_kaist_full_conservative_test" %ARCHIVE_DIR%\ 2>nul

echo 移动: tsne_vis
move "work_dirs\tsne_vis" %ARCHIVE_DIR%\ 2>nul

echo.
echo 归档完成!
pause