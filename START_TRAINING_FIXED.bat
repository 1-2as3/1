@echo off
echo ===============================================
echo  Stage1 LLVIP Training - NaN Fixed Version
echo ===============================================
echo.
echo Modifications applied:
echo  1. MACL Head: clamp(-50,50), nan_to_num, epsilon protection
echo  2. TwoStage: NaN input检查, 空样本返回零损失
echo  3. RoI Head: pos_bboxes空检测
echo  4. Config: AdamW lr=1e-4, GradClip=5.0, GroupNorm
echo  5. Debug: 每100轮打印MACL sim统计
echo.
echo Training Parameters:
echo  - Epochs: 1
echo  - Samples: 9,620 (train) + 2,405 (val)
echo  - Batch Size: 4
echo  - Optimizer: AdamW (lr=1e-4, wd=1e-4)
echo  - Normalization: GroupNorm (32 groups)
echo  - Gradient Clipping: max_norm=5.0
echo.
echo ⚠️ IMPORTANT: DO NOT press Ctrl+C!
echo Let the training run completely (~15-20 minutes)
echo ===============================================
echo.
pause

C:\Users\Xinyu\.conda\envs\py311\python.exe tools\train.py configs\llvip\stage1_llvip_pretrain.py --cfg-options train_cfg.max_epochs=1

echo.
echo ===============================================
echo Training completed! Check logs in:
echo work_dirs\stage1_llvip_pretrain\[timestamp]\
echo ===============================================
pause
