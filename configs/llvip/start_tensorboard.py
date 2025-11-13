"""
Plan C TensorBoard 监控脚本
============================
启动TensorBoard并提供关键指标监控指南
"""

import subprocess
import sys
import os
import webbrowser
import time

def start_tensorboard(logdir='work_dirs/stage2_2_planC_dualmodality_macl', port=6006):
    """启动TensorBoard服务"""
    
    print("="*70)
    print("Plan C TensorBoard 监控启动")
    print("="*70)
    
    # 检查目录是否存在
    if not os.path.exists(logdir):
        print(f"❌ 错误: 日志目录不存在: {logdir}")
        print("   请先开始训练,生成训练日志后再启动TensorBoard")
        return False
    
    # 检查是否有tf-events文件
    has_events = False
    for root, dirs, files in os.walk(logdir):
        if any(f.startswith('events.out.tfevents') for f in files):
            has_events = True
            break
    
    if not has_events:
        print(f"⚠️ 警告: {logdir} 中未找到TensorBoard事件文件")
        print("   TensorBoard将启动但可能显示为空")
        print("   开始训练后,刷新浏览器即可看到数据")
        print()
    
    url = f"http://localhost:{port}"
    
    print(f"📊 启动TensorBoard服务...")
    print(f"   - 日志目录: {logdir}")
    print(f"   - 端口: {port}")
    print(f"   - 访问地址: {url}")
    print()
    
    # 启动TensorBoard
    cmd = [
        sys.executable, '-m', 'tensorboard.main',
        '--logdir', logdir,
        '--port', str(port),
        '--bind_all'
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        print("⏳ 等待TensorBoard启动...")
        time.sleep(3)
        
        # 检查进程是否还在运行
        if process.poll() is not None:
            output = process.stdout.read()
            print("❌ TensorBoard启动失败!")
            print(output)
            return False
        
        print("✅ TensorBoard已启动!")
        print()
        print("="*70)
        print("🌐 在浏览器中打开: " + url)
        print("="*70)
        print()
        
        # 自动打开浏览器
        try:
            webbrowser.open(url)
            print("✓ 已自动打开浏览器")
        except:
            print("⚠️ 无法自动打开浏览器,请手动访问: " + url)
        
        print()
        print("="*70)
        print("关键监控面板:")
        print("="*70)
        print()
        print("📈 SCALARS (标量) - 最重要!")
        print("   - train/loss_macl        ← MACL对比损失 (应从0.5降至0.2)")
        print("   - train/loss_cls         ← 分类损失")
        print("   - train/loss_bbox        ← 回归损失")
        print("   - train/loss_total       ← 总损失")
        print("   - train/grad_norm        ← 梯度范数 (应在5-15)")
        print("   - val/pascal_voc/mAP     ← 验证mAP (目标≥0.60)")
        print("   - train/lr               ← 学习率曲线")
        print()
        print("🎯 监控要点:")
        print("   1. loss_macl必须出现且下降 (最重要!)")
        print("   2. mAP应在epoch 1回升至0.55+")
        print("   3. grad_norm稳定在5-15之间")
        print("   4. 各loss项协调下降,无震荡")
        print()
        print("⚠️ 异常信号:")
        print("   - loss_macl缺失 → 双模态配对失败")
        print("   - grad_norm > 20 → 学习率过高")
        print("   - mAP < 0.52 → 训练崩溃")
        print()
        print("="*70)
        print("按 Ctrl+C 停止TensorBoard")
        print("="*70)
        print()
        
        # 保持进程运行
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n⚠️ 停止TensorBoard...")
            process.terminate()
            process.wait()
            print("✓ TensorBoard已停止")
        
        return True
        
    except FileNotFoundError:
        print("❌ 错误: TensorBoard未安装!")
        print()
        print("请先安装TensorBoard:")
        print("   pip install tensorboard")
        return False
    except Exception as e:
        print(f"❌ 启动TensorBoard时出错: {e}")
        return False

def print_monitoring_guide():
    """打印监控指南"""
    
    print()
    print("="*70)
    print("📚 TensorBoard 监控完整指南")
    print("="*70)
    print()
    
    print("🔍 面板1: SCALARS (标量曲线)")
    print("-"*70)
    print()
    print("Loss曲线组:")
    print("  • train/loss_macl")
    print("    - 正常: 0.5 → 0.3 → 0.2 (收敛)")
    print("    - 异常: 持续>0.5 或震荡")
    print()
    print("  • train/loss_cls + train/loss_bbox")
    print("    - 正常: 平滑下降")
    print("    - 异常: 震荡或上升")
    print()
    print("  • train/loss_total")
    print("    - 正常: 0.3 → 0.2 → 0.15")
    print("    - 包含检测loss + MACL loss")
    print()
    print("指标曲线组:")
    print("  • val/pascal_voc/mAP")
    print("    - 目标: Epoch 1 ≥ 0.55, Epoch 6 ≥ 0.60")
    print("    - 低于0.52: 失败")
    print()
    print("  • train/grad_norm")
    print("    - 正常: 5-15")
    print("    - 异常: >20 (不稳定)")
    print()
    print("  • train/lr")
    print("    - warmup阶段应该上升")
    print("    - 之后保持恒定(ConstantLR)")
    print()
    
    print()
    print("🎨 面板2: IMAGES (可视化)")
    print("-"*70)
    print("  • 预测框可视化")
    print("  • Ground Truth对比")
    print("  (如果启用了DetVisualizationHook)")
    print()
    
    print()
    print("📊 面板3: DISTRIBUTIONS (分布)")
    print("-"*70)
    print("  • 权重分布")
    print("  • 梯度分布")
    print("  (高级调试用)")
    print()
    
    print()
    print("⚡ 实时对比技巧:")
    print("-"*70)
    print("  1. 点击左侧特定曲线名称可隐藏/显示")
    print("  2. 使用 'Smoothing' 滑块平滑曲线")
    print("  3. 切换 'Horizontal Axis' 为 'STEP' 或 'WALL'")
    print("  4. 点击 'Show data download links' 导出数据")
    print()
    
    print("="*70)
    print()

def create_tensorboard_shortcut():
    """创建快速启动脚本"""
    
    # Windows批处理脚本
    bat_content = """@echo off
echo ========================================
echo Plan C TensorBoard 快速启动
echo ========================================
echo.
python configs/llvip/start_tensorboard.py
pause
"""
    
    with open('start_tensorboard.bat', 'w') as f:
        f.write(bat_content)
    
    print("✅ 已创建快捷启动脚本: start_tensorboard.bat")
    print("   双击即可启动TensorBoard")
    print()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='启动Plan C TensorBoard监控')
    parser.add_argument('--logdir', type=str, 
                       default='work_dirs/stage2_2_planC_dualmodality_macl',
                       help='训练日志目录')
    parser.add_argument('--port', type=int, default=6006,
                       help='TensorBoard端口 (默认: 6006)')
    parser.add_argument('--guide', action='store_true',
                       help='显示监控指南')
    
    args = parser.parse_args()
    
    if args.guide:
        print_monitoring_guide()
    else:
        create_tensorboard_shortcut()
        success = start_tensorboard(args.logdir, args.port)
        
        if not success:
            sys.exit(1)
