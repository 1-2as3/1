"""
多实验TensorBoard对比启动脚本
================================
同时监控多个实验的训练曲线进行对比
"""

import subprocess
import sys
import os
import webbrowser
import time

def start_tensorboard_comparison(experiments, port=6006):
    """启动TensorBoard对比多个实验"""
    
    print("="*70)
    print("TensorBoard 多实验对比监控")
    print("="*70)
    print()
    
    # 检查实验目录
    valid_exps = []
    for name, path in experiments.items():
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
            valid_exps.append((name, path))
        else:
            print(f"✗ {name}: {path} (不存在)")
    
    if not valid_exps:
        print("\n❌ 没有找到有效的实验目录!")
        return False
    
    print()
    print(f"找到 {len(valid_exps)} 个实验,准备对比")
    print()
    
    # 构建logdir参数 (逗号分隔多个实验)
    logdir_arg = ','.join([f"{name}:{path}" for name, path in valid_exps])
    
    url = f"http://localhost:{port}"
    
    print(f"📊 启动TensorBoard服务...")
    print(f"   - 端口: {port}")
    print(f"   - 访问地址: {url}")
    print()
    
    cmd = [
        sys.executable, '-m', 'tensorboard.main',
        '--logdir_spec', logdir_arg,
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
        
        try:
            webbrowser.open(url)
            print("✓ 已自动打开浏览器")
        except:
            print("⚠️ 请手动访问: " + url)
        
        print()
        print("="*70)
        print("📊 对比分析要点:")
        print("="*70)
        print()
        print("1. 在SCALARS面板,所有实验的曲线会叠加显示")
        print("2. 不同实验用不同颜色区分")
        print("3. 重点对比:")
        print("   • loss_macl收敛速度")
        print("   • mAP提升幅度")
        print("   • grad_norm稳定性")
        print()
        print("4. 使用左侧过滤器筛选特定实验")
        print("5. 点击曲线名称可隐藏/显示")
        print()
        print("="*70)
        print("按 Ctrl+C 停止TensorBoard")
        print("="*70)
        print()
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n⚠️ 停止TensorBoard...")
            process.terminate()
            process.wait()
            print("✓ TensorBoard已停止")
        
        return True
        
    except Exception as e:
        print(f"❌ 启动TensorBoard时出错: {e}")
        return False

if __name__ == '__main__':
    # 定义要对比的实验
    experiments = {
        'Plan_C': 'work_dirs/stage2_2_planC_dualmodality_macl',
        'Plan_B': 'work_dirs/stage2_1_planB_macl_rescue',
        'Pure_Det': 'work_dirs/stage2_1_pure_detection',
        'Stage2.1': 'work_dirs/stage2_1_kaist_detonly',
    }
    
    print("可对比的实验:")
    print("="*70)
    for name in experiments.keys():
        print(f"  • {name}")
    print()
    
    start_tensorboard_comparison(experiments, port=6006)
