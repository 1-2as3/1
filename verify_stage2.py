"""
===================================================================================
KAIST Stage2 全流程验证脚本
===================================================================================
功能：
1. 配置与模型构建验证
2. 数据集加载测试
3. 模型前向-反向传播测试
4. 清理旧日志和测试文件
5. 提供正式训练命令

注意：此脚本仅用于验证环境和配置，不执行实际训练。
===================================================================================
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# 颜色输出（Windows 兼容）
try:
    import colorama
    colorama.init()
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
except:
    GREEN = YELLOW = RED = BLUE = RESET = ''

def print_header(text):
    """打印标题"""
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{text.center(80)}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}\n")

def print_step(step_num, text):
    """打印步骤"""
    print(f"\n{GREEN}🚀 Step {step_num}: {text}{RESET}")
    print("-" * 80)

def print_success(text):
    """打印成功消息"""
    print(f"{GREEN}✅ {text}{RESET}")

def print_warning(text):
    """打印警告消息"""
    print(f"{YELLOW}⚠️  {text}{RESET}")

def print_error(text):
    """打印错误消息"""
    print(f"{RED}❌ {text}{RESET}")

def run_test(script_name, description, timeout=300):
    """运行测试脚本"""
    print(f"\n正在运行: {script_name}")
    print(f"描述: {description}")
    print(f"超时时间: {timeout}秒\n")
    
    if not os.path.exists(script_name):
        print_error(f"测试脚本不存在: {script_name}")
        return False
    
    try:
        # 使用当前 Python 环境运行脚本
        python_exe = sys.executable
        print(f"正在执行...（如果卡住超过 {timeout} 秒将自动终止）")
        result = subprocess.run(
            [python_exe, script_name],
            capture_output=False,
            text=True,
            check=False,
            timeout=timeout
        )
        
        if result.returncode == 0:
            print_success(f"{script_name} 执行成功")
            return True
        else:
            print_error(f"{script_name} 执行失败 (返回码: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print_error(f"{script_name} 执行超时（超过 {timeout} 秒）")
        print_warning("建议检查数据集路径或减少测试样本数量")
        return False
    except Exception as e:
        print_error(f"运行 {script_name} 时出错: {e}")
        return False

def clean_old_logs():
    """清理旧日志和测试文件"""
    print_step("清理", "清理旧日志和测试文件")
    
    cleaned_count = 0
    
    # 清理 work_dirs 中的测试目录
    work_dirs = Path("work_dirs")
    if work_dirs.exists():
        for d in work_dirs.iterdir():
            if d.is_dir() and d.name.startswith("test_"):
                try:
                    shutil.rmtree(d)
                    print(f"  删除: {d}")
                    cleaned_count += 1
                except Exception as e:
                    print_warning(f"无法删除 {d}: {e}")
    
    # 清理临时可视化文件
    temp_files = [
        "sample_pair_*.jpg",
        "kaist_paired_sample_*.jpg",
        "temp_*.txt"
    ]
    
    for pattern in temp_files:
        for f in Path(".").glob(pattern):
            try:
                f.unlink()
                print(f"  删除: {f}")
                cleaned_count += 1
            except Exception as e:
                print_warning(f"无法删除 {f}: {e}")
    
    if cleaned_count > 0:
        print_success(f"已清理 {cleaned_count} 个文件/目录")
    else:
        print_success("没有需要清理的文件")

def main():
    """主函数"""
    print_header("KAIST Stage2 全流程验证")
    # 环境与注册表快速检查
    print_step(0, "环境检查")
    try:
        import mmdet, mmcv, mmengine, torch
        print(f"  mmdet: {mmdet.__version__}")
        print(f"  mmcv: {mmcv.__version__}")
        print(f"  mmengine: {mmengine.__version__}")
        print(f"  torch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        # 先注册所有模块（包括自定义模块）
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        
        from mmdet.registry import MODELS, DATASETS
        # 检查标准模块
        standard_models = ["StandardRoIHead", "FPN", "ResNet"]
        for name in standard_models:
            status = "✅" if name in MODELS.module_dict else "❌"
            print(f"  {status} MODELS[{name}]")
        
        # 检查自定义模块
        custom_models = ["AlignedRoIHead", "DomainAligner", "MMDLoss"]
        for name in custom_models:
            status = "✅" if name in MODELS.module_dict else "❌"
            print(f"  {status} MODELS[{name}] (自定义)")
        
        # 检查数据集
        datasets = ["KAISTDataset", "LLVIPDataset"]
        for name in datasets:
            status = "✅" if name in DATASETS.module_dict else "❌"
            print(f"  {status} DATASETS[{name}]")
        
        print_success("环境检查通过")
    except Exception as e:
        print_error(f"环境检查失败：{e}")
        import traceback
        traceback.print_exc()
        return
    
    print("此脚本将执行以下验证步骤:")
    print("  1. 配置与模型构建验证")
    print("  2. 数据探测（快速检查前3个样本）")
    print("  3. 合成梯度验证（前向+反向）")
    print("  4. 清理旧日志")
    print("\n注意: 为加快验证速度，已用合成梯度验证替代完整数据集加载")
    # 非交互模式，直接执行
    print("\n以非交互模式运行：3 秒后开始执行（Ctrl+C 可中断）...")
    try:
        import time
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n\n已取消验证。")
        return
    
    # 记录测试结果
    results = {}
    
    # Step 1: 配置与模型构建验证
    print_step(1, "配置与模型构建验证")
    results['build'] = run_test(
        "test_stage2_build.py",
        "验证配置文件加载和模型构建"
    )
    
    # Step 2: 数据探测（快速检查）
    print_step(2, "数据探测（快速检查前3个样本）")
    python_exe = sys.executable
    try:
        print("正在运行数据探测...")
        result = subprocess.run(
            [python_exe, "tools/data_probe.py",
             "--ann", "C:/KAIST_PROCESSED/ImageSets/train.txt",
             "--root", "C:/KAIST_PROCESSED",
             "--limit", "3"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(result.stdout)
            print_success("数据探测完成")
            results['data_probe'] = True
        else:
            print_error("数据探测失败")
            if result.stderr:
                print(result.stderr)
            results['data_probe'] = False
    except Exception as e:
        print_error(f"数据探测出错: {e}")
        results['data_probe'] = False
    
    # Step 3: 合成梯度验证（更快速）
    print_step(3, "合成梯度验证")
    try:
        print("正在运行合成梯度验证...")
        result = subprocess.run(
            [python_exe, "tools/grad_flow_synthetic_realmodel.py",
             "configs/llvip/stage2_kaist_domain_ft_nodomain.py",
             "--device", "cuda:0"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            # 只显示关键输出
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:
                print(line)
            print_success("合成梯度验证完成")
            results['synthetic_grad'] = True
        else:
            print_error("合成梯度验证失败")
            if result.stderr:
                print(result.stderr[-500:])  # 只显示最后500字符
            results['synthetic_grad'] = False
    except subprocess.TimeoutExpired:
        print_error("合成梯度验证超时（超过120秒）")
        results['synthetic_grad'] = False
    except Exception as e:
        print_error(f"合成梯度验证出错: {e}")
        results['synthetic_grad'] = False
    
    # Step 4: 清理
    clean_old_logs()
    
    # 打印汇总结果
    print_header("验证结果汇总")
    
    all_passed = True
    for test_name, passed in results.items():
        status = f"{GREEN}✅ 通过{RESET}" if passed else f"{RED}❌ 失败{RESET}"
        print(f"  {test_name.ljust(15)}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    
    if all_passed:
        print_success("所有验证测试通过！环境配置正确。")
        print("\n" + "=" * 80)
        print(f"{GREEN}✅ 可以开始正式训练{RESET}")
        print("=" * 80)
        print("\n执行以下命令开始训练:\n")
        print(f"  {BLUE}python tools/train.py configs/llvip/stage2_kaist_domain_ft_nodomain.py \\")
        print(f"      --work-dir work_dirs/stage2 \\")
        print(f"      --cfg-options load_from=work_dirs/stage1/latest.pth{RESET}")
        print("\n或使用完整 Python 路径:")
        print(f"  {BLUE}{sys.executable} tools\\train.py configs\\llvip\\stage2_kaist_domain_ft_nodomain.py \\")
        print(f"      --work-dir work_dirs\\stage2 \\")
        print(f"      --cfg-options load_from=work_dirs\\stage1\\latest.pth{RESET}")
        print("\n使用 FreezeHook 变体（推荐）:")
        print(f"  {BLUE}{sys.executable} tools\\train.py configs\\llvip\\stage2_kaist_domain_ft_nodomain_freezehook.py \\")
        print(f"      --work-dir work_dirs\\stage2 \\")
        print(f"      --cfg-options load_from=work_dirs\\stage1\\latest.pth{RESET}")
        print("\n可选参数:")
        print("  --resume              从中断点恢复训练")
        print("  --amp                 启用混合精度训练（加速）")
        print("  --cfg-options         覆盖配置项")
        print("\n示例（使用 FreezeHook 变体 + 混合精度）:")
        print(f"  {BLUE}{sys.executable} tools\\train.py configs\\llvip\\stage2_kaist_domain_ft_nodomain_freezehook.py \\")
        print(f"      --work-dir work_dirs\\stage2 --amp \\")
        print(f"      --cfg-options load_from=work_dirs\\stage1\\latest.pth{RESET}")
    else:
        print_error("部分验证测试失败，请检查错误信息。")
        print("\n建议:")
        print("  1. 检查数据集路径是否正确: C:/KAIST_PROCESSED/")
        print("  2. 确认配置文件: configs/llvip/stage2_kaist_domain_ft_nodomain.py")
        print("  3. 查看上方错误日志排查问题")
        print("  4. 检查 CUDA 是否可用（如需 GPU 训练）")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n验证已中断。")
    except Exception as e:
        print_error(f"验证过程出错: {e}")
        import traceback
        traceback.print_exc()
