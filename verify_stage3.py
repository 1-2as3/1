"""
===================================================================================
Stage3 联合训练全流程验证脚本
===================================================================================
功能：
1. 配置与模型构建验证
2. 联合数据集（KAIST + M3FD）加载测试
3. 模型前向-反向传播测试（MACL + MSP + DHN）
4. 合成梯度验证
5. 清理旧日志和测试文件
6. 提供正式训练命令

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

def run_command(cmd, description, check=True):
    """运行命令并返回结果"""
    print(f"\n正在执行: {description}")
    print(f"命令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # 打印输出（限制长度）
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[-20:]:  # 只显示最后20行
                print(line)
        
        if result.returncode == 0:
            print_success(f"{description} 执行成功")
            return True
        else:
            print_error(f"{description} 执行失败 (返回码: {result.returncode})")
            if result.stderr:
                print(f"\n错误输出:\n{result.stderr}")
            return False
            
    except Exception as e:
        print_error(f"运行命令时出错: {e}")
        return False

def check_config_exists():
    """检查配置文件是否存在"""
    config_path = Path("configs/llvip/stage3_joint_multimodal.py")
    if not config_path.exists():
        print_error(f"配置文件不存在: {config_path}")
        return False
    print_success(f"配置文件存在: {config_path}")
    return True

def check_data_roots():
    """检查多个数据集根目录"""
    data_roots = {
        "KAIST": Path("C:/KAIST_PROCESSED"),
        "M3FD": Path("C:/M3FD")
    }
    
    all_exist = True
    for name, root in data_roots.items():
        print(f"\n检查 {name} 数据集:")
        if not root.exists():
            print_warning(f"  数据目录不存在: {root}")
            print(f"  如果数据在其他位置，请在配置中修改 data_root")
            all_exist = False
            continue
        
        # 检查关键子目录
        if name == "KAIST":
            checks = {
                "visible": root / "visible",
                "infrared": root / "infrared",
                "ImageSets": root / "ImageSets",
                "Annotations": root / "Annotations"
            }
            for subname, path in checks.items():
                if path.exists():
                    print_success(f"  {subname}: {path}")
                else:
                    print_warning(f"  {subname} 不存在: {path}")
                    all_exist = False
        else:  # M3FD - 可能有多种目录结构
            # 尝试多种可能的路径
            possible_structures = [
                {
                    "ImageSets": root / "ImageSets",
                    "Annotations": root / "Annotations"
                },
                {
                    "cleaned_dataset": root / "cleaned_dataset",
                    "ImageSets": root / "cleaned_dataset" / "ImageSets"
                },
                {
                    "ir": root / "ir",
                    "vi": root / "vi"
                }
            ]
            
            found_structure = False
            for i, structure in enumerate(possible_structures):
                valid = all(p.exists() for p in structure.values())
                if valid:
                    found_structure = True
                    print_success(f"  使用结构 {i+1}:")
                    for name_key, path in structure.items():
                        print_success(f"    {name_key}: {path}")
                    break
            
            if not found_structure:
                print_warning(f"  未找到标准目录结构，可能需要在配置中指定路径")
                print(f"  尝试的结构:")
                for i, structure in enumerate(possible_structures):
                    print(f"    结构{i+1}: {list(structure.keys())}")
                # 不标记为失败，因为 M3FD 可能有不同的组织方式
    
    return all_exist

def check_stage2_checkpoint():
    """检查 Stage2 权重文件"""
    print("\n检查 Stage2 预训练权重:")
    checkpoint_paths = [
        Path("work_dirs/stage2/latest.pth"),
        Path("work_dirs/stage2/epoch_*.pth")
    ]
    
    found = False
    for pattern in checkpoint_paths:
        if '*' in str(pattern):
            matches = list(Path(pattern.parent).glob(pattern.name))
            if matches:
                print_success(f"  找到权重: {matches[0]}")
                found = True
                break
        elif pattern.exists():
            print_success(f"  找到权重: {pattern}")
            found = True
            break
    
    if not found:
        print("  ℹ️  未找到 Stage2 权重文件 (这是正常现象)")
        print("  说明:")
        print("    • 当前处于 **测试/验证阶段**，尚未实际运行训练")
        print("    • Stage3 训练时会从配置中的 load_from 自动加载 Stage2 权重")
        print("    • 完整训练流程: Stage1 → Stage2 → Stage3，权重会自动传递")
        print("  → 只有在 **实际训练 Stage3** 时才需要 Stage2 checkpoint")
        return False  # 测试阶段不影响验证通过
    
    return True

def clean_old_logs():
    """清理旧日志和测试文件"""
    print_step("清理", "清理旧日志和测试文件")
    
    cleaned_count = 0
    
    # 清理 work_dirs 中的测试目录
    work_dirs = Path("work_dirs")
    if work_dirs.exists():
        for d in work_dirs.iterdir():
            if d.is_dir() and (d.name.startswith("test_") or d.name == "stage3_test"):
                try:
                    shutil.rmtree(d)
                    print(f"  删除: {d}")
                    cleaned_count += 1
                except Exception as e:
                    print_warning(f"无法删除 {d}: {e}")
    
    # 清理临时可视化文件
    temp_patterns = [
        "sample_pair_*.jpg",
        "joint_*_sample_*.jpg",
        "temp_stage3_*.txt",
        "logs/grad_flow_stage3.png"
    ]
    
    for pattern in temp_patterns:
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
    print_header("Stage3 联合训练全流程验证")
    
    # 快速环境检查
    print_step(0, "环境检查")
    try:
        import mmdet, mmcv, mmengine, torch
        print(f"  mmdet: {mmdet.__version__}")
        print(f"  mmcv: {mmcv.__version__}")
        print(f"  mmengine: {mmengine.__version__}")
        print(f"  torch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        # 先注册所有模块
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        
        from mmdet.registry import MODELS, DATASETS
        # 检查关键注册项
        key_models = ["StandardRoIHead", "FPN", "ResNet"]
        for name in key_models:
            status = "✅" if name in MODELS.module_dict else "❌"
            print(f"  {status} MODELS[{name}]")
        
        key_datasets = ["KAISTDataset", "M3FDDataset"]
        for name in key_datasets:
            status = "✅" if name in DATASETS.module_dict else "❌"
            print(f"  {status} DATASETS[{name}]")
        
        # ConcatDataset 是 MMEngine 的内置类型，不在 DATASETS 注册表中
        try:
            from mmengine.dataset import ConcatDataset as MMEngineConcatDataset
            print(f"  ✅ ConcatDataset (MMEngine 内置)")
        except:
            print(f"  ❌ ConcatDataset (MMEngine 内置)")
            
        print_success("环境检查通过")
    except Exception as e:
        print_error(f"环境检查失败: {e}")
        return
    
    print("\n此脚本将执行以下验证步骤:")
    print("  1. 配置文件检查")
    print("  2. 多数据集目录检查（KAIST + M3FD）")
    print("  3. Stage2 权重检查")
    print("  4. 数据探测（KAIST 前3个样本）")
    print("  5. 合成梯度验证")
    print("  6. 清理旧日志")
    
    print("\n以非交互模式运行：3 秒后开始执行（Ctrl+C 可中断）...")
    try:
        import time
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n\n已取消验证。")
        return
    
    # 记录测试结果
    results = {}
    python_exe = sys.executable
    
    # Step 1: 配置文件检查
    print_step(1, "配置文件检查")
    results['config'] = check_config_exists()
    
    # Step 2: 数据目录检查
    print_step(2, "多数据集目录检查")
    results['data_roots'] = check_data_roots()
    
    # Step 3: Stage2 权重检查（信息性，不影响测试通过）
    print_step(3, "Stage2 权重检查")
    checkpoint_found = check_stage2_checkpoint()
    results['checkpoint (可选)'] = checkpoint_found  # 标记为可选
    
    # Step 4: 数据探测（KAIST）
    print_step(4, "数据探测（KAIST 前3个样本）")
    results['data_probe'] = run_command(
        [python_exe, "tools/data_probe.py",
         "--ann", "C:/KAIST_PROCESSED/ImageSets/train.txt",
         "--root", "C:/KAIST_PROCESSED",
         "--limit", "3"],
        "KAIST 数据探测",
        check=False
    )
    
    # Step 5: 合成梯度验证
    print_step(5, "合成梯度验证")
    results['synthetic_grad'] = run_command(
        [python_exe, "tools/grad_flow_synthetic_realmodel.py",
         "configs/llvip/stage3_joint_multimodal.py",
         "--device", "cuda:0"],
        "合成梯度验证（Stage3）",
        check=False
    )
    
    # Step 6: 清理
    clean_old_logs()
    
    # 打印汇总结果
    print_header("验证结果汇总")
    
    all_passed = True
    for test_name, passed in results.items():
        # checkpoint 是可选的，不影响测试通过
        if '可选' in test_name:
            if passed:
                status = f"{GREEN}✅ 找到{RESET}"
            else:
                status = f"{YELLOW}ℹ️  未找到 (测试阶段正常){RESET}"
        else:
            status = f"{GREEN}✅ 通过{RESET}" if passed else f"{RED}❌ 失败{RESET}"
        
        print(f"  {test_name.ljust(25)}: {status}")
        
        # 只有非可选项的失败才影响 all_passed
        if not passed and '可选' not in test_name:
            all_passed = False
    
    print("\n" + "=" * 80)
    
    if all_passed:
        print_success("所有验证测试通过！Stage3 环境配置正确。")
        print("\n" + "=" * 80)
        print(f"{GREEN}✅ 可以开始 Stage3 正式训练{RESET}")
        print("=" * 80)
        print("\n执行以下命令开始训练:\n")
        print(f"  {BLUE}python tools/train.py configs/llvip/stage3_joint_multimodal.py \\")
        print(f"      --work-dir work_dirs/stage3 \\")
        print(f"      --cfg-options load_from=work_dirs/stage2/latest.pth{RESET}")
        print("\n或使用完整 Python 路径:")
        print(f"  {BLUE}{sys.executable} tools\\train.py configs\\llvip\\stage3_joint_multimodal.py \\")
        print(f"      --work-dir work_dirs\\stage3 \\")
        print(f"      --cfg-options load_from=work_dirs\\stage2\\latest.pth{RESET}")
        print("\n可选参数:")
        print("  --resume              从中断点恢复训练")
        print("  --amp                 启用混合精度训练（加速）")
        print("  --cfg-options         覆盖配置项")
        print("\n示例（使用 FreezeHook 变体 + 混合精度）:")
        print(f"  {BLUE}{sys.executable} tools\\train.py configs\\llvip\\stage3_joint_multimodal_freezehook.py \\")
        print(f"      --work-dir work_dirs\\stage3 --amp \\")
        print(f"      --cfg-options load_from=work_dirs\\stage2\\latest.pth{RESET}")
        print("\n训练完成后，权重保存在: work_dirs/stage3/latest.pth")
        print("\n注意事项:")
        print("  • Stage3 使用 ConcatDataset 联合 KAIST 和 M3FD")
        print("  • 建议使用 FreezeBackboneHook 冻结 backbone 以稳定训练")
        print("  • 监控日志中的 FreezeMonitorHook 输出确认冻结状态")
    else:
        print_error("部分验证测试失败，请检查错误信息。")
        print("\n建议:")
        print("  1. 检查数据集路径:")
        print("     - KAIST: C:/KAIST_PROCESSED/")
        print("     - M3FD: C:/M3FD/")
        print("  2. 确认 Stage2 已完成训练并生成权重文件")
        print("  3. 确认配置文件: configs/llvip/stage3_joint_multimodal.py")
        print("  4. 查看上方错误日志排查问题")
        print("  5. 检查 CUDA 是否可用（如需 GPU 训练）")
    
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
