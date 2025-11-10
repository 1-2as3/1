#!/usr/bin/env python3
"""清除所有测试缓存，准备正式训练"""

import os
import shutil
from pathlib import Path

# ANSI 颜色代码
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"

def print_section(title):
    """打印分节标题"""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{title:^80}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")

def clean_pycache():
    """清除所有 __pycache__ 目录"""
    print(f"{YELLOW}🧹 清除 Python 缓存 (__pycache__)...{RESET}")
    count = 0
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_dir = Path(root) / '__pycache__'
            try:
                shutil.rmtree(cache_dir)
                print(f"  ✓ 删除: {cache_dir}")
                count += 1
            except Exception as e:
                print(f"  ✗ 失败: {cache_dir} - {e}")
    print(f"{GREEN}✅ 删除了 {count} 个 __pycache__ 目录{RESET}\n")

def clean_logs():
    """清除日志文件"""
    print(f"{YELLOW}🧹 清除日志文件...{RESET}")
    log_patterns = [
        '*.log',
        'test_*.log',
        'verify_*.log',
        'grad_flow_*.log'
    ]
    count = 0
    for pattern in log_patterns:
        for log_file in Path('.').glob(pattern):
            try:
                log_file.unlink()
                print(f"  ✓ 删除: {log_file}")
                count += 1
            except Exception as e:
                print(f"  ✗ 失败: {log_file} - {e}")
    
    # 清除 tools 目录下的日志
    tools_dir = Path('tools')
    if tools_dir.exists():
        for log_file in tools_dir.glob('*.log'):
            try:
                log_file.unlink()
                print(f"  ✓ 删除: {log_file}")
                count += 1
            except Exception as e:
                print(f"  ✗ 失败: {log_file} - {e}")
    
    print(f"{GREEN}✅ 删除了 {count} 个日志文件{RESET}\n")

def clean_work_dirs():
    """清除临时工作目录"""
    print(f"{YELLOW}🧹 清除临时工作目录...{RESET}")
    
    # 列出 work_dirs 下的所有目录
    work_dirs = Path('work_dirs')
    if work_dirs.exists():
        print(f"  发现 work_dirs 目录，内容:")
        subdirs = list(work_dirs.iterdir())
        if subdirs:
            for subdir in subdirs:
                if subdir.is_dir():
                    print(f"    - {subdir.name}/")
            
            print(f"\n  {RED}警告: 这些可能包含训练权重!{RESET}")
            print(f"  建议手动检查后再决定是否删除")
            print(f"  如需保留，请移动到安全位置")
        else:
            print(f"  work_dirs 为空")
    else:
        print(f"  work_dirs 目录不存在")
    print()

def clean_test_outputs():
    """清除测试输出文件"""
    print(f"{YELLOW}🧹 清除测试输出文件...{RESET}")
    test_patterns = [
        'test_output_*.txt',
        'test_result_*.json',
        'synthetic_*.pth',
        'debug_*.png',
        'temp_*.py'
    ]
    count = 0
    for pattern in test_patterns:
        for file in Path('.').glob(pattern):
            try:
                file.unlink()
                print(f"  ✓ 删除: {file}")
                count += 1
            except Exception as e:
                print(f"  ✗ 失败: {file} - {e}")
    
    print(f"{GREEN}✅ 删除了 {count} 个测试输出文件{RESET}\n")

def clean_mmdet_cache():
    """清除 MMDetection 相关缓存"""
    print(f"{YELLOW}🧹 清除 MMDetection 缓存...{RESET}")
    cache_dirs = [
        Path.home() / '.cache' / 'mmdet',
        Path.home() / '.cache' / 'mmcv',
        Path.home() / '.cache' / 'mmengine',
    ]
    count = 0
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            try:
                # 只清除缓存文件，不删除整个目录
                for item in cache_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                        count += 1
                print(f"  ✓ 清除: {cache_dir} ({count} 文件)")
            except Exception as e:
                print(f"  ✗ 失败: {cache_dir} - {e}")
        else:
            print(f"  ℹ 不存在: {cache_dir}")
    
    if count > 0:
        print(f"{GREEN}✅ 清除了 {count} 个缓存文件{RESET}\n")
    else:
        print(f"{GREEN}✅ 没有缓存文件需要清除{RESET}\n")

def clean_jupyter_checkpoints():
    """清除 Jupyter notebook 检查点"""
    print(f"{YELLOW}🧹 清除 Jupyter checkpoints...{RESET}")
    count = 0
    for checkpoint_dir in Path('.').rglob('.ipynb_checkpoints'):
        try:
            shutil.rmtree(checkpoint_dir)
            print(f"  ✓ 删除: {checkpoint_dir}")
            count += 1
        except Exception as e:
            print(f"  ✗ 失败: {checkpoint_dir} - {e}")
    
    if count > 0:
        print(f"{GREEN}✅ 删除了 {count} 个 checkpoint 目录{RESET}\n")
    else:
        print(f"{GREEN}✅ 没有 checkpoint 需要清除{RESET}\n")

def clean_pyc_files():
    """清除 .pyc 文件"""
    print(f"{YELLOW}🧹 清除 .pyc 文件...{RESET}")
    count = 0
    for pyc_file in Path('.').rglob('*.pyc'):
        try:
            pyc_file.unlink()
            count += 1
            if count <= 10:  # 只显示前10个
                print(f"  ✓ 删除: {pyc_file}")
        except Exception as e:
            print(f"  ✗ 失败: {pyc_file} - {e}")
    
    if count > 10:
        print(f"  ... 共 {count} 个文件")
    print(f"{GREEN}✅ 删除了 {count} 个 .pyc 文件{RESET}\n")

def main():
    print_section("清除所有测试缓存，准备正式训练")
    
    print(f"{BLUE}此脚本将清除以下内容:{RESET}")
    print(f"  • Python 缓存 (__pycache__, .pyc)")
    print(f"  • 测试日志文件")
    print(f"  • 测试输出文件")
    print(f"  • Jupyter checkpoints")
    print(f"  • MMDetection 缓存")
    print(f"  • 显示 work_dirs 内容（需手动处理）")
    
    print(f"\n{YELLOW}⚠️  注意: work_dirs 中可能包含训练权重，需要手动检查{RESET}")
    print(f"{YELLOW}⚠️  建议在清理前备份重要数据{RESET}\n")
    
    response = input("确认开始清理? [y/N]: ").strip().lower()
    if response != 'y':
        print(f"{RED}❌ 已取消清理{RESET}")
        return
    
    # 执行清理
    clean_pycache()
    clean_pyc_files()
    clean_logs()
    clean_test_outputs()
    clean_jupyter_checkpoints()
    clean_mmdet_cache()
    clean_work_dirs()
    
    print_section("清理完成")
    print(f"{GREEN}✅ 所有测试缓存已清除{RESET}")
    print(f"\n{BLUE}接下来可以开始正式训练:{RESET}")
    print(f"\n{GREEN}Stage 1 (LLVIP - MACL+MSP):{RESET}")
    print(f"  python tools/train.py configs/llvip/stage1_llvip_macl_msp.py \\")
    print(f"      --work-dir work_dirs/stage1")
    
    print(f"\n{GREEN}Stage 2 (KAIST - 域对齐微调):{RESET}")
    print(f"  python tools/train.py configs/llvip/stage2_kaist_domain_ft_nodomain.py \\")
    print(f"      --work-dir work_dirs/stage2 \\")
    print(f"      --cfg-options load_from=work_dirs/stage1/latest.pth")
    
    print(f"\n{GREEN}Stage 3 (联合训练):{RESET}")
    print(f"  python tools/train.py configs/llvip/stage3_joint_multimodal.py \\")
    print(f"      --work-dir work_dirs/stage3 \\")
    print(f"      --cfg-options load_from=work_dirs/stage2/latest.pth")
    
    print(f"\n{YELLOW}提示:{RESET}")
    print(f"  • 可以运行 verify_stage1.py / verify_stage2.py / verify_stage3.py 进行最后检查")
    print(f"  • 训练前确保数据集路径正确")
    print(f"  • 建议使用 --amp 启用混合精度训练加速")
    print()

if __name__ == '__main__':
    main()
