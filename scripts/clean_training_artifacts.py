"""
Training Artifacts Cleanup Script
清理所有历史训练产生的文件，为正式训练准备干净环境

清理内容：
1. work_dirs/ 中所有训练目录（保留结构）
2. reports/ 中的临时训练导出
3. 根目录的训练日志文件
4. TensorBoard events 文件
5. 测试可视化输出

警告：此操作不可逆，请确保已备份重要数据！
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def get_dir_size(path):
    """计算目录大小（MB）"""
    total = 0
    try:
        for entry in Path(path).rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception as e:
        print(f"    [WARNING] Error calculating size: {e}")
    return total / 1024 / 1024

def safe_remove(path, is_dir=True):
    """安全删除文件或目录"""
    try:
        if is_dir:
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
        return True
    except Exception as e:
        print(f"    [ERROR] Failed to remove {path}: {e}")
        return False

def main():
    root = Path(r"C:\Users\Xinyu\mmdetection")
    
    print("=" * 80)
    print("TRAINING ARTIFACTS CLEANUP SCRIPT")
    print("=" * 80)
    print(f"\nRoot directory: {root}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 统计变量
    total_deleted_size = 0
    total_deleted_files = 0
    total_deleted_dirs = 0
    
    # ========== 1. work_dirs 清理 ==========
    print(f"\n[1] Cleaning work_dirs/")
    work_dirs = root / "work_dirs"
    
    if work_dirs.exists():
        # 需要清理的目录列表（包括实际存在的和配置中的）
        cleanup_targets = [
            "stage1_llvip_pretrain",      # Stage 1 训练输出
            "llvip_pretrain",              # 可能的旧命名
            "stage2_kaist_domain_ft",      # Stage 2 训练输出
            "kaist_domain_ft",             # 可能的旧命名
            "stage3_joint_multimodal",     # Stage 3 训练输出
            "metrics_logs",                # 指标日志（7.5GB！）
            "tsne_vis",                    # t-SNE 可视化
        ]
        
        for target in cleanup_targets:
            target_path = work_dirs / target
            if target_path.exists():
                # 计算大小
                dir_size = get_dir_size(target_path)
                file_count = sum(1 for _ in target_path.rglob('*') if _.is_file())
                
                print(f"  Removing {target}/ ({file_count} files, {dir_size:.1f} MB)")
                
                if safe_remove(target_path):
                    total_deleted_size += dir_size
                    total_deleted_files += file_count
                    total_deleted_dirs += 1
                    print(f"    [OK] Deleted successfully")
                else:
                    print(f"    [FAILED] Could not delete")
            else:
                print(f"  {target}/ - Not found, skipping")
    else:
        print("  work_dirs/ not found")
    
    # ========== 2. reports/ 清理 ==========
    print(f"\n[2] Cleaning reports/ temporary exports")
    reports = root / "reports"
    
    if reports.exists():
        # 清理带时间戳的临时导出目录
        temp_patterns = ["training_*", "test_*", "debug_*", "temp_*"]
        
        for pattern in temp_patterns:
            for temp_dir in reports.glob(pattern):
                if temp_dir.is_dir():
                    dir_size = get_dir_size(temp_dir)
                    file_count = sum(1 for _ in temp_dir.rglob('*') if _.is_file())
                    
                    print(f"  Removing {temp_dir.name}/ ({file_count} files, {dir_size:.2f} MB)")
                    
                    if safe_remove(temp_dir):
                        total_deleted_size += dir_size
                        total_deleted_files += file_count
                        total_deleted_dirs += 1
                        print(f"    [OK] Deleted successfully")
    else:
        print("  reports/ not found")
    
    # ========== 3. 根目录日志清理 ==========
    print(f"\n[3] Cleaning root directory logs")
    log_patterns = ["*.log", "*.log.json"]
    
    for pattern in log_patterns:
        for log_file in root.glob(pattern):
            if log_file.is_file():
                size_kb = log_file.stat().st_size / 1024
                print(f"  Removing {log_file.name} ({size_kb:.1f} KB)")
                
                if safe_remove(log_file, is_dir=False):
                    total_deleted_size += size_kb / 1024
                    total_deleted_files += 1
                    print(f"    [OK] Deleted successfully")
    
    # ========== 4. 其他训练产物清理 ==========
    print(f"\n[4] Cleaning other training artifacts")
    
    # 可能的其他目录
    other_dirs = ["vis_data", "runs", "outputs", "temp", "debug_output"]
    
    for dirname in other_dirs:
        dirpath = root / dirname
        if dirpath.exists():
            dir_size = get_dir_size(dirpath)
            file_count = sum(1 for _ in dirpath.rglob('*') if _.is_file())
            
            print(f"  Removing {dirname}/ ({file_count} files, {dir_size:.2f} MB)")
            
            if safe_remove(dirpath):
                total_deleted_size += dir_size
                total_deleted_files += file_count
                total_deleted_dirs += 1
                print(f"    [OK] Deleted successfully")
    
    # ========== 5. 清理测试脚本产生的临时文件 ==========
    print(f"\n[5] Cleaning test script outputs")
    test_outputs = [
        "test_output.txt",
        "verify_output.txt", 
        "debug_output.txt",
        "__pycache__",
        "*.pyc",
    ]
    
    for pattern in test_outputs:
        if "*" in pattern:
            # 通配符模式
            for file in root.glob(pattern):
                if file.is_file():
                    size_kb = file.stat().st_size / 1024
                    if safe_remove(file, is_dir=False):
                        total_deleted_size += size_kb / 1024
                        total_deleted_files += 1
        else:
            # 直接路径
            filepath = root / pattern
            if filepath.exists():
                if filepath.is_dir():
                    dir_size = get_dir_size(filepath)
                    if safe_remove(filepath):
                        total_deleted_size += dir_size
                        total_deleted_dirs += 1
                else:
                    size_kb = filepath.stat().st_size / 1024
                    if safe_remove(filepath, is_dir=False):
                        total_deleted_size += size_kb / 1024
                        total_deleted_files += 1
    
    # ========== 最终统计 ==========
    print("\n" + "=" * 80)
    print("CLEANUP SUMMARY")
    print("=" * 80)
    print(f"\nDeleted:")
    print(f"  - Directories: {total_deleted_dirs}")
    print(f"  - Files: {total_deleted_files}")
    print(f"  - Total size: {total_deleted_size:.1f} MB ({total_deleted_size/1024:.2f} GB)")
    
    # ========== 验证清理结果 ==========
    print(f"\n[VERIFY] Checking remaining artifacts...")
    
    remaining_items = []
    
    if work_dirs.exists():
        remaining_in_work = [d.name for d in work_dirs.iterdir() if d.is_dir()]
        if remaining_in_work:
            remaining_items.append(f"work_dirs/: {', '.join(remaining_in_work)}")
    
    if reports.exists():
        temp_dirs = [d.name for d in reports.iterdir() if d.is_dir() and 
                     any(pattern in d.name for pattern in ["training_", "test_", "debug_", "temp_"])]
        if temp_dirs:
            remaining_items.append(f"reports/: {', '.join(temp_dirs)}")
    
    root_logs = list(root.glob("*.log*"))
    if root_logs:
        remaining_items.append(f"root logs: {len(root_logs)} files")
    
    if remaining_items:
        print("  [WARNING] Some items remain:")
        for item in remaining_items:
            print(f"    - {item}")
    else:
        print("  [OK] All training artifacts cleaned successfully!")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Cleanup completed!")
    print("=" * 80)
    print("\nReady for clean training runs:")
    print("  Stage 1: python tools/train.py configs/llvip/stage1_llvip_pretrain.py")
    print("  Stage 2: python tools/train.py configs/llvip/stage2_kaist_domain_ft.py")
    print("  Stage 3: python tools/train.py configs/llvip/stage3_joint_multimodal.py")
    print("=" * 80)

if __name__ == "__main__":
    # 安全确认
    print("=" * 80)
    print("WARNING: This will DELETE all training artifacts!")
    print("=" * 80)
    print("\nThe following will be removed:")
    print("  - All checkpoint files (.pth)")
    print("  - All training logs")
    print("  - All TensorBoard events")
    print("  - All temporary exports")
    print("  - Total estimated size: ~14 GB")
    print("\nThis operation is IRREVERSIBLE!")
    print("=" * 80)
    
    response = input("\nDo you want to continue? (type 'yes' to confirm): ")
    
    if response.lower() == 'yes':
        print("\nStarting cleanup...\n")
        main()
    else:
        print("\nOperation cancelled. No files were deleted.")
