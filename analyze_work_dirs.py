"""
work_dirs 目录清理脚本
=======================
分析各目录的价值并提供清理建议
"""

import os
import os.path as osp
from pathlib import Path
import shutil

def get_dir_size(path):
    """计算目录大小(MB)"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except:
        pass
    return total / (1024 * 1024)  # MB

def analyze_work_dirs():
    """分析work_dirs下所有目录"""
    
    work_dirs = Path('work_dirs')
    if not work_dirs.exists():
        print("work_dirs 目录不存在!")
        return
    
    print("="*80)
    print("work_dirs 目录空间占用分析")
    print("="*80)
    
    dirs_info = []
    
    for d in work_dirs.iterdir():
        if not d.is_dir():
            continue
        
        size_mb = get_dir_size(str(d))
        
        # 统计文件类型
        pth_count = len(list(d.rglob('*.pth')))
        log_count = len(list(d.rglob('*.log')))
        vis_count = len(list(d.rglob('vis_data')))
        
        dirs_info.append({
            'name': d.name,
            'size_mb': size_mb,
            'pth_count': pth_count,
            'log_count': log_count,
            'vis_count': vis_count,
            'path': str(d)
        })
    
    # 按大小排序
    dirs_info.sort(key=lambda x: x['size_mb'], reverse=True)
    
    total_size = sum(d['size_mb'] for d in dirs_info)
    
    print(f"\n总计: {len(dirs_info)} 个目录, {total_size:.1f} MB")
    print("\n" + "-"*80)
    print(f"{'目录名':<40} {'大小(MB)':<12} {'checkpoints':<12} {'日志':<8}")
    print("-"*80)
    
    for d in dirs_info:
        print(f"{d['name']:<40} {d['size_mb']:>10.1f}   {d['pth_count']:>10}   {d['log_count']:>6}")
    
    print("\n" + "="*80)
    print("清理建议 (基于目录用途和价值)")
    print("="*80)
    
    # 分类建议
    keep_dirs = []
    archive_dirs = []
    delete_dirs = []
    
    for d in dirs_info:
        name = d['name']
        
        # 关键目录 - 保留
        if any(k in name for k in ['stage1_longrun_full', 'stage2_1_pure_detection']):
            keep_dirs.append(d)
            continue
        
        # Plan B失败目录 - 可删除
        if 'planB_macl_rescue' in name:
            delete_dirs.append(d)
            continue
        
        # 测试/验证目录 - 可删除
        if any(k in name for k in ['test_validation', 'sanity', 'emergency']):
            delete_dirs.append(d)
            continue
        
        # 归档目录 - 已归档可删除
        if '_archive' in name:
            delete_dirs.append(d)
            continue
        
        # 旧版本/冗余目录
        if any(k in name for k in ['_v1', 'conservative', 'remote']):
            archive_dirs.append(d)
            continue
        
        # 可视化目录
        if 'tsne_vis' in name or 'vis_data' in name:
            archive_dirs.append(d)
            continue
        
        # 其他目录需要检查checkpoint
        if d['pth_count'] > 0:
            keep_dirs.append(d)
        else:
            delete_dirs.append(d)
    
    print("\n✅ 保留目录 (包含重要checkpoint):")
    keep_size = 0
    for d in keep_dirs:
        print(f"   {d['name']:<40} {d['size_mb']:>8.1f} MB  ({d['pth_count']} checkpoints)")
        keep_size += d['size_mb']
    print(f"   小计: {keep_size:.1f} MB")
    
    print("\n⚠️ 可归档目录 (中间结果,可选择性保留):")
    archive_size = 0
    for d in archive_dirs:
        print(f"   {d['name']:<40} {d['size_mb']:>8.1f} MB")
        archive_size += d['size_mb']
    print(f"   小计: {archive_size:.1f} MB")
    
    print("\n🗑️ 建议删除目录 (失败实验/测试日志):")
    delete_size = 0
    for d in delete_dirs:
        print(f"   {d['name']:<40} {d['size_mb']:>8.1f} MB")
        delete_size += d['size_mb']
    print(f"   小计: {delete_size:.1f} MB")
    
    print("\n" + "="*80)
    print(f"删除后可释放空间: {delete_size:.1f} MB")
    print(f"归档后可额外释放: {archive_size:.1f} MB")
    print(f"保留空间: {keep_size:.1f} MB")
    print("="*80)
    
    return keep_dirs, archive_dirs, delete_dirs

def generate_cleanup_script(keep_dirs, archive_dirs, delete_dirs):
    """生成清理脚本"""
    
    script_lines = [
        "@echo off",
        "REM work_dirs 清理脚本",
        "REM 生成时间: " + __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "",
        "echo ========================================",
        "echo work_dirs 清理脚本",
        "echo ========================================",
        "echo.",
        "echo 将删除以下目录:",
    ]
    
    for d in delete_dirs:
        script_lines.append(f"echo   - {d['name']} ({d['size_mb']:.1f} MB)")
    
    script_lines.extend([
        "echo.",
        "pause",
        "echo.",
        "echo 开始清理...",
        ""
    ])
    
    for d in delete_dirs:
        script_lines.append(f"echo 删除: {d['name']}")
        script_lines.append(f"rd /s /q \"{d['path']}\" 2>nul")
        script_lines.append("")
    
    delete_size = sum(d['size_mb'] for d in delete_dirs)
    
    script_lines.extend([
        "echo.",
        f"echo 清理完成! 释放空间约 {delete_size:.1f} MB",
        "echo.",
        "pause"
    ])
    
    # 写入脚本
    script_path = 'cleanup_work_dirs.bat'
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(script_lines))
    
    print(f"\n✅ 已生成清理脚本: {script_path}")
    print("   运行命令: cleanup_work_dirs.bat")
    
    # 生成归档脚本
    if archive_dirs:
        archive_lines = [
            "@echo off",
            "REM work_dirs 归档脚本",
            "",
            f"set ARCHIVE_DIR=work_dirs_archive_{__import__('datetime').datetime.now().strftime('%Y%m%d')}",
            "mkdir %ARCHIVE_DIR% 2>nul",
            "echo 归档到: %ARCHIVE_DIR%",
            "echo.",
        ]
        
        for d in archive_dirs:
            archive_lines.append(f"echo 移动: {d['name']}")
            archive_lines.append(f"move \"{d['path']}\" %ARCHIVE_DIR%\\ 2>nul")
            archive_lines.append("")
        
        archive_lines.extend([
            "echo.",
            "echo 归档完成!",
            "pause"
        ])
        
        archive_path = 'archive_work_dirs.bat'
        with open(archive_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(archive_lines))
        
        print(f"✅ 已生成归档脚本: {archive_path}")
        print("   (可选) 运行命令: archive_work_dirs.bat")

if __name__ == '__main__':
    keep, archive, delete = analyze_work_dirs()
    print("\n")
    generate_cleanup_script(keep, archive, delete)
    
    print("\n" + "="*80)
    print("下一步操作:")
    print("="*80)
    print("1. 审查上述建议")
    print("2. 运行 cleanup_work_dirs.bat 删除无用目录")
    print("3. (可选) 运行 archive_work_dirs.bat 归档中间结果")
    print("="*80)
