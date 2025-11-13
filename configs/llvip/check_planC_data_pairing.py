"""
Plan C 训练前数据配对检查脚本
================================
运行: python configs/llvip/check_planC_data_pairing.py
"""

import os
import os.path as osp

def check_kaist_pairing():
    """检查KAIST数据集的visible/infrared配对情况"""
    
    data_root = 'C:/KAIST_processed/'
    train_file = osp.join(data_root, 'ImageSets', 'train.txt')
    
    print("="*70)
    print("Plan C 数据配对检查")
    print("="*70)
    
    # 1. 检查train.txt是否存在
    if not osp.exists(train_file):
        print(f"❌ 错误: {train_file} 不存在!")
        return False
    
    # 2. 读取train.txt
    with open(train_file, 'r') as f:
        sample_ids = [line.strip() for line in f if line.strip()]
    
    print(f"✓ train.txt 包含 {len(sample_ids)} 个样本ID")
    
    # 3. 统计模态分布
    visible_count = 0
    lwir_count = 0
    unknown_count = 0
    
    for sid in sample_ids:
        lower = sid.lower()
        if 'visible' in lower:
            visible_count += 1
        elif any(k in lower for k in ['lwir', 'infrared', 'thermal']):
            lwir_count += 1
        else:
            unknown_count += 1
    
    print(f"\n模态分布:")
    print(f"  - Visible 样本: {visible_count}")
    print(f"  - LWIR/Infrared 样本: {lwir_count}")
    print(f"  - 未知模态: {unknown_count}")
    
    # 4. 检查配对率
    if visible_count == 0 or lwir_count == 0:
        print(f"\n❌ 严重错误: 缺少双模态数据!")
        print(f"   MACL需要visible和infrared配对样本才能工作")
        print(f"   当前只有一种模态,Plan C会失败!")
        return False
    
    # 5. 检查物理文件是否存在
    visible_dir = osp.join(data_root, 'visible')
    infrared_dir = osp.join(data_root, 'infrared')
    
    if not osp.exists(visible_dir):
        print(f"\n❌ 错误: {visible_dir} 目录不存在!")
        return False
    
    if not osp.exists(infrared_dir):
        print(f"\n❌ 错误: {infrared_dir} 目录不存在!")
        return False
    
    print(f"\n✓ 模态目录存在:")
    print(f"  - {visible_dir}")
    print(f"  - {infrared_dir}")
    
    # 6. 抽样检查配对情况
    print(f"\n配对抽样检查 (前5对):")
    
    # 提取base_id映射
    base_id_map = {}
    for sid in sample_ids:
        base_id = sid.replace('_visible', '').replace('_lwir', '')
        base_id = base_id.replace('_infrared', '').replace('_thermal', '')
        
        if base_id not in base_id_map:
            base_id_map[base_id] = {'visible': [], 'infrared': []}
        
        lower = sid.lower()
        if 'visible' in lower:
            base_id_map[base_id]['visible'].append(sid)
        elif any(k in lower for k in ['lwir', 'infrared', 'thermal']):
            base_id_map[base_id]['infrared'].append(sid)
    
    # 统计配对情况
    paired_count = 0
    unpaired_count = 0
    
    for i, (base_id, modalities) in enumerate(list(base_id_map.items())[:5]):
        vis = modalities['visible']
        ir = modalities['infrared']
        
        if len(vis) > 0 and len(ir) > 0:
            print(f"  [{i+1}] ✓ 配对成功:")
            print(f"      Visible: {vis[0]}")
            print(f"      Infrared: {ir[0]}")
            
            # 检查物理文件
            vis_path = osp.join(visible_dir, f'{vis[0]}.jpg')
            ir_path = osp.join(infrared_dir, f'{ir[0]}.jpg')
            
            if not osp.exists(vis_path):
                print(f"      ⚠️ 警告: {vis_path} 文件不存在!")
            if not osp.exists(ir_path):
                print(f"      ⚠️ 警告: {ir_path} 文件不存在!")
            
            paired_count += 1
        else:
            unpaired_count += 1
    
    # 7. 全局配对统计
    total_paired = sum(1 for base_id, mods in base_id_map.items() 
                       if len(mods['visible']) > 0 and len(mods['infrared']) > 0)
    
    pairing_rate = total_paired / len(base_id_map) * 100 if len(base_id_map) > 0 else 0
    
    print(f"\n全局配对统计:")
    print(f"  - 唯一场景数: {len(base_id_map)}")
    print(f"  - 完整配对数: {total_paired}")
    print(f"  - 配对率: {pairing_rate:.1f}%")
    
    # 8. 最终判断
    print(f"\n{'='*70}")
    if pairing_rate >= 90:
        print("✅ Plan C 数据配对检查通过!")
        print("   可以启动训练,MACL将正常工作")
        return True
    elif pairing_rate >= 50:
        print("⚠️ 警告: 配对率偏低,但可尝试训练")
        print("   建议监控loss_macl是否正常收敛")
        return True
    else:
        print("❌ Plan C 不可行: 配对率过低!")
        print("   需要重新处理数据集,确保visible/infrared一一对应")
        return False

if __name__ == '__main__':
    success = check_kaist_pairing()
    exit(0 if success else 1)
