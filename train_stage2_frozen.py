"""
Stage2 训练启动脚本（带 Backbone 真正冻结）

使用方法：
    python train_stage2_frozen.py

说明：
    1. 加载 Stage2 配置
    2. 构建模型后显式冻结 Backbone 参数
    3. 验证冻结状态
    4. 启动训练
"""
import sys
import os
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules

def freeze_backbone(model):
    """显式冻结 Backbone 所有参数"""
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
            frozen_count += 1
    return frozen_count

def verify_freeze_status(model):
    """验证冻结状态"""
    backbone_params = {'total': 0, 'trainable': 0, 'frozen': 0}
    other_params = {'total': 0, 'trainable': 0, 'frozen': 0}
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params['total'] += param.numel()
            if param.requires_grad:
                backbone_params['trainable'] += param.numel()
            else:
                backbone_params['frozen'] += param.numel()
        else:
            other_params['total'] += param.numel()
            if param.requires_grad:
                other_params['trainable'] += param.numel()
            else:
                other_params['frozen'] += param.numel()
    
    return backbone_params, other_params

def main():
    print("=" * 80)
    print("Stage2 训练（带 Backbone 冻结）")
    print("=" * 80)
    
    # 1. 注册所有模块
    print("\n[1/5] 注册模块...")
    register_all_modules(init_default_scope=True)
    
    # 2. 加载配置
    print("[2/5] 加载配置...")
    cfg_path = 'configs/llvip/stage2_kaist_domain_ft.py'
    cfg = Config.fromfile(cfg_path)
    
    # 检查 load_from 是否存在
    if 'load_from' in cfg and cfg.load_from:
        if not os.path.exists(cfg.load_from):
            print(f"\n⚠️  警告：预训练权重文件不存在: {cfg.load_from}")
            print("   训练将从 ImageNet 预训练开始")
            response = input("   是否继续？(y/n): ")
            if response.lower() != 'y':
                print("   训练已取消")
                return
    
    # 3. 创建 Runner
    print("[3/5] 创建 Runner...")
    runner = Runner.from_cfg(cfg)
    
    # 4. 冻结 Backbone
    print("[4/5] 冻结 Backbone 参数...")
    frozen_count = freeze_backbone(runner.model)
    print(f"   已冻结 {frozen_count} 个 Backbone 参数")
    
    # 5. 验证冻结状态
    print("[5/5] 验证冻结状态...")
    backbone_params, other_params = verify_freeze_status(runner.model)
    
    print(f"\n   Backbone:")
    print(f"     总参数: {backbone_params['total']:,} ({backbone_params['total']/1e6:.2f}M)")
    print(f"     可训练: {backbone_params['trainable']:,} ({backbone_params['trainable']/1e6:.2f}M)")
    print(f"     冻结:   {backbone_params['frozen']:,} ({backbone_params['frozen']/1e6:.2f}M)")
    
    print(f"\n   其他模块:")
    print(f"     总参数: {other_params['total']:,} ({other_params['total']/1e6:.2f}M)")
    print(f"     可训练: {other_params['trainable']:,} ({other_params['trainable']/1e6:.2f}M)")
    print(f"     冻结:   {other_params['frozen']:,} ({other_params['frozen']/1e6:.2f}M)")
    
    # 检查是否真正冻结
    if backbone_params['trainable'] > 0:
        print("\n❌ 错误：Backbone 仍有可训练参数！")
        print("   请检查冻结逻辑")
        return
    else:
        print("\n✓ Backbone 已完全冻结")
    
    # 6. 开始训练
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80 + "\n")
    
    runner.train()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练已中断")
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        import traceback
        traceback.print_exc()
