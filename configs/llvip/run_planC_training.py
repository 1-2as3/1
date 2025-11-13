"""
Plan C 训练启动与实时监控脚本
================================

功能:
1. 启动训练
2. 实时监控关键指标
3. 自动判断是否需要中断

使用方法:
    python configs/llvip/run_planC_training.py
"""

import subprocess
import sys
import time
import os
import os.path as osp
import re

class PlanCMonitor:
    """Plan C训练监控器"""
    
    def __init__(self, work_dir='./work_dirs/stage2_2_planC_dualmodality_macl'):
        self.work_dir = work_dir
        self.current_epoch = 0
        self.current_map = 0.0
        self.loss_macl_found = False
        self.fail_criteria_met = False
        
    def parse_log_line(self, line):
        """解析训练日志行"""
        
        # 检测loss_macl是否出现
        if 'loss_macl' in line and not self.loss_macl_found:
            self.loss_macl_found = True
            print("\n" + "="*70)
            print("✅ 关键里程碑: loss_macl 已出现!")
            print("   MACL对比学习正在工作,双模态配对成功")
            print("="*70 + "\n")
        
        # 解析mAP
        match = re.search(r'pascal_voc/mAP:\s+([\d.]+)', line)
        if match:
            self.current_map = float(match.group(1))
            
        # 解析epoch
        match = re.search(r'Epoch\((?:train|val)\)\s+\[(\d+)\]', line)
        if match:
            self.current_epoch = int(match.group(1))
    
    def check_fail_criteria(self):
        """检查失败判定条件"""
        
        if self.current_epoch == 1 and self.current_map > 0:
            if self.current_map < 0.52:
                print("\n" + "🔴"*35)
                print("❌ 失败判定: Epoch 1 mAP={:.4f} < 0.52".format(self.current_map))
                print("   训练已彻底崩溃 (梯度错向/特征漂移)")
                print("   建议: 停止训练,检查配置或降低lambda1")
                print("🔴"*35 + "\n")
                self.fail_criteria_met = True
                return True
            
            elif self.current_map < 0.55:
                print("\n" + "⚠️"*35)
                print("⚠️ 警告: Epoch 1 mAP={:.4f} < 0.55".format(self.current_map))
                print("   表现欠佳,但可通过调参救回")
                print("   建议: 若Epoch 2无改善,调整lambda1=0.005或lr=3e-5")
                print("⚠️"*35 + "\n")
            
            elif self.current_map >= 0.55:
                print("\n" + "✅"*35)
                print("✅ 成功: Epoch 1 mAP={:.4f} ≥ 0.55".format(self.current_map))
                print("   训练方向正确,继续监控!")
                print("✅"*35 + "\n")
        
        elif self.current_epoch == 2 and self.current_map > 0:
            if self.current_map < 0.55:
                print("\n" + "🔴"*35)
                print("❌ 失败判定: Epoch 2 mAP={:.4f} < 0.55".format(self.current_map))
                print("   连续2个epoch低迷,需要调整策略")
                print("   建议: lambda1减半至0.005,或提高lr至8e-5")
                print("🔴"*35 + "\n")
                self.fail_criteria_met = True
                return True
        
        elif self.current_epoch == 3 and self.current_map > 0:
            if self.current_map < 0.58:
                print("\n" + "⚠️"*35)
                print("⚠️ 警告: Epoch 3 mAP={:.4f} < 0.58".format(self.current_map))
                print("   Plan C可能无法达到目标 (0.60+)")
                print("   建议: 考虑切换到Plan D或E")
                print("⚠️"*35 + "\n")
        
        # 检查loss_macl是否缺失
        if self.current_epoch >= 1 and not self.loss_macl_found:
            print("\n" + "🔴"*35)
            print("❌ 严重错误: Epoch 1完成但loss_macl未出现!")
            print("   可能原因:")
            print("   1. return_modality_pair未生效")
            print("   2. PairedDetDataPreprocessor未启用")
            print("   3. 模型加载时MACL head未正确初始化")
            print("   建议: 立即停止,检查配置")
            print("🔴"*35 + "\n")
            self.fail_criteria_met = True
            return True
        
        return False

def run_training_with_monitor():
    """启动训练并实时监控"""
    
    config_file = 'configs/llvip/stage2_2_planC_dualmodality_macl.py'
    
    print("="*70)
    print("Plan C 训练启动")
    print("="*70)
    print(f"配置文件: {config_file}")
    print(f"工作目录: ./work_dirs/stage2_2_planC_dualmodality_macl")
    print("\n关键监控指标:")
    print("  1. loss_macl 是否出现 (必须在前100 iter内出现)")
    print("  2. Epoch 1 mAP ≥ 0.55 (低于0.52=失败)")
    print("  3. grad_norm < 15 (正常范围)")
    print("="*70 + "\n")
    
    # 启动训练进程
    cmd = [sys.executable, 'tools/train.py', config_file]
    
    monitor = PlanCMonitor()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时读取输出
        for line in process.stdout:
            # 打印原始日志
            print(line, end='')
            sys.stdout.flush()
            
            # 解析关键指标
            monitor.parse_log_line(line)
            
            # 检查失败条件
            if monitor.check_fail_criteria():
                print("\n⚠️ 监控器建议中断训练,是否继续? (按Ctrl+C停止)")
                # 不自动杀进程,让用户决定
        
        # 等待进程结束
        return_code = process.wait()
        
        print("\n" + "="*70)
        print(f"训练进程结束,返回码: {return_code}")
        print("="*70)
        
        return return_code
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断训练")
        process.terminate()
        process.wait()
        return -1
    except Exception as e:
        print(f"\n❌ 训练过程出错: {e}")
        return -1

if __name__ == '__main__':
    # 检查配置文件是否存在
    config_file = 'configs/llvip/stage2_2_planC_dualmodality_macl.py'
    if not osp.exists(config_file):
        print(f"❌ 错误: 配置文件不存在: {config_file}")
        sys.exit(1)
    
    # 启动训练
    exit_code = run_training_with_monitor()
    sys.exit(exit_code)
