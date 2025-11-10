#!/usr/bin/env python3
"""测试结果汇总显示效果"""

# ANSI 颜色代码
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# 模拟测试结果
results = {
    'config': True,
    'data_roots': True,
    'checkpoint (可选)': False,  # 这个是可选的，未找到 checkpoint
    'data_probe': True,
    'synthetic_grad': True,
}

print("\n" + "=" * 80)
print("验证结果汇总")
print("=" * 80 + "\n")

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
    print(f"{GREEN}✅ 所有必需验证测试通过！Stage3 环境配置正确。{RESET}")
    print(f"\n{GREEN}可以开始 Stage3 正式训练{RESET}")
else:
    print(f"{RED}❌ 部分测试失败，请检查上方日志{RESET}")

print("=" * 80)
