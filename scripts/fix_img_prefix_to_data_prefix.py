# VS Code Agent Prompt
# 任务：自动修复 MMDetection 3.x 数据集参数不兼容问题
# 目标：
#   1️⃣ 将 configs/ 目录下的所有 "img_prefix=" 改为 "data_prefix="
#   2️⃣ 同步修改自定义数据集类（LLVIPDataset, KAISTDataset, M3FDDataset）
#       若构造函数仍使用 img_prefix 参数，则替换为 data_prefix
#   3️⃣ 确保所有数据集类的 __getitem__() / load_data_list() 使用 self.data_prefix 调用
#   4️⃣ 进行一次快速验证（DATASETS.build）确保无 TypeError
# 环境：
#   MMDetection 3.x（mmdet=3.0.0, mmcv=2.0.1, mmengine=0.9.1）

import os, re

root_dir = "C:/Users/Xinyu/mmdetection"

# 1️⃣ 扫描 configs 目录中所有 python 配置文件并替换 img_prefix → data_prefix
for dirpath, _, filenames in os.walk(os.path.join(root_dir, "configs")):
    for f in filenames:
        if f.endswith(".py"):
            file_path = os.path.join(dirpath, f)
            with open(file_path, "r", encoding="utf-8") as fr:
                content = fr.read()
            if "img_prefix" in content:
                new_content = re.sub(r"img_prefix\s*=", "data_prefix=", content)
                with open(file_path, "w", encoding="utf-8") as fw:
                    fw.write(new_content)
                print(f"✅ 已修复: {file_path}")

# 2️⃣ 扫描 mmdet/datasets/ 中自定义数据集文件，替换构造签名
for dataset_file in ["llvip_dataset.py", "kaist_dataset.py", "m3fd_dataset.py"]:
    fpath = os.path.join(root_dir, "mmdet", "datasets", dataset_file)
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as fr:
            code = fr.read()

        # 替换构造函数签名
        code = re.sub(r"def __init__\((.*?)img_prefix(.*?)\):",
                      r"def __init__(\1data_prefix\2):", code)
        # 替换内部引用
        code = re.sub(r"self\.img_prefix", "self.data_prefix", code)
        with open(fpath, "w", encoding="utf-8") as fw:
            fw.write(code)
        print(f"✅ 已更新数据集定义: {dataset_file}")

# 3️⃣ 检查是否存在 __getitem__ 或 load_data_list 引用旧变量
for dataset_file in ["llvip_dataset.py", "kaist_dataset.py", "m3fd_dataset.py"]:
    fpath = os.path.join(root_dir, "mmdet", "datasets", dataset_file)
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as fr:
            code = fr.read()
        if "img_prefix" in code:
            print(f"⚠️ 注意: {dataset_file} 中仍存在 'img_prefix'，请手动检查逻辑。")

# 4️⃣ 快速验证数据集注册可用性
try:
    from mmengine.config import Config
    from mmdet.registry import DATASETS

    cfg_path = "C:/Users/Xinyu/mmdetection/configs/llvip/stage2_kaist_domain_ft.py"
    cfg = Config.fromfile(cfg_path)
    dataset = DATASETS.build(cfg.data["train"])
    print("✅ 数据集构建成功，修改完成！")
except Exception as e:
    print("❌ 验证失败，请检查配置文件与数据集定义")
    print(e)
