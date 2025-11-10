import os, random

root = r"C:/LLVIP/LLVIP/ImageSets/Main"
train_path = os.path.join(root, "train.txt")
val_path = os.path.join(root, "val.txt")
test_path = os.path.join(root, "test.txt")

with open(train_path, "r") as f:
    train_list = [x.strip() for x in f.readlines() if x.strip()]

# 去重
train_list = list(dict.fromkeys(train_list))

# 从 train 中抽取 10%
random.shuffle(train_list)
num_test = max(1, int(len(train_list) * 0.1))
test_list = train_list[:num_test]
train_list = train_list[num_test:]

# 写回文件
with open(train_path, "w") as f:
    f.write("\n".join(train_list))
with open(test_path, "w") as f:
    f.write("\n".join(test_list))

print(f"✅ train 剩余: {len(train_list)} 张")
print(f"✅ test 新增: {len(test_list)} 张")

# 验证 val 是否存在
if os.path.exists(val_path):
    with open(val_path, "r") as f:
        val_list = [x.strip() for x in f.readlines() if x.strip()]
    val_list = list(dict.fromkeys(val_list))
    print(f"✅ val 存在: {len(val_list)} 张")
else:
    print("⚠️ 未检测到 val.txt，可从 train 中再划出 10% 作为 val。")

# 路径有效性检查与自动删除
jpeg_dir = os.path.join(root, "..", "..", "visible", "train")
all_files = train_list + test_list
valid_train = []
valid_test = []
removed = []

for sample in train_list:
    img_path = os.path.join(jpeg_dir, sample + ".jpg")
    if os.path.exists(img_path):
        valid_train.append(sample)
    else:
        removed.append(sample)

for sample in test_list:
    img_path = os.path.join(jpeg_dir, sample + ".jpg")
    if os.path.exists(img_path):
        valid_test.append(sample)
    else:
        removed.append(sample)

if removed:
    print(f"⚠️ 检测到 {len(removed)} 个无效样本，已自动删除。")
    # 重新写回文件
    with open(train_path, "w") as f:
        f.write("\n".join(valid_train))
    with open(test_path, "w") as f:
        f.write("\n".join(valid_test))
    print(f"✅ train 更新后: {len(valid_train)} 张")
    print(f"✅ test 更新后: {len(valid_test)} 张")
else:
    print("✅ 所有样本路径有效！")
