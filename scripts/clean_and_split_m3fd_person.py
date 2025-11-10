import os, random, shutil
import xml.etree.ElementTree as ET

root = r"C:/M3FD"
anno_dir = os.path.join(root, "Annotation")
ir_dir = os.path.join(root, "ir")
vi_dir = os.path.join(root, "vi")
out_root = os.path.join(root, "cleaned_dataset")

os.makedirs(os.path.join(out_root, "Annotations"), exist_ok=True)
os.makedirs(os.path.join(out_root, "JPEGImages"), exist_ok=True)
os.makedirs(os.path.join(out_root, "ImageSets"), exist_ok=True)

def is_person_only(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall("object")
    new_objects = []
    for obj in objects:
        name = obj.find("name").text.lower()
        if name in ["person", "people", "pedestrian"]:
            new_objects.append(obj)
    if len(new_objects) == 0:
        return None
    # 保留仅person目标
    for obj in objects:
        if obj not in new_objects:
            root.remove(obj)
    return tree

xml_files = [f for f in os.listdir(anno_dir) if f.endswith(".xml")]
cleaned_files = []

for xml_file in xml_files:
    xml_path = os.path.join(anno_dir, xml_file)
    new_tree = is_person_only(xml_path)
    if new_tree is None: continue
    new_tree.write(os.path.join(out_root, "Annotations", xml_file))
    base = os.path.splitext(xml_file)[0]
    for src_dir in [ir_dir, vi_dir]:
        src_path = os.path.join(src_dir, base + ".bmp")
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(out_root, "JPEGImages"))
    cleaned_files.append(base)

# 划分 8:1:1
random.shuffle(cleaned_files)
n = len(cleaned_files)
train, val, test = cleaned_files[:int(0.8*n)], cleaned_files[int(0.8*n):int(0.9*n)], cleaned_files[int(0.9*n):]

for subset, files in [("train", train), ("val", val), ("test", test)]:
    with open(os.path.join(out_root, "ImageSets", subset + ".txt"), "w") as f:
        f.write("\n".join(files))
print(f"✅ 清洗完成，共保留 {len(cleaned_files)} 个 person 样本。")
print("划分比例：train=%d, val=%d, test=%d" % (len(train), len(val), len(test)))
