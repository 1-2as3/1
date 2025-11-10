import os, scipy.io, xml.etree.ElementTree as ET, shutil, random
from tqdm import tqdm

root = r"C:/KAIST"
anno_root = os.path.join(root, "annotations", "annotations")
output_root = r"C:/KAIST_processed"

os.makedirs(os.path.join(output_root, "Annotations"), exist_ok=True)
os.makedirs(os.path.join(output_root, "visible"), exist_ok=True)
os.makedirs(os.path.join(output_root, "infrared"), exist_ok=True)
os.makedirs(os.path.join(output_root, "ImageSets"), exist_ok=True)

def parse_vbb(vbb_path):
    """解析 VBB 文件，返回每帧的 person 框"""
    mat = scipy.io.loadmat(vbb_path)
    obj_lists = mat["A"][0][0][1][0]
    obj_lbls = [str(x[0]) for x in mat["A"][0][0][4][0]]
    person_ids = [i for i, l in enumerate(obj_lbls) if "person" in l.lower()]
    annotations = {}
    for frame_id, objs in enumerate(obj_lists):
        frame_objs = []
        for obj in objs[0]:
            oid = int(obj[0][0][0]) - 1
            if oid not in person_ids:
                continue
            pos = obj[1][0]
            x, y, w, h = pos
            if w < 10 or h < 10:  # 跳过极小目标
                continue
            frame_objs.append([x, y, x + w, y + h])
        if frame_objs:
            annotations[frame_id] = frame_objs
    return annotations

def save_voc(xml_path, img_name, boxes, size=(640, 512, 3)):
    ann = ET.Element("annotation")
    ET.SubElement(ann, "filename").text = img_name
    size_node = ET.SubElement(ann, "size")
    for n,v in zip(["width","height","depth"], size):
        ET.SubElement(size_node, n).text = str(v)
    for box in boxes:
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = "person"
        bnd = ET.SubElement(obj, "bndbox")
        for k,v in zip(["xmin","ymin","xmax","ymax"], box):
            ET.SubElement(bnd, k).text = str(int(v))
    ET.ElementTree(ann).write(xml_path)

def find_seq_path(set_name, seq_name):
    """根据 set 名和序列名找到实际路径"""
    candidates = [
        os.path.join(root, set_name, seq_name),
        os.path.join(root, set_name + "_A", seq_name),
        os.path.join(root, set_name + "_B", seq_name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def process_set(set_name):
    set_dir = os.path.join(anno_root, set_name)
    if not os.path.exists(set_dir):
        return
    for vbb_file in os.listdir(set_dir):
        if not vbb_file.endswith(".vbb"):
            continue
        seq_name = vbb_file.replace(".vbb", "")
        seq_path = find_seq_path(set_name, seq_name)
        if seq_path is None:
            continue
        vbb_path = os.path.join(set_dir, vbb_file)
        annotations = parse_vbb(vbb_path)
        for frame_id, boxes in annotations.items():
            fname = f"I{frame_id:05d}.jpg"
            vis_src = os.path.join(seq_path, "visible", fname)
            ir_src = os.path.join(seq_path, "lwir", fname)

            # 仅当两种模态都存在时才保留该样本
            if not (os.path.exists(vis_src) and os.path.exists(ir_src)):
                continue

            # 复制可见光
            vis_name = f"{set_name}_{seq_name}_visible_{fname}"
            vis_dst = os.path.join(output_root, "visible", vis_name)
            shutil.copy(vis_src, vis_dst)
            vis_xml = os.path.join(output_root, "Annotations", vis_name.replace(".jpg", ".xml"))
            save_voc(vis_xml, vis_name, boxes)

            # 复制红外
            ir_name = f"{set_name}_{seq_name}_lwir_{fname}"
            ir_dst = os.path.join(output_root, "infrared", ir_name)
            shutil.copy(ir_src, ir_dst)
            ir_xml = os.path.join(output_root, "Annotations", ir_name.replace(".jpg", ".xml"))
            save_voc(ir_xml, ir_name, boxes)
    print(f"✅ Finished: {set_name}")

sets = [f"set{str(i).zfill(2)}" for i in range(12)]
for s in tqdm(sets):
    process_set(s)

# 按场景划分（set00–06: train, set07: val, set08–11: test）
all_xml = os.listdir(os.path.join(output_root, "Annotations"))
train = [x.replace(".xml","") for x in all_xml if any(f"set0{i}" in x for i in range(7))]
val = [x.replace(".xml","") for x in all_xml if "set07" in x]
test = [x.replace(".xml","") for x in all_xml if any(f"set{i}" in x for i in range(8,12))]

for name, items in [("train", train), ("val", val), ("test", test)]:
    with open(os.path.join(output_root, "ImageSets", f"{name}.txt"), "w") as f:
        f.write("\n".join(items))

print(f"✅ 处理完成：train={len(train)}, val={len(val)}, test={len(test)}")
