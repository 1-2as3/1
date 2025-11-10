"""
KAIST Dataset Cleaning Script
清理任务：
1. 删除无标注的图片对（空XML）
2. 删除不成对的样本（缺少红外或可见光图片）
3. 保留完整且有标注的样本

运行前建议备份数据！
"""

import os
import glob
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

def main():
    root = r"C:\KAIST_processed"
    
    # 检查目录结构
    required_dirs = ["Annotations", "infrared", "visible"]
    for d in required_dirs:
        if not os.path.exists(os.path.join(root, d)):
            print(f"[ERROR] Directory not found: {d}")
            return
    
    print("=" * 80)
    print("KAIST DATASET CLEANING SCRIPT")
    print("=" * 80)
    
    # 统计初始数据
    xmls = glob.glob(os.path.join(root, "Annotations", "*.xml"))
    infrared_imgs = glob.glob(os.path.join(root, "infrared", "*.jpg"))
    visible_imgs = glob.glob(os.path.join(root, "visible", "*.jpg"))
    
    print(f"\n[BEFORE] Initial statistics:")
    print(f"  Total XML files: {len(xmls)}")
    print(f"  Infrared images: {len(infrared_imgs)}")
    print(f"  Visible images: {len(visible_imgs)}")
    
    # 分离 lwir 和 visible 的 XML 文件
    lwir_xmls = [x for x in xmls if "_lwir_" in os.path.basename(x)]
    visible_xmls = [x for x in xmls if "_visible_" in os.path.basename(x)]
    
    print(f"\n[INFO] XML file breakdown:")
    print(f"  LWIR annotations: {len(lwir_xmls)}")
    print(f"  Visible annotations: {len(visible_xmls)}")
    
    # 构建名称集合（不含扩展名）
    lwir_xml_names = set(os.path.splitext(os.path.basename(f))[0] for f in lwir_xmls)
    vis_xml_names = set(os.path.splitext(os.path.basename(f))[0] for f in visible_xmls)
    
    infrared_names = set(os.path.splitext(os.path.basename(f))[0] for f in infrared_imgs)
    visible_names = set(os.path.splitext(os.path.basename(f))[0] for f in visible_imgs)
    
    print(f"\n[STEP 1] Checking for unpaired samples...")
    
    # 统计变量
    deleted_xmls = 0
    deleted_images = 0
    empty_annotations = 0
    
    # 创建备份目录（可选）
    backup_dir = os.path.join(root, "deleted_samples_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    # 处理所有 XML 文件
    for xml_path in xmls:
        xml_name = os.path.splitext(os.path.basename(xml_path))[0]
        should_delete = False
        reason = ""
        
        # 判断是 lwir 还是 visible XML
        if "_lwir_" in xml_name:
            # 检查对应的红外图片是否存在
            if xml_name not in infrared_names:
                should_delete = True
                reason = "Missing infrared image"
            
            # 检查配对的可见光图片是否存在
            vis_name = xml_name.replace("_lwir_", "_visible_")
            if vis_name not in visible_names:
                should_delete = True
                reason = "Missing paired visible image"
                
        elif "_visible_" in xml_name:
            # 检查对应的可见光图片是否存在
            if xml_name not in visible_names:
                should_delete = True
                reason = "Missing visible image"
            
            # 检查配对的红外图片是否存在
            ir_name = xml_name.replace("_visible_", "_lwir_")
            if ir_name not in infrared_names:
                should_delete = True
                reason = "Missing paired infrared image"
        
        # 检查 XML 是否为空（无标注对象）
        if not should_delete:
            try:
                tree = ET.parse(xml_path)
                objects = tree.findall(".//object")
                
                if len(objects) == 0:
                    should_delete = True
                    reason = "Empty annotation (no objects)"
                    empty_annotations += 1
            except Exception as e:
                print(f"[WARNING] Failed to parse {xml_name}: {e}")
                should_delete = True
                reason = "XML parsing error"
        
        # 删除不符合条件的样本
        if should_delete:
            # 备份到 backup 目录（可选）
            # shutil.move(xml_path, os.path.join(backup_dir, os.path.basename(xml_path)))
            
            # 直接删除
            os.remove(xml_path)
            deleted_xmls += 1
            
            # 删除对应的图片（如果存在）
            if "_lwir_" in xml_name:
                ir_img = os.path.join(root, "infrared", f"{xml_name}.jpg")
                vis_name = xml_name.replace("_lwir_", "_visible_")
                vis_img = os.path.join(root, "visible", f"{vis_name}.jpg")
                
                if os.path.exists(ir_img):
                    os.remove(ir_img)
                    deleted_images += 1
                if os.path.exists(vis_img):
                    os.remove(vis_img)
                    deleted_images += 1
                    
            elif "_visible_" in xml_name:
                vis_img = os.path.join(root, "visible", f"{xml_name}.jpg")
                ir_name = xml_name.replace("_visible_", "_lwir_")
                ir_img = os.path.join(root, "infrared", f"{ir_name}.jpg")
                
                if os.path.exists(vis_img):
                    os.remove(vis_img)
                    deleted_images += 1
                if os.path.exists(ir_img):
                    os.remove(ir_img)
                    deleted_images += 1
            
            if deleted_xmls % 100 == 0:
                print(f"  Processed {deleted_xmls} problematic samples...")
    
    # 统计清理后的数据
    xmls_after = glob.glob(os.path.join(root, "Annotations", "*.xml"))
    infrared_after = glob.glob(os.path.join(root, "infrared", "*.jpg"))
    visible_after = glob.glob(os.path.join(root, "visible", "*.jpg"))
    
    print(f"\n[AFTER] Cleaned statistics:")
    print(f"  Total XML files: {len(xmls_after)}")
    print(f"  Infrared images: {len(infrared_after)}")
    print(f"  Visible images: {len(visible_after)}")
    
    print(f"\n[SUMMARY] Cleaning results:")
    print(f"  Deleted XML files: {deleted_xmls}")
    print(f"  Deleted images: {deleted_images}")
    print(f"  Empty annotations found: {empty_annotations}")
    print(f"  Remaining samples: {len(xmls_after) // 2} pairs")
    
    # 验证清理后的数据完整性
    print(f"\n[VERIFY] Data integrity check:")
    lwir_after = len([x for x in xmls_after if "_lwir_" in x])
    vis_after = len([x for x in xmls_after if "_visible_" in x])
    
    if lwir_after == vis_after == len(infrared_after) == len(visible_after):
        print(f"  [OK] All samples are properly paired ({lwir_after} pairs)")
    else:
        print(f"  [WARNING] Mismatch detected:")
        print(f"    LWIR XMLs: {lwir_after}")
        print(f"    Visible XMLs: {vis_after}")
        print(f"    Infrared images: {len(infrared_after)}")
        print(f"    Visible images: {len(visible_after)}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] KAIST dataset cleaning completed!")
    print("=" * 80)

if __name__ == "__main__":
    # 安全确认
    print("WARNING: This script will DELETE files from C:\\KAIST_processed\\")
    print("Please ensure you have a backup before proceeding!")
    response = input("\nDo you want to continue? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        main()
    else:
        print("Operation cancelled.")
