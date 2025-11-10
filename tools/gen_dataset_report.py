#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动化数据检查与可视化脚本：支持 LLVIP / KAIST / M3FD。

输出：
- ImageSets/Main/train.txt, val.txt  (80/20 划分)
- analysis_report/
  ├── pair_completeness_bar.png
  ├── image_size_hist.png
  ├── sample_distribution.png
  └── summary.json

依赖：matplotlib、os、cv2、json、pathlib、tqdm、numpy
"""
from __future__ import annotations

import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set

# cv2, numpy, matplotlib are optional - only needed for stats/plots
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    
try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 尝试设置中文字体，防止中文标签乱码
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# ===================== 数据集配置（可在此修改默认值） =====================
# 可选值：["LLVIP", "KAIST", "M3FD"]
dataset = "LLVIP"

# 根据不同数据集设定默认路径与目录映射
DEFAULTS = {
    "LLVIP": {
        "data_root": r"C:/LLVIP/LLVIP",
        "vis_dir": "visible",
        "ir_dir": "infrared",
        "ann_dir": "Annotations",
    },
    "KAIST": {
        "data_root": r"C:/KAIST",
        "vis_dir": "visible",
        "ir_dir": "lwir",
        "ann_dir": "Annotations",
    },
    "M3FD": {
        "data_root": r"C:/M3FD",
        "vis_dir": "VIS",
        "ir_dir": "IR",
        "ann_dir": "labels",  # YOLO 格式标签
    },
}

RANDOM_SEED = 3407

# ===================== 工具函数 =====================

def resolve_dataset_paths(ds_name: str, data_root: str|None = None) -> Tuple[Path, str, str, str]:
    assert ds_name in DEFAULTS, f"未知数据集：{ds_name}，可选：{list(DEFAULTS.keys())}"
    d = DEFAULTS[ds_name].copy()
    if data_root:
        d["data_root"] = data_root
    root = Path(d["data_root"]).resolve()
    return root, d["vis_dir"], d["ir_dir"], d["ann_dir"]


def list_images_recursive(root: Path) -> Dict[str, Path]:
    """递归收集图像：返回 {id: path}
    id 定义为相对根目录去除扩展名后的路径（以 / 连接），例如：train/000001
    这样可同时兼容是否存在 train/val 子目录的情况。
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    mapping: Dict[str, Path] = {}
    if not root.exists():
        return mapping
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            rel = p.relative_to(root)
            sample_id = str(rel.with_suffix(""))  # 去掉扩展名
            # 正规化分隔符
            sample_id = sample_id.replace("\\", "/")
            mapping[sample_id] = p
    return mapping


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ===================== 1) 生成 VOC 索引文件 =====================

def generate_index_files(data_root: Path, ids: List[str]) -> Tuple[List[str], List[str]]:
    """生成 ImageSets/Main/train.txt 与 val.txt (80/20)。
    返回 (train_ids, val_ids)。
    """
    random.seed(RANDOM_SEED)
    ids_sorted = sorted(ids)
    random.shuffle(ids_sorted)
    n_total = len(ids_sorted)
    n_train = int(0.8 * n_total)
    train_ids = ids_sorted[:n_train]
    val_ids = ids_sorted[n_train:]

    main_dir = data_root / "ImageSets" / "Main"
    ensure_dir(main_dir)

    def _write_list(fp: Path, id_list: List[str]):
        with fp.open("w", encoding="utf-8") as f:
            for sid in id_list:
                # 只写 basename（不带扩展名、不含子目录），与标准 VOC 对齐
                basename = sid.split("/")[-1]
                f.write(basename + "\n")

    _write_list(main_dir / "train.txt", train_ids)
    _write_list(main_dir / "val.txt", val_ids)
    return train_ids, val_ids


# ===================== 2) 校验配对完整性 =====================

def check_pair_completeness(vis_map: Dict[str, Path], ir_map: Dict[str, Path]) -> Tuple[Set[str], Dict[str, List[str]]]:
    """检查配对完整性：
    返回：
      - paired_ids: 同时存在于可见光与红外的样本 ID 集合（以相对路径 id 定义）
      - missing_report: {"missing_vis": [...], "missing_ir": [...]} 缺失样本 ID 列表
    注意：这里以相对路径 id 进行匹配，可兼容 train/val 或其它子目录结构。
    """
    vis_ids = set(vis_map.keys())
    ir_ids = set(ir_map.keys())
    # 用 basename 进行匹配，最大化配对可能（避免子目录不一致）
    vis_base_to_ids: Dict[str, List[str]] = {}
    for sid in vis_ids:
        b = sid.split("/")[-1]
        vis_base_to_ids.setdefault(b, []).append(sid)
    ir_base_to_ids: Dict[str, List[str]] = {}
    for sid in ir_ids:
        b = sid.split("/")[-1]
        ir_base_to_ids.setdefault(b, []).append(sid)

    common_basenames = set(vis_base_to_ids.keys()) & set(ir_base_to_ids.keys())
    paired_ids: Set[str] = set()
    missing_vis: List[str] = []
    missing_ir: List[str] = []

    # 以 basename 为核心匹配；如出现一对多，这里仅取第一对
    for b in sorted(common_basenames):
        v_id = vis_base_to_ids[b][0]
        i_id = ir_base_to_ids[b][0]
        # 取统一的 ID（用 basename）
        paired_ids.add(b)

    # 统计缺失（以 basename 维度）
    all_basenames = set(list(vis_base_to_ids.keys()) + list(ir_base_to_ids.keys()))
    for b in sorted(all_basenames):
        if b not in vis_base_to_ids:
            missing_vis.append(b)
        if b not in ir_base_to_ids:
            missing_ir.append(b)

    missing_report = {"missing_vis": missing_vis, "missing_ir": missing_ir}
    return paired_ids, missing_report


# ===================== 3) 统计图像尺寸 =====================

def _read_image_size(p: Path) -> Tuple[int, int] | None:
    img = cv2.imread(str(p))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def compute_image_stats(vis_map: Dict[str, Path], ir_map: Dict[str, Path], paired_basenames: Set[str]) -> Dict[str, dict]:
    """计算图像尺寸统计，返回各模态的统计 dict。
    stats = {
      'visible': {'count': int, 'avg_w': float, 'avg_h': float, 'widths': [..], 'heights': [..]},
      'infrared': {...}
    }
    仅统计可成功读取的图像。
    """
    def collect_sizes(path_map: Dict[str, Path], filter_basenames: Set[str]) -> Tuple[List[int], List[int]]:
        ws, hs = [], []
        for sid, p in tqdm(path_map.items(), desc="读取尺寸", leave=False):
            b = sid.split("/")[-1]
            if b not in filter_basenames:
                continue
            wh = _read_image_size(p)
            if wh is not None:
                w, h = wh
                ws.append(w)
                hs.append(h)
        return ws, hs

    v_ws, v_hs = collect_sizes(vis_map, paired_basenames)
    i_ws, i_hs = collect_sizes(ir_map, paired_basenames)

    def summarize(ws: List[int], hs: List[int]) -> Dict[str, float | int | List[int]]:
        if len(ws) == 0:
            return {"count": 0, "avg_w": 0, "avg_h": 0, "widths": [], "heights": []}
        return {
            "count": len(ws),
            "avg_w": float(np.mean(ws)),
            "avg_h": float(np.mean(hs)),
            "widths": ws,
            "heights": hs,
        }

    return {
        "visible": summarize(v_ws, v_hs),
        "infrared": summarize(i_ws, i_hs),
    }


# ===================== 4) 可视化绘图 =====================

def plot_visualizations(output_dir: Path,
                        total_count: int,
                        train_count: int,
                        val_count: int,
                        pair_rate: float,
                        vis_rate: float,
                        ir_rate: float,
                        stats: Dict[str, dict]) -> None:
    ensure_dir(output_dir)

    # a) 配对完整率柱状图
    plt.figure(figsize=(6, 4))
    labels = ["可见光/Visible", "红外/Infrared", "配对/Paired"]
    rates = [vis_rate * 100.0, ir_rate * 100.0, pair_rate * 100.0]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    plt.bar(labels, rates, color=colors)
    plt.ylim(0, 100)
    plt.ylabel("完整率 / Completeness (%)")
    plt.title("配对完整率 / Pair Completeness")
    plt.tight_layout()
    plt.savefig(output_dir / "pair_completeness_bar.png", dpi=150)
    plt.close()

    # b) 图像尺寸直方图（宽&高，两个子图，区分模态）
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = 30
    # 宽度
    axes[0].hist(stats["visible"].get("widths", []), bins=bins, alpha=0.6, label="Visible", color="#4C78A8")
    axes[0].hist(stats["infrared"].get("widths", []), bins=bins, alpha=0.6, label="Infrared", color="#F58518")
    axes[0].set_title("宽度分布 / Width Distribution")
    axes[0].set_xlabel("宽度 / Width")
    axes[0].set_ylabel("频数 / Count")
    axes[0].legend()
    # 高度
    axes[1].hist(stats["visible"].get("heights", []), bins=bins, alpha=0.6, label="Visible", color="#4C78A8")
    axes[1].hist(stats["infrared"].get("heights", []), bins=bins, alpha=0.6, label="Infrared", color="#F58518")
    axes[1].set_title("高度分布 / Height Distribution")
    axes[1].set_xlabel("高度 / Height")
    axes[1].set_ylabel("频数 / Count")
    axes[1].legend()
    fig.suptitle("图像分辨率直方图 / Image Size Histogram", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "image_size_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # c) 训练/验证样本数量直方图
    plt.figure(figsize=(6, 4))
    labels = ["训练/Train", "验证/Val"]
    counts = [train_count, val_count]
    plt.bar(labels, counts, color=["#72B7B2", "#E45756"])
    plt.ylabel("样本数 / Samples")
    plt.title("样本划分 / Sample Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "sample_distribution.png", dpi=150)
    plt.close()


# ===================== 5) 保存 summary.json =====================

def save_summary_json(output_dir: Path, summary: dict) -> None:
    ensure_dir(output_dir)
    fp = output_dir / "summary.json"
    with fp.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(description="生成数据集报告与索引文件 (LLVIP/KAIST/M3FD)")
    parser.add_argument("--dataset", type=str, default=dataset, choices=list(DEFAULTS.keys()), help="数据集名称")
    parser.add_argument("--data-root", type=str, default=None, help="数据集根目录，默认按 dataset 选择的预设路径")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个样本（调试用）")
    parser.add_argument("--skip-stats", action="store_true", help="跳过图像尺寸统计以加速，仅生成索引与概览")
    parser.add_argument("--skip-plots", action="store_true", help="跳过绘图生成，仅输出 summary.json")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"], help="指定要扫描的数据分割：train, test, 或 all")
    args = parser.parse_args()

    # 解析路径
    data_root, vis_dir, ir_dir, ann_dir = resolve_dataset_paths(args.dataset, args.data_root)
    
    # 根据split参数调整路径
    if args.split != "all" and args.dataset == "LLVIP":
        # LLVIP的visible和infrared下有train/test子目录
        vis_dir = str(Path(vis_dir) / args.split)
        ir_dir = str(Path(ir_dir) / args.split)
    
    vis_root = (data_root / vis_dir)
    ir_root = (data_root / ir_dir)
    ann_root = (data_root / ann_dir)

    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] data_root: {data_root}")
    print(f"[INFO] vis_dir: {vis_root}")
    print(f"[INFO] ir_dir:  {ir_root}")
    print(f"[INFO] ann_dir: {ann_root}")

    # 0) 基础路径检查
    if not vis_root.exists():
        print(f"[WARN] 可见光目录不存在: {vis_root}")
    if not ir_root.exists():
        print(f"[WARN] 红外目录不存在:   {ir_root}")
    if not ann_root.exists():
        print(f"[WARN] 标注目录不存在:   {ann_root}")

    # 扫描图像
    print("[INFO] 扫描可见光图像...")
    vis_map = list_images_recursive(vis_root)
    print(f"[OK] 可见光图像数: {len(vis_map)}")

    print("[INFO] 扫描红外图像...")
    ir_map = list_images_recursive(ir_root)
    print(f"[OK] 红外图像数: {len(ir_map)}")

    # 2) 配对完整性
    paired_basenames, missing_report = check_pair_completeness(vis_map, ir_map)

    # 1) 生成索引文件（基于配对 basenames，确保训练可用）
    all_basenames = sorted(list(set(list(vis_map.keys()) + list(ir_map.keys()))))
    # 写入索引时仅写 basename；因此取 paired_basenames 为主
    # 如需要包含未配对的样本，也可更改为 all_basenames
    selected_ids = sorted(list(paired_basenames))
    if args.limit is not None:
        selected_ids = selected_ids[:args.limit]
    train_ids, val_ids = generate_index_files(data_root, selected_ids)

    # 3) 统计
    total_count = len(selected_ids)
    train_count = len(train_ids)
    val_count = len(val_ids)

    # 完整率（以 basenames 维度）
    total_basenames = len(set(list(vis_map.keys()) + list(ir_map.keys())))
    vis_rate = len(set([k.split("/")[-1] for k in vis_map.keys()])) / max(total_basenames, 1)
    ir_rate = len(set([k.split("/")[-1] for k in ir_map.keys()])) / max(total_basenames, 1)
    pair_rate = len(paired_basenames) / max(total_basenames, 1)

    # 图像尺寸统计（仅对配对样本进行统计，便于对齐）
    if args.skip_stats or not HAS_CV2 or not HAS_NUMPY:
        if not args.skip_stats and (not HAS_CV2 or not HAS_NUMPY):
            print("[WARN] cv2 or numpy not available, skipping stats")
        stats = {
            "visible": {"count": 0, "avg_w": 0, "avg_h": 0, "widths": [], "heights": []},
            "infrared": {"count": 0, "avg_w": 0, "avg_h": 0, "widths": [], "heights": []},
        }
    else:
        stats = compute_image_stats(vis_map, ir_map, paired_basenames)

    # 4) 可视化
    out_dir = data_root / "analysis_report"
    if not args.skip_plots and HAS_MATPLOTLIB:
        plot_visualizations(out_dir, total_count, train_count, val_count, pair_rate, vis_rate, ir_rate, stats)
    elif not args.skip_plots:
        print("[WARN] matplotlib not available, skipping plots")

    # 5) summary.json
    summary = {
        "dataset": args.dataset,
        "data_root": str(data_root),
        "total_samples": total_count,
        "train_samples": train_count,
        "val_samples": val_count,
        "pair_completeness": pair_rate,
        "visible_presence_rate": vis_rate,
        "infrared_presence_rate": ir_rate,
        "avg_visible_size": [stats["visible"].get("avg_w", 0), stats["visible"].get("avg_h", 0)],
        "avg_infrared_size": [stats["infrared"].get("avg_w", 0), stats["infrared"].get("avg_h", 0)],
        "missing_counts": {
            "missing_vis": len(missing_report.get("missing_vis", [])),
            "missing_ir": len(missing_report.get("missing_ir", [])),
        },
        "notes": "索引文件按 80/20 随机划分；仅将配对样本写入 train/val 列表，以保障跨模态训练。",
    }
    save_summary_json(out_dir, summary)

    # 6) 控制台报告
    print("\n================ 数据集报告 ===============")
    print(f"[OK] Total samples: {total_count}")
    print(f"[OK] Train: {train_count}, Val: {val_count}")
    print(f"[OK] Paired samples: {len(paired_basenames)} / {total_basenames} ({pair_rate*100:.1f}%)")
    if missing_report.get("missing_ir"):
        print(f"[WARN] Missing infrared images: {len(missing_report['missing_ir'])}")
    if missing_report.get("missing_vis"):
        print(f"[WARN] Missing visible images: {len(missing_report['missing_vis'])}")
    print(f"[OK] Average visible size: {int(stats['visible'].get('avg_w',0))}x{int(stats['visible'].get('avg_h',0))}")
    print(f"[OK] Average infrared size: {int(stats['infrared'].get('avg_w',0))}x{int(stats['infrared'].get('avg_h',0))}")
    print(f"[OK] 报告与图表输出目录: {out_dir}")


if __name__ == "__main__":
    main()
