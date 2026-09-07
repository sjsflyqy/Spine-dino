"""
Spine-DINO 数据集整理脚本
将 F:\AAAzpj\all_dataset 下所有脊柱图片整理为 DINOv2 训练所需格式

输出结构:
  F:\AAAzpj\spine_dino_dataset\
  └── train\
      └── spine\
          ├── BUU2000_AP_00001.png
          ├── ...

自监督训练不需要标签，所有图片放在同一个伪类别目录 "spine" 下。
"""

import os
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict


# ======================== 配置 ========================
SOURCE_DIR = r"F:\AAAzpj\all_dataset"
OUTPUT_DIR = r"F:\AAAzpj\spine_dino_dataset"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# zll 各子目录图片不重复，全量收录

# =====================================================


def get_file_hash(filepath, chunk_size=8192):
    """计算文件 MD5，用于去重"""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def collect_images(source_dir):
    """
    遍历 source_dir，收集所有图片路径。
    对 zll 目录特殊处理：只保留 ap/la，跳过 test*/train/val（避免重复）。
    返回: list of (src_path, prefix_tag)
    """
    source = Path(source_dir)
    collected = []

    for dataset_dir in sorted(source.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        print(f"\n扫描数据集: {dataset_name}")

        # 所有数据集统一处理：递归扫描全部图片
        imgs = _find_images(dataset_dir)
        print(f"  合计: {len(imgs)} 张")
        for p in imgs:
            rel = p.relative_to(dataset_dir)
            parts = rel.parts
            sub_tag = parts[0] if len(parts) > 1 else "root"
            collected.append((p, f"{dataset_name}_{sub_tag}"))

    return collected


def _find_images(directory):
    """递归找出目录下所有图片"""
    result = []
    for p in Path(directory).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            result.append(p)
    return sorted(result)


def deduplicate(collected):
    """
    基于文件 MD5 去重。
    返回去重后的列表，并报告重复数量。
    """
    print("\n正在去重（计算文件哈希）...")
    seen_hashes = {}
    unique = []
    dup_count = 0

    for i, (path, tag) in enumerate(collected):
        if i % 500 == 0:
            print(f"  进度: {i}/{len(collected)}")
        fhash = get_file_hash(path)
        if fhash not in seen_hashes:
            seen_hashes[fhash] = path
            unique.append((path, tag))
        else:
            dup_count += 1

    print(f"  原始图片: {len(collected)} 张")
    print(f"  重复图片: {dup_count} 张")
    print(f"  去重后:   {len(unique)} 张")
    return unique


def copy_to_output(unique_images, output_dir):
    """
    将图片复制到输出目录，统一命名为 {tag}_{index:05d}.png 格式。
    """
    out_path = Path(output_dir) / "train" / "spine"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n正在复制图片到: {out_path}")

    # 按 tag 分组计数
    tag_counters = defaultdict(int)
    copied = 0
    failed = 0

    for src_path, tag in unique_images:
        tag_counters[tag] += 1
        idx = tag_counters[tag]
        # 保留原始扩展名
        ext = src_path.suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"
        new_name = f"{tag}_{idx:05d}{ext}"
        dst = out_path / new_name

        try:
            shutil.copy2(src_path, dst)
            copied += 1
        except Exception as e:
            print(f"  复制失败: {src_path} -> {e}")
            failed += 1

        if copied % 1000 == 0 and copied > 0:
            print(f"  已复制: {copied} 张...")

    print(f"\n完成！成功复制: {copied} 张，失败: {failed} 张")
    return out_path, tag_counters


def print_summary(output_path, tag_counters):
    """打印最终统计"""
    total = sum(tag_counters.values())
    print("\n" + "=" * 50)
    print("数据集整理完成！")
    print("=" * 50)
    print(f"输出目录: {output_path}")
    print(f"总图片数: {total} 张")
    print("\n各子集明细:")
    for tag, count in sorted(tag_counters.items()):
        print(f"  {tag:40s}: {count:5d} 张")
    print("=" * 50)
    print("\n下一步: 运行 generate_metadata.py 生成训练所需的元数据文件")


def main():
    print("Spine-DINO 数据集整理")
    print(f"来源: {SOURCE_DIR}")
    print(f"输出: {OUTPUT_DIR}")
    print("=" * 50)

    # 1. 收集所有图片
    collected = collect_images(SOURCE_DIR)
    print(f"\n收集到原始图片: {len(collected)} 张")

    # 2. 去重
    unique = deduplicate(collected)

    # 3. 复制到输出目录
    out_path, tag_counters = copy_to_output(unique, OUTPUT_DIR)

    # 4. 打印统计
    print_summary(out_path, tag_counters)


if __name__ == "__main__":
    main()
