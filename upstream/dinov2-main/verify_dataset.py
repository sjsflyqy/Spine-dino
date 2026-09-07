"""
Spine-DINO 数据集验证脚本
检查整理后的数据集是否可以被正常读取，
以及图片质量是否符合训练要求。
"""

import argparse
import sys
import random
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "SpinePretrain-v1" / "spine_dino_dataset"
SAMPLE_CHECK_COUNT = 500   # 随机抽查的图片数量
MIN_SIZE = 64              # 图片最小边长（像素），低于此视为异常
TARGET_SIZE = 518          # RAD-DINO 训练尺寸，用于统计尺寸分布


def verify_dataset(dataset_root, sample_count=500):
    root = Path(dataset_root)
    train_dir = root / "train" / "spine"

    if not train_dir.exists():
        print(f"错误：目录不存在 {train_dir}")
        print("请先运行 prepare_spine_dataset.py")
        sys.exit(1)

    # 获取所有图片
    all_images = sorted(list(train_dir.glob("*")))
    all_images = [p for p in all_images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    total = len(all_images)
    print(f"找到图片总数: {total} 张")

    if total == 0:
        print("错误：没有找到任何图片！")
        sys.exit(1)

    # 随机抽样检查
    sample = random.sample(all_images, min(sample_count, total))
    print(f"\n随机抽查 {len(sample)} 张图片...")

    corrupted = []
    too_small = []
    sizes = []
    modes = {}

    for img_path in tqdm(sample, desc="验证中"):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                mode = img.mode
                sizes.append((w, h))
                modes[mode] = modes.get(mode, 0) + 1

                if min(w, h) < MIN_SIZE:
                    too_small.append((img_path.name, w, h))
        except (UnidentifiedImageError, Exception) as e:
            corrupted.append((img_path.name, str(e)))

    # 统计结果
    print("\n" + "=" * 60)
    print("验证报告")
    print("=" * 60)
    print(f"总图片数:     {total}")
    print(f"抽查数量:     {len(sample)}")
    print(f"损坏图片:     {len(corrupted)}")
    print(f"过小图片:     {len(too_small)} (<{MIN_SIZE}px)")

    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"\n图片尺寸统计（抽样）:")
        print(f"  宽度  — min:{min(widths)}, max:{max(widths)}, 均值:{sum(widths)//len(widths)}")
        print(f"  高度  — min:{min(heights)}, max:{max(heights)}, 均值:{sum(heights)//len(heights)}")
        small_count = sum(1 for w, h in sizes if w < TARGET_SIZE or h < TARGET_SIZE)
        print(f"  小于 {TARGET_SIZE}px 的图片: {small_count}/{len(sizes)} ({100*small_count//len(sizes)}%)")

    print(f"\n图片模式分布:")
    for mode, count in sorted(modes.items(), key=lambda x: -x[1]):
        pct = 100 * count // len(sample)
        print(f"  {mode:10s}: {count:4d} 张 ({pct}%)")

    # 警告
    if corrupted:
        print(f"\n⚠ 损坏图片示例（前5个）:")
        for name, err in corrupted[:5]:
            print(f"  {name}: {err}")

    if too_small:
        print(f"\n⚠ 过小图片示例（前5个）:")
        for name, w, h in too_small[:5]:
            print(f"  {name}: {w}×{h}")

    if len(corrupted) == 0 and len(too_small) == 0:
        print("\n✓ 数据集验证通过，可以开始训练！")
    else:
        print(f"\n⚠ 建议处理上述问题后再训练。")

    # 前缀分布（来自哪个子数据集）
    prefix_counts = {}
    for p in all_images:
        prefix = "_".join(p.stem.split("_")[:2])
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    
    print(f"\n各子数据集图片数量:")
    for prefix, count in sorted(prefix_counts.items()):
        bar = "█" * (count // 100)
        print(f"  {prefix:30s}: {count:5d} {bar}")

    print("=" * 60)
    return total, len(corrupted), len(too_small)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_CHECK_COUNT)
    args = parser.parse_args()
    total, corrupted, too_small = verify_dataset(args.root, args.sample_count)
    
    if total < 10000:
        print(f"\n提示：当前数据量 {total} 张，建议自监督预训练至少 2 万张以上以获得更好效果。")
    elif total >= 20000:
        print(f"\n✓ 数据量 {total} 张，充足，可以进行高质量预训练。")
