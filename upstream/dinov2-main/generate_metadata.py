"""
Spine-DINO 元数据生成脚本
为 DINOv2 的 ImageNet 数据加载器生成所需的 .npy 元数据文件

在运行 prepare_spine_dataset.py 之后执行此脚本。

生成:
  F:\AAAzpj\spine_dino_dataset\extra\
  ├── entries-TRAIN.npy     图片索引表
  ├── class-ids-TRAIN.npy   类别ID（全0，只有一个伪类别"spine"）
  └── class-names-TRAIN.npy 类别名（["spine"]）
"""

import os
import numpy as np
from pathlib import Path


DATASET_ROOT = r"F:\AAAzpj\spine_dino_dataset"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def generate_metadata(dataset_root):
    root = Path(dataset_root)
    train_dir = root / "train"
    extra_dir = root / "extra"
    extra_dir.mkdir(exist_ok=True)

    # 收集所有图片
    all_images = []
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])

    class_names = [d.name for d in class_dirs]
    class_id_map = {name: i for i, name in enumerate(class_names)}

    print(f"发现 {len(class_names)} 个类别: {class_names}")

    for class_dir in class_dirs:
        class_id = class_id_map[class_dir.name]
        imgs = sorted([
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ])
        for img in imgs:
            all_images.append((img.relative_to(train_dir).as_posix(), class_id))

    print(f"总图片数: {len(all_images)} 张")

    # 生成 entries：每条记录为 (class_id, start_byte, size_byte, filename_hash, extra)
    # DINOv2 的 ImageNet 类期望的 entries 格式：
    # dtype = [("split_index", "<i8"), ("class_index", "<i8"), ("start_byte", "<i8"),
    #          ("stop_byte", "<i8"), ("flags", "u1"), ("extra", "<i8")]
    # 简化版：直接生成图片路径列表和类别ID数组即可供自定义数据集使用
    
    # 生成文件路径列表（相对于 train/）
    image_files = [img for img, _ in all_images]
    class_ids = np.array([cid for _, cid in all_images], dtype=np.int64)
    class_names_arr = np.array(class_names)

    # 保存
    np.save(extra_dir / "image_files-TRAIN.npy", np.array(image_files))
    np.save(extra_dir / "class-ids-TRAIN.npy", class_ids)
    np.save(extra_dir / "class-names-TRAIN.npy", class_names_arr)

    print(f"\n元数据已保存至: {extra_dir}")
    print(f"  image_files-TRAIN.npy : {len(image_files)} 条")
    print(f"  class-ids-TRAIN.npy   : shape {class_ids.shape}")
    print(f"  class-names-TRAIN.npy : {class_names_arr.tolist()}")

    # 同时生成一个简单的 txt 文件清单（方便调试）
    list_file = extra_dir / "train_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for img_path, cid in all_images:
            f.write(f"{img_path}\t{cid}\n")
    print(f"  train_list.txt        : {len(all_images)} 行")

    return len(all_images)


if __name__ == "__main__":
    print("Spine-DINO 元数据生成")
    print("=" * 50)
    total = generate_metadata(DATASET_ROOT)
    print(f"\n完成！共 {total} 张图片")
    print("\n下一步: 运行 verify_dataset.py 验证数据集完整性")
