# Copyright (c) SpineDINO Project
#
# Licensed under the Apache License, Version 2.0

import logging
import os
from enum import Enum
from typing import Callable, Optional

import numpy as np

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")


class SpineDINO(ExtendedVisionDataset):
    """
    脊柱X光图像自监督训练数据集，适配 DINOv2 训练流程。

    使用 generate_metadata.py 生成的元数据文件。

    期望的目录结构：
        root/
        ├── train/
        │   └── spine/
        │       ├── BUU2000_AP_00001.jpg
        │       └── ...
        └── extra/              ← 传入 extra 参数指向此目录
            ├── image_files-TRAIN.npy   相对路径列表（相对于 train/）
            ├── class-ids-TRAIN.npy     每张图片的类别索引（全 0）
            └── class-names-TRAIN.npy   类别名称（["spine"]）

    dataset_path 示例（用于训练配置）：
        SpineDINO:split=TRAIN:root=F:/AAAzpj/spine_dino_dataset:extra=F:/AAAzpj/spine_dino_dataset/extra
    """

    class Split(Enum):
        TRAIN = "train"
        VAL = "val"

    def __init__(
        self,
        *,
        split: "SpineDINO.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self._extra_root = extra
        self._image_files: Optional[np.ndarray] = None
        self._class_ids: Optional[np.ndarray] = None

        logger.info(f"SpineDINO dataset: split={split.value}, root={root}, extra={extra}")

    @property
    def split(self) -> "SpineDINO.Split":
        return self._split

    # ── 元数据加载 ───────────────────────────────────────────────────────────

    def _load_image_files(self) -> np.ndarray:
        if self._image_files is None:
            path = os.path.join(
                self._extra_root,
                f"image_files-{self._split.value.upper()}.npy",
            )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"找不到元数据文件: {path}\n"
                    "请先运行 generate_metadata.py 生成数据集元数据。"
                )
            self._image_files = np.load(path, allow_pickle=False)
            logger.info(f"SpineDINO: 已加载 {len(self._image_files)} 条图片路径")
        return self._image_files

    def _load_class_ids(self) -> np.ndarray:
        if self._class_ids is None:
            path = os.path.join(
                self._extra_root,
                f"class-ids-{self._split.value.upper()}.npy",
            )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"找不到元数据文件: {path}\n"
                    "请先运行 generate_metadata.py 生成数据集元数据。"
                )
            self._class_ids = np.load(path, allow_pickle=False)
        return self._class_ids

    # ── ExtendedVisionDataset 接口实现 ───────────────────────────────────────

    def get_image_data(self, index: int) -> bytes:
        image_files = self._load_image_files()
        # relpath 形如 "spine/BUU2000_AP_00001.jpg"，相对于 train/
        relpath = str(image_files[index])
        full_path = os.path.join(self.root, "train", relpath)
        with open(full_path, mode="rb") as f:
            return f.read()

    def get_target(self, index: int) -> int:
        class_ids = self._load_class_ids()
        return int(class_ids[index])

    def get_targets(self) -> np.ndarray:
        return self._load_class_ids()

    def __len__(self) -> int:
        return len(self._load_image_files())
