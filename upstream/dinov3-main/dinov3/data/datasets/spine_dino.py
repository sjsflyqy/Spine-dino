"""Spine radiograph dataset adapter for DINOv3 self-supervised training."""

import logging
import os
from enum import Enum
from typing import Callable, Optional

import numpy as np

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov3")


class SpineDINO(ExtendedVisionDataset):
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
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self._split = split
        self._extra_root = extra
        self._image_files: Optional[np.ndarray] = None
        self._class_ids: Optional[np.ndarray] = None
        logger.info("SpineDINO dataset: split=%s, root=%s, extra=%s", split.value, root, extra)

    def _load_array(self, stem: str) -> np.ndarray:
        path = os.path.join(self._extra_root, f"{stem}-{self._split.value.upper()}.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing SpineDINO metadata: {path}")
        return np.load(path, allow_pickle=False, mmap_mode="r")

    def _get_image_files(self) -> np.ndarray:
        if self._image_files is None:
            self._image_files = self._load_array("image_files")
            logger.info("Loaded %d SpineDINO image paths", len(self._image_files))
        return self._image_files

    def _get_class_ids(self) -> np.ndarray:
        if self._class_ids is None:
            self._class_ids = self._load_array("class-ids")
        return self._class_ids

    def get_image_data(self, index: int) -> bytes:
        relpath = str(self._get_image_files()[index])
        full_path = os.path.join(self.root, self._split.value, relpath)
        with open(full_path, "rb") as handle:
            return handle.read()

    def get_target(self, index: int) -> int:
        return int(self._get_class_ids()[index])

    def get_targets(self) -> np.ndarray:
        return self._get_class_ids()

    def __len__(self) -> int:
        return len(self._get_image_files())
