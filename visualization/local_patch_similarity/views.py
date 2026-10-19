"""Fixed multi-crop inputs and source-coordinate query selection."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch

from methods.geotopo_dino.geometry.transforms import transform_points


@contextmanager
def seeded_cpu_random(seed):
    state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        random.setstate(state)


@dataclass
class Views:
    source: Image.Image
    anchor: torch.Tensor
    locals: torch.Tensor
    anchor_geometry: dict
    local_geometry: list[dict]
    patch_size: int = 14

    @property
    def grid(self):
        return self.locals.shape[-1] // self.patch_size

    @property
    def transforms(self):
        return torch.stack([item["transform"] for item in self.local_geometry])

    @property
    def source_centres(self):
        centres = patch_centres(self.grid, self.patch_size)
        return torch.stack([transform_points(torch.linalg.inv(t), centres) for t in self.transforms])


def patch_centres(grid, patch_size):
    axis = (torch.arange(grid, dtype=torch.float32) + 0.5) * patch_size
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x.flatten(), y.flatten()), dim=-1)


def make_views(source, *, seed=42, mode="clean", anchor_size=518, local_size=196,
               local_scale=(0.2, 0.5), crop_ratio=(0.75, 4 / 3), flip_probability=0.5):
    upstream = str(Path(__file__).resolve().parents[2] / "upstream/dinov2-main")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from methods.geotopo_dino.data.augmentations import DataAugmentationGeoTopoDINO

    if mode not in {"clean", "train"}:
        raise ValueError("mode must be clean or train")
    augmentation = DataAugmentationGeoTopoDINO(
        (0.5, 1.0), local_scale, 8, global_crops_size=anchor_size,
        local_crops_size=local_size, patch_size=14, crop_ratio=crop_ratio,
        horizontal_flip_probability=flip_probability if mode == "train" else 0.0,
    )
    # Sample geometry once, independently of photometric RNG draws. Thus clean
    # and train modes with the same seed have exactly the same crop boxes.
    anchor_photometric = augmentation.anchor_photometric
    augmentation.anchor_photometric = torch.nn.Identity()
    augmentation.random_global_photometric = torch.nn.Identity()
    photometric = augmentation.local_photometric if mode == "train" else torch.nn.Identity()
    augmentation.local_photometric = torch.nn.Identity()
    with seeded_cpu_random(seed):
        output = augmentation(source)
    if mode == "train":
        # Replay the fixed geometry with the training photometric transforms;
        # tensors are normalized only after photometric PIL augmentation.
        from torchvision.transforms import functional as TVF
        from torchvision.transforms import InterpolationMode
        with seeded_cpu_random(seed + 1_000_003):
            augmentation.anchor_photometric = anchor_photometric
            augmentation._sample_flip = lambda: output["anchor_geometry"]["flipped"]
            anchor, anchor_geometry = augmentation._full_fov_anchor(source)
            locals_ = []
            for geometry in output["local_geometry"]:
                left, top, right, bottom = map(int, geometry["box_xyxy"].tolist())
                view = TVF.resized_crop(source, top, left, bottom - top, right - left,
                                        [local_size, local_size], InterpolationMode.BICUBIC, antialias=True)
                if geometry["flipped"]:
                    view = TVF.hflip(view)
                locals_.append(augmentation._normalize_pil(photometric(view)))
        output["global_crops"][0] = anchor
        output["anchor_geometry"] = anchor_geometry
        output["local_crops"] = locals_
    return Views(source, output["global_crops"][0], torch.stack(output["local_crops"]),
                 output["anchor_geometry"], output["local_geometry"])


def display_tensor(tensor):
    mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    return ((tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def select_queries(views, queries=None, source_queries=None):
    selected = []
    for local, row, col in queries or []:
        if not (0 <= local < 8 and 0 <= row < views.grid and 0 <= col < views.grid):
            raise ValueError(f"Query {(local, row, col)} outside [8, {views.grid}, {views.grid}]")
        selected.append({"local": local, "row": row, "column": col, "selection": "local_patch"})
    for xy in source_queries or []:
        point = torch.tensor(xy, dtype=torch.float32)
        if not torch.isfinite(point).all() or not (0 <= point[0] < views.source.width and 0 <= point[1] < views.source.height):
            raise ValueError(f"Source query {xy} outside image {views.source.size}")
        for local, transform in enumerate(views.transforms):
            mapped = transform_points(transform, point)
            if ((mapped >= 0) & (mapped < views.locals.shape[-1])).all():
                col, row = (mapped / views.patch_size).floor().long().tolist()
                selected.append({"local": local, "row": row, "column": col,
                                 "selection": "source_xy", "requested_source_xy": list(xy)})
                break
        else:
            raise ValueError(f"Source query {xy} is not covered by any local; choose another seed or --query")
    if not selected:
        selected = [{"local": 0, "row": views.grid // 2, "column": views.grid // 2,
                     "selection": "default_centre_not_anatomical"}]
    for query in selected:
        query["index"] = query["row"] * views.grid + query["column"]
        query["source_xy"] = views.source_centres[query["local"], query["index"]].tolist()
    return selected
