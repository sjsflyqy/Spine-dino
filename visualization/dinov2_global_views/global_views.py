"""Create the two official DINOv2 global views and displayable copies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageOps
import torch

from visualization.dino_spine_maps.preprocessing import (
    GeometryTransform,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PreparedImage,
)


GLOBAL_CROP_SIZE = 518
LOCAL_CROP_SIZE = 196


@dataclass(frozen=True)
class DinoV2GlobalViews:
    source_image: Image.Image
    views: tuple[PreparedImage, PreparedImage]
    seed: int
    global_crop_scale: tuple[float, float]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _to_display_image(normalized: torch.Tensor) -> Image.Image:
    pixels = normalized.detach().cpu().float().permute(1, 2, 0).numpy()
    pixels = pixels * IMAGENET_STD + IMAGENET_MEAN
    pixels = np.clip(np.rint(pixels * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _identity_prepared_image(normalized: torch.Tensor, patch_size: int) -> PreparedImage:
    display = _to_display_image(normalized)
    width, height = display.size
    if width % patch_size or height % patch_size:
        raise ValueError(
            f"Global crop {width}x{height} is not divisible by patch size {patch_size}"
        )
    grid_width = width // patch_size
    grid_height = height // patch_size
    geometry = GeometryTransform(
        original_width=width,
        original_height=height,
        resized_width=width,
        resized_height=height,
        model_width=width,
        model_height=height,
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        scale=1.0,
        patch_size=patch_size,
        grid_width=grid_width,
        grid_height=grid_height,
    )
    return PreparedImage(
        original=display,
        model_image=display.copy(),
        tensor=normalized.detach().cpu().contiguous(),
        valid_patch_mask=np.ones((grid_height, grid_width), dtype=np.bool_),
        geometry=geometry,
    )


def create_dinov2_global_views(
    image_path: str | Path,
    *,
    repo_root: Path,
    patch_size: int = 14,
    seed: int = 0,
    global_crop_scale: tuple[float, float] = (0.5, 1.0),
) -> DinoV2GlobalViews:
    """Apply the upstream DINOv2 augmentation and return its two global crops."""

    if patch_size != 14:
        raise ValueError(
            "The native DINOv2 518x518 global view requires the ViT-B/14 architecture"
        )
    scale_min, scale_max = global_crop_scale
    if not 0 < scale_min <= scale_max <= 1:
        raise ValueError("global_crop_scale must satisfy 0 < min <= max <= 1")

    path = Path(image_path).expanduser().resolve()
    try:
        with Image.open(path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not read raster image: {path}") from exc

    source_root = repo_root / "upstream" / "dinov2-main"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from dinov2.data.augmentations import DataAugmentationDINO

    _seed_everything(seed)
    augmentation = DataAugmentationDINO(
        global_crops_scale=global_crop_scale,
        local_crops_scale=(0.2, 0.5),
        local_crops_number=0,
        global_crops_size=GLOBAL_CROP_SIZE,
        local_crops_size=LOCAL_CROP_SIZE,
    )
    output = augmentation(source)
    crops = output["global_crops_teacher"]
    if len(crops) != 2:
        raise RuntimeError(f"Expected two DINOv2 teacher global crops, got {len(crops)}")
    views = tuple(_identity_prepared_image(crop, patch_size) for crop in crops)
    return DinoV2GlobalViews(
        source_image=source,
        views=(views[0], views[1]),
        seed=seed,
        global_crop_scale=global_crop_scale,
    )

