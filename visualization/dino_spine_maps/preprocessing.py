"""Aspect-preserving image preprocessing with an invertible geometry record."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
IMAGENET_MEAN_RGB = tuple(int(round(value * 255)) for value in IMAGENET_MEAN)


@dataclass(frozen=True)
class GeometryTransform:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    model_width: int
    model_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    scale: float
    patch_size: int
    grid_width: int
    grid_height: int

    def as_dict(self) -> dict:
        return asdict(self)

    def grid_to_original(self, grid: np.ndarray, *, nearest: bool = False) -> np.ndarray:
        """Upsample a patch grid, remove padding, and restore original dimensions."""

        array = np.asarray(grid, dtype=np.float32)
        if array.shape != (self.grid_height, self.grid_width):
            raise ValueError(
                f"Expected grid {(self.grid_height, self.grid_width)}, got {array.shape}"
            )
        interpolation = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
        model_map = Image.fromarray(array, mode="F").resize(
            (self.model_width, self.model_height), resample=interpolation
        )
        content = model_map.crop(
            (
                self.pad_left,
                self.pad_top,
                self.pad_left + self.resized_width,
                self.pad_top + self.resized_height,
            )
        )
        restored = content.resize(
            (self.original_width, self.original_height), resample=interpolation
        )
        return np.asarray(restored, dtype=np.float32)


@dataclass(frozen=True)
class PreparedImage:
    original: Image.Image
    model_image: Image.Image
    tensor: torch.Tensor
    valid_patch_mask: np.ndarray
    geometry: GeometryTransform


def _ceil_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def prepare_image(
    image_path: str | Path,
    *,
    patch_size: int,
    long_side: int = 896,
    min_valid_fraction: float = 0.5,
) -> PreparedImage:
    """Load a raster image, letterbox minimally, and normalize it for DINO."""

    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    if long_side <= 0:
        raise ValueError("long_side must be positive")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1]")

    try:
        with Image.open(path) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"Could not read {path} as a raster image. DICOM input is not enabled in this first version."
        ) from exc

    original_width, original_height = original.size
    scale = long_side / max(original_width, original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    resized = original.resize((resized_width, resized_height), Image.Resampling.BICUBIC)

    model_width = _ceil_multiple(resized_width, patch_size)
    model_height = _ceil_multiple(resized_height, patch_size)
    pad_left = (model_width - resized_width) // 2
    pad_top = (model_height - resized_height) // 2
    pad_right = model_width - resized_width - pad_left
    pad_bottom = model_height - resized_height - pad_top
    model_image = Image.new("RGB", (model_width, model_height), IMAGENET_MEAN_RGB)
    model_image.paste(resized, (pad_left, pad_top))

    valid_pixels = np.zeros((model_height, model_width), dtype=np.float32)
    valid_pixels[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = 1.0
    grid_height = model_height // patch_size
    grid_width = model_width // patch_size
    fractions = valid_pixels.reshape(
        grid_height, patch_size, grid_width, patch_size
    ).mean(axis=(1, 3))
    valid_patch_mask = fractions >= min_valid_fraction

    pixels = np.asarray(model_image, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
    geometry = GeometryTransform(
        original_width=original_width,
        original_height=original_height,
        resized_width=resized_width,
        resized_height=resized_height,
        model_width=model_width,
        model_height=model_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        scale=scale,
        patch_size=patch_size,
        grid_width=grid_width,
        grid_height=grid_height,
    )
    return PreparedImage(
        original=original,
        model_image=model_image,
        tensor=tensor,
        valid_patch_mask=valid_patch_mask,
        geometry=geometry,
    )

