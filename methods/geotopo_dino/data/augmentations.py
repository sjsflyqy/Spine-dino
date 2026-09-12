"""DINOv2 multi-crop augmentation with explicit crop geometry."""

from __future__ import annotations

import logging
import random
from typing import Sequence

import torch
import torch.nn.functional as torch_F
from torchvision import transforms
from torchvision.transforms import functional as TVF

from dinov2.data.transforms import GaussianBlur, make_normalize_transform

from ..geometry.transforms import (
    crop_resize_transform,
    horizontal_flip_transform,
    letterbox_transform,
)


logger = logging.getLogger("dinov2")


class DataAugmentationGeoTopoDINO:
    """Create one full-FOV anchor, one random global and K local views.

    The anchor replaces one of DINOv2's two random global crops.  Teacher and
    student receive the same photometrically augmented view, as in the upstream
    implementation; masking is applied only inside the student backbone.
    """

    def __init__(
        self,
        global_crops_scale: Sequence[float],
        local_crops_scale: Sequence[float],
        local_crops_number: int,
        *,
        global_crops_size: int = 518,
        local_crops_size: int = 196,
        patch_size: int = 14,
        crop_ratio: Sequence[float] = (3.0 / 4.0, 4.0 / 3.0),
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        if global_crops_size % patch_size or local_crops_size % patch_size:
            raise ValueError("global and local sizes must be divisible by patch_size")
        self.global_crops_scale = tuple(global_crops_scale)
        self.local_crops_scale = tuple(local_crops_scale)
        self.local_crops_number = int(local_crops_number)
        self.global_crops_size = int(global_crops_size)
        self.local_crops_size = int(local_crops_size)
        self.patch_size = int(patch_size)
        self.crop_ratio = tuple(crop_ratio)
        self.horizontal_flip_probability = float(horizontal_flip_probability)

        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        self.anchor_photometric = transforms.Compose([color_jittering, GaussianBlur(p=1.0)])
        self.random_global_photometric = transforms.Compose(
            [color_jittering, GaussianBlur(p=0.1), transforms.RandomSolarize(threshold=128, p=0.2)]
        )
        self.local_photometric = transforms.Compose([color_jittering, GaussianBlur(p=0.5)])
        self.to_tensor = transforms.ToTensor()
        self.normalize = make_normalize_transform()

        logger.info(
            "GeoTopo views: anchor=%d, random_global=%d, locals=%dx%d, patch=%d",
            global_crops_size,
            global_crops_size,
            local_crops_number,
            local_crops_size,
            patch_size,
        )

    def _sample_flip(self) -> bool:
        return random.random() < self.horizontal_flip_probability

    def _normalize_pil(self, image) -> torch.Tensor:
        return self.normalize(self.to_tensor(image))

    def _full_fov_anchor(self, image):
        source_width, source_height = image.size
        scale = min(
            self.global_crops_size / float(source_width),
            self.global_crops_size / float(source_height),
        )
        resized_width = max(1, min(self.global_crops_size, round(source_width * scale)))
        resized_height = max(1, min(self.global_crops_size, round(source_height * scale)))
        resized = TVF.resize(
            image,
            [resized_height, resized_width],
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        resized = self.anchor_photometric(resized)
        tensor = self._normalize_pil(resized)

        remaining_width = self.global_crops_size - resized_width
        remaining_height = self.global_crops_size - resized_height
        pad_left = remaining_width // 2
        pad_right = remaining_width - pad_left
        pad_top = remaining_height // 2
        pad_bottom = remaining_height - pad_top
        tensor = torch_F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

        pixel_valid = torch.ones((1, resized_height, resized_width), dtype=torch.float32)
        pixel_valid = torch_F.pad(
            pixel_valid,
            (pad_left, pad_right, pad_top, pad_bottom),
            value=0.0,
        )
        transform = letterbox_transform(
            source_size=(source_height, source_width),
            resized_size=(resized_height, resized_width),
            padding_left=pad_left,
            padding_top=pad_top,
        )

        flipped = self._sample_flip()
        if flipped:
            tensor = TVF.hflip(tensor)
            pixel_valid = TVF.hflip(pixel_valid)
            transform = horizontal_flip_transform(self.global_crops_size) @ transform

        coverage = torch_F.avg_pool2d(
            pixel_valid.unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).squeeze(0).squeeze(0)
        token_valid = coverage >= 1.0 - 1e-6
        if not token_valid.any():
            token_valid = coverage > 0.5

        return tensor, {
            "transform": transform,
            "valid_mask": token_valid,
            "source_size": torch.tensor([source_height, source_width], dtype=torch.long),
            "resized_size": torch.tensor([resized_height, resized_width], dtype=torch.long),
            "padding_ltrb": torch.tensor([pad_left, pad_top, pad_right, pad_bottom], dtype=torch.long),
            "flipped": flipped,
        }

    def _random_resized_view(self, image, *, size: int, scale, photometric):
        source_width, source_height = image.size
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            image,
            scale=scale,
            ratio=self.crop_ratio,
        )
        view = TVF.resized_crop(
            image,
            top,
            left,
            height,
            width,
            [size, size],
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        transform = crop_resize_transform(
            top=top,
            left=left,
            height=height,
            width=width,
            output_size=(size, size),
        )
        flipped = self._sample_flip()
        if flipped:
            view = TVF.hflip(view)
            transform = horizontal_flip_transform(size) @ transform
        view = self._normalize_pil(photometric(view))
        return view, {
            "transform": transform,
            "box_xyxy": torch.tensor(
                [left, top, left + width, top + height], dtype=torch.float32
            ),
            "source_size": torch.tensor([source_height, source_width], dtype=torch.long),
            "area_fraction": float(height * width) / float(source_height * source_width),
            "aspect_ratio": float(width) / float(height),
            "flipped": flipped,
        }

    def __call__(self, image):
        anchor, anchor_meta = self._full_fov_anchor(image)
        random_global, random_global_meta = self._random_resized_view(
            image,
            size=self.global_crops_size,
            scale=self.global_crops_scale,
            photometric=self.random_global_photometric,
        )
        local_pairs = [
            self._random_resized_view(
                image,
                size=self.local_crops_size,
                scale=self.local_crops_scale,
                photometric=self.local_photometric,
            )
            for _ in range(self.local_crops_number)
        ]
        local_crops = [pair[0] for pair in local_pairs]
        local_geometry = [pair[1] for pair in local_pairs]
        return {
            "global_crops": [anchor, random_global],
            "global_crops_teacher": [anchor, random_global],
            "local_crops": local_crops,
            "anchor_geometry": anchor_meta,
            "random_global_geometry": random_global_meta,
            "local_geometry": local_geometry,
            "offsets": (),
        }
