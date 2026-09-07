"""Image preprocessing shared by RAD-DINO-family comparisons."""

from __future__ import annotations

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int = 518, train: bool = False):
    operations = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
    ]
    if train:
        # Geometry-changing augmentations are deliberately excluded here.
        operations.append(
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        )
    operations.extend(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)
