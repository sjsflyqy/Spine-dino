"""Image preprocessing shared by RAD-DINO-family comparisons."""

from __future__ import annotations

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_normalization(spec=None):
    """Return ``(mean, std)`` for a model config.

    ``None`` keeps the historical ImageNet preprocessing.  An explicit null
    mean/std pair disables normalization, which is required by the DAX
    baseline's published preprocessing.
    """
    if spec is None:
        return IMAGENET_MEAN, IMAGENET_STD
    if not isinstance(spec, dict):
        raise ValueError("model.normalization must be a mapping with mean and std")
    mean, std = spec.get("mean"), spec.get("std")
    if mean is None and std is None:
        return None, None
    if not isinstance(mean, (list, tuple)) or not isinstance(std, (list, tuple)):
        raise ValueError("model.normalization requires both mean and std, or both null")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("model.normalization mean/std must contain three channels")
    return tuple(float(value) for value in mean), tuple(float(value) for value in std)


def build_transform(image_size: int = 518, train: bool = False, normalization=None):
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
    operations.extend([transforms.Lambda(lambda image: image.convert("RGB")), transforms.ToTensor()])
    mean, std = resolve_normalization(normalization)
    if mean is not None:
        operations.append(transforms.Normalize(mean, std))
    return transforms.Compose(operations)
