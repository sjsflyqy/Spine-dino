"""Shared physical views and padding-aware query geometry for heterogeneous ViTs."""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from backbone.preprocessing import IMAGENET_MEAN, IMAGENET_STD
from methods.geotopo_dino.geometry.transforms import (
    crop_resize_transform, horizontal_flip_transform, letterbox_transform, transform_points,
)


@dataclass
class View:
    image: Image.Image
    tensor: torch.Tensor
    pixel_valid: torch.Tensor
    transform: torch.Tensor
    metadata: dict


def read_source(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def make_view(source, box, size, flipped=False):
    left, top, right, bottom = map(int, box)
    width, height = right - left, bottom - top
    if not (0 <= left < right <= source.width and 0 <= top < bottom <= source.height):
        raise ValueError("Crop is outside the source image")
    scale = min(size / width, size / height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    pixels = np.asarray(source.crop(box).resize(resized, Image.Resampling.BICUBIC)).copy()
    content = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    pad_left, pad_top = (size - resized[0]) // 2, (size - resized[1]) // 2
    padding = (pad_left, size - resized[0] - pad_left, pad_top, size - resized[1] - pad_top)
    tensor = F.pad((content - mean) / std, padding, value=0)
    valid = F.pad(torch.ones(1, resized[1], resized[0]), padding, value=0)
    transform = letterbox_transform(source_size=(height, width), resized_size=(resized[1], resized[0]),
                                    padding_left=pad_left, padding_top=pad_top)
    transform = transform @ crop_resize_transform(top=top, left=left, height=height, width=width,
                                                output_size=(height, width))
    if flipped:
        tensor, valid = tensor.flip(-1), valid.flip(-1)
        transform = horizontal_flip_transform(size) @ transform
    display = ((tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return View(Image.fromarray(display), tensor, valid[0].bool(), transform,
                {"box_xyxy": list(map(int, box)), "flipped": bool(flipped), "size": size,
                 "resized_wh": list(resized), "transform_source_to_view": transform.tolist()})


def box_iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return overlap / max(1, area_a + area_b - overlap)


def sample_pair(source, *, seed, size, min_iou=0.35, max_iou=0.78, flip_probability=0.0):
    rng = np.random.default_rng(seed)
    for _ in range(256):
        boxes = []
        for _ in range(2):
            width = max(2, round(source.width * rng.uniform(0.65, 1.0)))
            height = max(2, round(source.height * rng.uniform(0.55, 0.88)))
            left = int(rng.integers(0, source.width - width + 1))
            top = int(rng.integers(0, source.height - height + 1))
            boxes.append((left, top, left + width, top + height))
        centres = [np.asarray(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)) for b in boxes]
        displacement = np.linalg.norm(centres[0] - centres[1]) / math.hypot(*source.size)
        iou = box_iou(*boxes)
        if min_iou <= iou <= max_iou and displacement >= 0.08:
            flips = rng.random(2) < flip_probability
            return tuple(make_view(source, box, size, bool(flip)) for box, flip in zip(boxes, flips))
    raise ValueError("Could not sample distinct overlapping crops with the requested constraints")


def grid_centres(size, patch_size):
    axis = (torch.arange(size // patch_size, dtype=torch.float32) + 0.5) * patch_size
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x.flatten(), y.flatten()), dim=-1)


def valid_tokens(view, patch_size, border=1):
    coverage = F.avg_pool2d(view.pixel_valid[None, None].float(), patch_size, patch_size)[0, 0]
    valid = coverage >= 1 - 1e-6
    if border:
        # Padding with invalid values also erodes the outer crop boundary.
        invalid = F.pad((~valid).float()[None, None], (border,) * 4, value=1)
        valid = F.max_pool2d(invalid, 2 * border + 1, 1)[0, 0] == 0
    return valid


def sample_tokens(tokens, xy, size):
    """Bilinearly sample raw contextual tokens at continuous view-pixel positions."""
    side = math.isqrt(tokens.shape[0])
    if side * side != tokens.shape[0]:
        raise ValueError("Expected a square patch grid")
    grid = (xy.to(tokens.device).float() * (2 / size) - 1).reshape(1, 1, -1, 2)
    values = F.grid_sample(tokens.T.reshape(1, -1, side, side).float(), grid,
                           mode="bilinear", align_corners=False, padding_mode="zeros")
    return values[0, :, 0].T


def mask_at(mask, xy, size):
    grid = (xy.float() * (2 / size) - 1).reshape(1, 1, -1, 2)
    values = F.grid_sample(mask[None, None].float(), grid, align_corners=False)[0, 0, 0]
    return values >= 1 - 1e-5


def common_queries(a, b, patch_sizes, border=1):
    size, reference = a.image.width, patch_sizes[0]
    xy_a = grid_centres(size, reference)
    source_xy = transform_points(torch.linalg.inv(a.transform), xy_a)
    gt_b = transform_points(b.transform, source_xy)
    valid = torch.ones(len(xy_a), dtype=torch.bool)
    for patch in set(patch_sizes):
        valid &= mask_at(valid_tokens(a, patch, border), xy_a, size)
        valid &= mask_at(valid_tokens(b, patch, border), gt_b, size)
    # Entire reference-query footprint must be present in B, not only its centre.
    half = reference / 2
    offsets = torch.tensor([[-half, -half], [half, -half], [half, half], [-half, half]])
    corners = transform_points(b.transform @ torch.linalg.inv(a.transform), xy_a[:, None] + offsets)
    for corner in range(4):
        for patch in set(patch_sizes):
            valid &= mask_at(valid_tokens(b, patch, border), corners[:, corner], size)
    return {"indices": torch.arange(len(xy_a))[valid], "xy_a": xy_a[valid],
            "source_xy": source_xy[valid], "gt_b": gt_b[valid]}


def image_quality(image):
    small = image.copy()
    small.thumbnail((768, 768))
    gray = np.asarray(small.convert("L"), dtype=np.float32) / 255
    gy, gx = np.gradient(gray)
    return {"std": float(gray.std()), "gradient": float(np.hypot(gx, gy).mean()),
            "score": float(np.clip(gray.std() / 0.20, 0, 1))}


def query_texture(view, xy, patch_size):
    gray = np.asarray(view.image.convert("L"), dtype=np.float32) / 255
    gy, gx = np.gradient(gray)
    gradient = np.hypot(gx, gy)
    means, gradients, spreads = [], [], []
    for x, y in xy.tolist():
        left, top = int(x - patch_size / 2), int(y - patch_size / 2)
        region = gray[top:top + patch_size, left:left + patch_size]
        means.append(float(region.mean()))
        spreads.append(float(region.std()))
        gradients.append(float(gradient[top:top + patch_size, left:left + patch_size].mean()))
    return np.asarray(means), np.asarray(gradients), np.asarray(spreads)
