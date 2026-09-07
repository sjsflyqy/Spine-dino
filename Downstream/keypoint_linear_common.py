"""Shared utilities for strict frozen-backbone landmark linear probes."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from backbone.preprocessing import IMAGENET_MEAN, IMAGENET_STD


Image.MAX_IMAGE_PIXELS = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_keypoint_transform(height: int, width: int, train: bool = False):
    operations = [
        transforms.Resize(
            (height, width),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    ]
    if train:
        # Photometric augmentation preserves landmark coordinates. Geometry
        # augmentation is deliberately excluded from the benchmark protocol.
        operations.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
    operations.extend(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def deterministic_split(
    identifiers: list[str], seed: int, test_fraction: float = 0.2,
    validation_fraction_of_remainder: float = 0.1,
) -> dict[str, list[str]]:
    identifiers = sorted(identifiers)
    random.Random(seed).shuffle(identifiers)
    test_count = max(1, round(len(identifiers) * test_fraction))
    test = identifiers[:test_count]
    remainder = identifiers[test_count:]
    validation_count = max(
        1, round(len(remainder) * validation_fraction_of_remainder)
    )
    validation = remainder[:validation_count]
    train = remainder[validation_count:]
    return {"train": train, "val": validation, "test": test}


def gaussian_heatmap_loss(
    logits: torch.Tensor, coordinates: torch.Tensor, sigma: float = 1.5
) -> torch.Tensor:
    """Cross entropy against normalized Gaussian targets on the token grid."""
    _, _, height, width = logits.shape
    dtype = logits.dtype
    y_grid = torch.arange(height, device=logits.device, dtype=dtype).view(1, 1, height, 1)
    x_grid = torch.arange(width, device=logits.device, dtype=dtype).view(1, 1, 1, width)
    target_x = coordinates[..., 0].to(dtype).unsqueeze(-1).unsqueeze(-1) * (width - 1)
    target_y = coordinates[..., 1].to(dtype).unsqueeze(-1).unsqueeze(-1) * (height - 1)
    squared_distance = (x_grid - target_x).square() + (y_grid - target_y).square()
    targets = torch.exp(-squared_distance / (2.0 * sigma * sigma))
    targets = targets / targets.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    log_probabilities = F.log_softmax(logits.flatten(2), dim=-1).view_as(logits)
    return -(targets * log_probabilities).sum(dim=(-2, -1)).mean()


def soft_argmax_2d(logits: torch.Tensor) -> torch.Tensor:
    """Decode heatmap logits to normalized (x, y) coordinates."""
    batch, landmarks, height, width = logits.shape
    probabilities = F.softmax(logits.flatten(2).float(), dim=-1).view(
        batch, landmarks, height, width
    )
    x_axis = torch.linspace(0.0, 1.0, width, device=logits.device)
    y_axis = torch.linspace(0.0, 1.0, height, device=logits.device)
    predicted_x = (probabilities.sum(dim=2) * x_axis).sum(dim=-1)
    predicted_y = (probabilities.sum(dim=3) * y_axis).sum(dim=-1)
    return torch.stack((predicted_x, predicted_y), dim=-1)


def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def trainable_parameter_report(model: nn.Module) -> dict[str, int]:
    backbone = sum(
        parameter.numel()
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    )
    total = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "backbone_trainable": backbone,
        "head_trainable": total - backbone,
        "total_trainable": total,
        # Kept for compatibility with existing logs.
        "probe_trainable": total,
    }


def is_adapter_finetune(model: nn.Module) -> bool:
    return bool(getattr(model.backbone, "adapter_enabled", False))


def build_downstream_optimizer(model: nn.Module, training: dict):
    """Use a conservative LR for LoRA and retain the old LR for task heads."""
    adapter_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    if not adapter_parameters:
        optimizer = torch.optim.AdamW(
            head_parameters,
            lr=float(training["learning_rate"]),
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        return optimizer, None
    groups = []
    if adapter_parameters:
        groups.append({
            "params": adapter_parameters,
            "lr": float(training.get("adapter_learning_rate", 1e-4)),
        })
    if head_parameters:
        groups.append({
            "params": head_parameters,
            "lr": float(training.get("head_learning_rate", training["learning_rate"])),
        })
    optimizer = torch.optim.AdamW(
        groups, weight_decay=float(training.get("weight_decay", 0.01))
    )
    warmup_epochs = int(training.get("warmup_epochs", 0))
    total_epochs = int(training["epochs"])

    def lr_multiplier(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / warmup_epochs
        progress = (epoch_index - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    return optimizer, scheduler


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def load_trainable_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected trainable checkpoint keys: {unexpected}")


def checkpoint_filename(model: nn.Module) -> str:
    return "best_lora_finetune.pt" if is_adapter_finetune(model) else "best_linear_probe.pt"
