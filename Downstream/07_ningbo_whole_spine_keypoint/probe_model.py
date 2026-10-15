"""Optional task-07 heads with frozen backbones or LoRA fine-tuning."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from backbone import build_backbone
from keypoint_decoder import extract_intermediate_feature_maps
from keypoint_linear_common import load_trainable_state_dict
from model import PairedKeypointLinearProbe


PROBE_TYPES = (
    "legacy_linear", "multilayer_linear", "spatial_linear", "mlp_probe", "cnn_probe",
)


def adapter_spec(config: dict | None = None) -> dict:
    """Canonicalize the LoRA settings used by the shared backbone builder."""
    config = config or {}
    if not config.get("enabled", False):
        return {"enabled": False}
    return {
        "enabled": True,
        "type": str(config.get("type", "lora")).lower(),
        "rank": int(config.get("rank", 8)),
        "alpha": float(config.get("alpha", 16.0)),
        "dropout": float(config.get("dropout", 0.05)),
        "target_modules": sorted(set(config.get("target_modules", ["qkv", "query", "key", "value"]))),
    }


def normalize_probe_config(config: dict | None = None) -> dict:
    """Resolve defaults so checkpoints describe the exact readout architecture."""
    if config is not None and not isinstance(config, dict):
        raise ValueError("model.probe must be a mapping")
    config = deepcopy(config or {})
    unknown = set(config) - {"type", "feature_layers", "hidden_channels"}
    if unknown:
        raise ValueError(f"Unknown probe settings: {sorted(unknown)}")
    kind = config.get("type", "legacy_linear")
    if kind not in PROBE_TYPES:
        raise ValueError(f"Unknown probe type {kind!r}; choose from {PROBE_TYPES}")
    layers = config.get(
        "feature_layers", [2, 5, 8, 11] if kind == "multilayer_linear" else "last"
    )
    if layers != "last":
        if (
            not isinstance(layers, (list, tuple)) or not layers
            or any(type(index) is not int or index < 0 for index in layers)
            or list(layers) != sorted(set(layers))
        ):
            raise ValueError("probe.feature_layers must be 'last' or increasing unique block indices")
        layers = list(layers)
    if kind == "legacy_linear" and layers != "last":
        raise ValueError("legacy_linear must use feature_layers: last")
    if kind == "multilayer_linear" and (layers == "last" or len(layers) < 2):
        raise ValueError("multilayer_linear requires at least two frozen feature layers")
    spec = {"type": kind, "feature_layers": layers}
    if kind in {"mlp_probe", "cnn_probe"}:
        width = config.get("hidden_channels", 256)
        if type(width) is not int or width < 1:
            raise ValueError("probe.hidden_channels must be a positive integer")
        spec["hidden_channels"] = width
    elif "hidden_channels" in config:
        raise ValueError(f"hidden_channels does not apply to {kind}")
    return spec


def _make_head(channels: int, landmarks: int, spec: dict) -> nn.Module:
    kind = spec["type"]
    if kind == "multilayer_linear":
        return nn.Conv2d(channels, landmarks, 1)
    if kind == "spatial_linear":
        return nn.Conv2d(channels, landmarks, 3, padding=1)
    width = spec["hidden_channels"]
    if kind == "mlp_probe":
        return nn.Sequential(
            nn.Conv2d(channels, width, 1), nn.GELU(),
            nn.Conv2d(width, landmarks, 1),
        )
    groups = next(group for group in (32, 16, 8, 4, 2, 1) if width % group == 0)
    return nn.Sequential(
        nn.Conv2d(channels, width, 1), nn.GroupNorm(groups, width), nn.GELU(),
        nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(groups, width), nn.GELU(),
        nn.Conv2d(width, landmarks, 1),
    )


class PairedFrozenFeatureProbe(PairedKeypointLinearProbe):
    """Read one or several feature maps, optionally training LoRA adapters."""

    def __init__(self, backbone_name: str, weights: str | None, spec: dict,
                 adapter: dict | None = None):
        nn.Module.__init__(self)
        self.probe_spec = deepcopy(spec)
        self.backbone = build_backbone(backbone_name, weights, freeze=True, adapter=adapter)
        self.backbone.eval()
        layers = spec["feature_layers"]
        count = 1 if layers == "last" else len(layers)
        channels = 2 * self.backbone.embedding_dim * count
        self.ap_head = _make_head(channels, 68, spec)
        self.lat_head = _make_head(channels, 68, spec)

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        context = (
            torch.enable_grad()
            if self.training and self.backbone.adapter_enabled
            else torch.no_grad()
        )
        with context:
            return self._extract_features(images)

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        layers = self.probe_spec["feature_layers"]
        if layers == "last":
            return self.backbone(images).feature_map
        model = self.backbone.model
        depth = len(model.blocks) if hasattr(model, "blocks") else model.config.num_hidden_layers
        if layers[-1] >= depth:
            raise ValueError(f"Requested block {layers[-1]}, but backbone has {depth} blocks")
        maps = extract_intermediate_feature_maps(self.backbone, images, layers)
        if not hasattr(model, "get_intermediate_layers"):
            # Native DINO uses its frozen final LayerNorm for each selected block.
            # The shared decoder extractor returns raw HF hidden states; apply the
            # equivalent frozen norm here without changing that existing protocol.
            if not hasattr(model, "layernorm"):
                raise ValueError("HF multi-layer probes require the backbone's frozen layernorm")
            maps = [
                model.layernorm(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                for value in maps
            ]
        return torch.cat(maps, dim=1)

    def forward(self, ap_images: torch.Tensor, lat_images: torch.Tensor):
        if ap_images.shape != lat_images.shape:
            raise ValueError("Task 07 AP and LAT batches must have identical shapes")
        batch = ap_images.shape[0]
        features = self._features(torch.cat((ap_images, lat_images), dim=0))
        ap_features, lat_features = features[:batch], features[batch:]
        return {
            "ap": self.ap_head(torch.cat((ap_features, lat_features), dim=1)),
            "lat": self.lat_head(torch.cat((lat_features, ap_features), dim=1)),
        }


def build_probe(model_config: dict, weights: str | None = None) -> nn.Module:
    weights = weights if weights is not None else model_config.get("weights")
    spec = normalize_probe_config(model_config.get("probe"))
    adapter = model_config.get("adapter")
    if spec["type"] == "legacy_linear":
        # Use the original class itself, preserving initialization and state keys.
        model = PairedKeypointLinearProbe(model_config["backbone"], weights, adapter=adapter)
        model.probe_spec = spec
    else:
        model = PairedFrozenFeatureProbe(model_config["backbone"], weights, spec, adapter=adapter)
    model.adapter_spec = adapter_spec(adapter)
    return model


def restore_probe(model: nn.Module, checkpoint: dict) -> None:
    """Restore heads plus optional LoRA, rejecting incomplete trainable states."""
    saved_spec = normalize_probe_config(checkpoint.get("probe_spec"))
    if saved_spec != model.probe_spec:
        raise ValueError(f"Checkpoint probe mismatch: saved={saved_spec}, requested={model.probe_spec}")
    saved_adapter = checkpoint.get("adapter_spec")
    if saved_adapter is None:
        saved_adapter = checkpoint.get("config", {}).get("model", {}).get("adapter")
    if adapter_spec(saved_adapter) != model.adapter_spec:
        raise ValueError("Checkpoint adapter mismatch: enabled state or LoRA settings differ")
    if model.backbone.adapter_enabled:
        state = checkpoint.get("trainable_state")
        if not isinstance(state, dict):
            raise ValueError("LoRA checkpoint must contain trainable_state (adapters and heads)")
        expected = {name for name, value in model.named_parameters() if value.requires_grad}
        missing = sorted(expected - state.keys())
        unexpected = sorted(state.keys() - expected)
        if missing or unexpected:
            raise ValueError(f"Incomplete LoRA checkpoint: missing={missing}, unexpected={unexpected}")
        load_trainable_state_dict(model, state)
    else:
        model.load_probe_state_dict(checkpoint["probe"])
