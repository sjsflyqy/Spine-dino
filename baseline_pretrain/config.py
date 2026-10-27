"""Local configuration validation; does not import or alter upstream configs."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path, overrides=()):
    config = yaml.safe_load(resolve(path).read_text())
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"Expected key=value override, got {override!r}")
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"Unknown configuration key: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f"Unknown configuration key: {key}")
        parsed = yaml.safe_load(value)
        # PyYAML treats CLI spellings such as 4e-4 as strings; retain the
        # declared numeric type without changing literal paths/string values.
        if isinstance(parsed, str) and type(node[parts[-1]]) is float:
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        node[parts[-1]] = parsed
    validate(config)
    return config


def validate(c):
    if c["method"] not in {"dax", "chexfound", "cxformer", "medical_mae", "ray_dino"}:
        raise ValueError("method must be dax, chexfound, cxformer, medical_mae or ray_dino")
    if not c.get("initialization"):
        raise ValueError("A public initialization checkpoint is required; random-init mode is not provided")
    t = c["training"]
    for key in ("batch_size_per_gpu", "accumulation_steps", "total_updates", "checkpoint_every", "keep_last", "log_every", "expected_world_size", "prefetch_factor"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("workers", "warmup_updates", "freeze_last_layer_updates"):
        if type(t[key]) is not int or t[key] < 0:
            raise ValueError(f"training.{key} must be a nonnegative integer")
    if t["warmup_updates"] >= t["total_updates"] or t["freeze_last_layer_updates"] >= t["total_updates"]:
        raise ValueError("warmup/freeze must be shorter than total_updates")
    if t["parallel"] not in {"ddp", "fsdp", "none"} or t["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Invalid parallel or precision setting")
    if t["fsdp_sharding"] not in {"FULL_SHARD", "SHARD_GRAD_OP"}:
        raise ValueError("Invalid FSDP sharding policy")
    if t["parallel"] == "fsdp" and c["method"] != "medical_mae" and t["fsdp_sharding"] != "FULL_SHARD":
        raise ValueError("EMA teacher FSDP requires FULL_SHARD so student/teacher shards match outside forward")
    for key in ("learning_rate", "min_learning_rate", "backbone_lr_multiplier", "patch_embed_lr_multiplier", "layerwise_decay", "weight_decay", "weight_decay_end", "clip_grad"):
        if not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative")
    if not 0 < t["layerwise_decay"] <= 1 or not t["learning_rate"] > 0 or not t["backbone_lr_multiplier"] > 0:
        raise ValueError("Invalid LR/decay multipliers")
    if t["min_learning_rate"] > t["learning_rate"]:
        raise ValueError("min_learning_rate exceeds learning_rate")
    m, crops = c["model"], c["crops"]
    if m["dim"] % m["heads"] or m["depth"] < 1 or not 0 <= m["drop_path"] < 1:
        raise ValueError("Invalid transformer architecture")
    for size in (m["image_size"], crops["global_size"], crops["local_size"]):
        if size < m["patch_size"] or size % m["patch_size"]:
            raise ValueError("Crop and model sizes must be positive multiples of patch_size")
    if crops["global_size"] != m["image_size"]:
        raise ValueError("Set model.image_size to the global crop size for strict initialization")
    if c["method"] == "medical_mae":
        if not 0 < c["objective"]["mask_ratio"] < 1:
            raise ValueError("MAE mask_ratio must lie in (0,1)")
    else:
        o = c["objective"]
        if not 0 < o["student_temperature"] or not 0 < o["teacher_temperature"] or not 0 < o["teacher_temperature_start"]:
            raise ValueError("Temperatures must be positive")
        if not 0 <= o["center_momentum"] < 1 or not 0 <= o["teacher_momentum"] <= o["teacher_momentum_end"] <= 1:
            raise ValueError("Invalid center/teacher momentum")
        if not 0 <= o["temperature_warmup_updates"] <= t["total_updates"]:
            raise ValueError("Temperature warmup exceeds budget")
        if crops["local_count"] < 0 or o["centering"] not in {"centering", "sinkhorn_knopp"}:
            raise ValueError("Invalid crop count or centering")
        if c["method"] in {"chexfound", "cxformer", "ray_dino"}:
            if not 0 < o["mask_sample_probability"] <= 1 or not 0 < o["mask_ratio_min"] <= o["mask_ratio_max"] < 1:
                raise ValueError("Invalid iBOT masking probabilities")
            if o["koleo_scope"] not in {"local", "global"}:
                raise ValueError("koleo_scope must be local or global")


def compatibility_config(c):
    c = copy.deepcopy(c)
    for key in ("workers", "prefetch_factor", "checkpoint_every", "keep_last", "log_every"):
        c["training"].pop(key, None)
    return c


def config_digest(c):
    return hashlib.sha256(json.dumps(compatibility_config(c), sort_keys=True).encode()).hexdigest()
