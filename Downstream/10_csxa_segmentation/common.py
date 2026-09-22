"""Paths, configuration and reproducibility helpers for Task 10."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[1]
LEVELS = ("C3", "C4", "C5", "C6", "C7")


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_config(path):
    config = yaml.safe_load(resolve(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name="auto"):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def read_splits(directory=None):
    directory = Path(directory or TASK / "splits")
    result = {}
    seen = set()
    for split in ("train", "val", "test"):
        ids = (directory / f"{split}.txt").read_text().splitlines()
        if len(ids) != 200 or len(set(ids)) != 200 or seen.intersection(ids):
            raise ValueError(f"Expected 200 distinct, disjoint IDs: {split}")
        seen.update(ids)
        result[split] = ids
    return result


def parameter_counts(model):
    return {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "backbone_trainable": sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)}


def trainable_state(model):
    # Frozen backbone weights are specified by path + hash, not duplicated.
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {n: v.detach().cpu() for n, v in model.state_dict().items()
            if not n.startswith("backbone.") or n in names}


def restore_trainable(model, state):
    expected = set(trainable_state(model))
    if set(state) != expected:
        raise ValueError(f"Checkpoint keys differ: missing={expected-set(state)}, extra={set(state)-expected}")
    model.load_state_dict(state, strict=False)


def weight_fingerprint(path):
    path = resolve(path)
    if path.is_file():
        return {"path": str(path), "sha256": sha256(path)}
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.iterdir() if p.suffix in {".safetensors", ".bin", ".json", ".pth"})
    if not files:
        raise ValueError(f"No local model files: {path}")
    return {"path": str(path), "files": {p.name: sha256(p) for p in files}}
