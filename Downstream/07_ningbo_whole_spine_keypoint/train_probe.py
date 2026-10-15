"""Train optional task-07 heads with a frozen backbone or configured LoRA."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Downstream"))

from dataset import CORNER_ORDER, VERTEBRAE, PairedWholeSpineDataset, audit_pairs
from keypoint_linear_common import (
    build_downstream_optimizer,
    deterministic_split,
    gaussian_heatmap_loss,
    is_adapter_finetune,
    save_json,
    seed_everything,
    trainable_parameter_report,
    trainable_state_dict,
)
from probe_model import PROBE_TYPES, build_probe, normalize_probe_config, restore_probe
from train import evaluate, make_loader


def load_probe_config(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", type=Path, help="Override model.weights")
    parser.add_argument("--output-dir", type=Path, help="Use an empty experiment directory")
    parser.add_argument("--probe-type", choices=PROBE_TYPES, help="Select a head; reset feature layers to its default")
    parser.add_argument("--feature-layers", help="Override with 'last' or zero-based indices, e.g. 2,5,8,11")
    parser.add_argument("--hidden-channels", type=int, help="MLP/CNN hidden width (default: 256)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.expanduser().read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Config requires a model mapping")
    if not isinstance(config.get("training"), dict):
        raise ValueError("Config requires a training mapping")
    model_config = config["model"]
    previous_probe = model_config.get("probe")
    probe = deepcopy(previous_probe or {})
    if args.probe_type is not None:
        # Selecting another preset must not accidentally keep multi-layer inputs
        # or hidden-width settings from the previous kind of head.
        probe = {"type": args.probe_type}
    if args.feature_layers is not None:
        probe["feature_layers"] = (
            "last" if args.feature_layers == "last"
            else [int(index) for index in args.feature_layers.split(",")]
        )
    if args.hidden_channels is not None:
        probe["hidden_channels"] = args.hidden_channels
    model_config["probe"] = normalize_probe_config(probe)
    if args.weights is not None:
        model_config["weights"] = str(args.weights.expanduser())
    if not model_config.get("weights"):
        raise ValueError("model.weights is required")
    for key in ("epochs", "batch_size", "workers"):
        value = getattr(args, key)
        if value is not None:
            config["training"][key] = value
    config["device"] = args.device
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir.expanduser())
    elif previous_probe is None or args.probe_type is not None or not config.get("output_dir"):
        name = Path(config.get("output_dir", model_config["backbone"])).name
        experiment_root = (
            "task07_keypoint_lora_probe"
            if (model_config.get("adapter") or {}).get("enabled", False)
            else "task07_keypoint_probe"
        )
        config["output_dir"] = str(
            Path("outputs/downstream") / experiment_root / model_config["probe"]["type"] / name
        )
    return config


def main(config: dict | None = None):
    config = deepcopy(load_probe_config() if config is None else config)
    training = config["training"]
    model_config = config["model"]
    model_config["probe"] = normalize_probe_config(model_config.get("probe"))
    spec = model_config["probe"]
    epochs = int(training["epochs"])
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    if epochs < 1 or accumulation_steps < 1 or int(training["batch_size"]) < 1:
        raise ValueError("epochs, batch_size and gradient_accumulation_steps must be >= 1")
    output_dir = (REPO_ROOT / config["output_dir"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Experiment directory is not empty: {output_dir}; choose a new --output-dir")

    seed = int(training.get("seed", 42))
    seed_everything(seed)
    device_name = config.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    clean, excluded = audit_pairs(REPO_ROOT / "Downstream/data/task07")
    splits = deterministic_split(list(clean), seed)
    if any(not identifiers for identifiers in splits.values()):
        raise ValueError("Task 07 requires nonempty train, validation and test paired splits")
    height = int(model_config.get("image_height", 896))
    width = int(model_config.get("image_width", 448))
    datasets = {
        split: PairedWholeSpineDataset(
            REPO_ROOT / "Downstream/data/task07", identifiers, clean,
            height, width, split == "train",
        )
        for split, identifiers in splits.items()
    }
    loaders = {
        split: make_loader(dataset, training, split == "train")
        for split, dataset in datasets.items()
    }
    weights = Path(model_config["weights"]).expanduser()
    if not weights.is_absolute():
        weights = REPO_ROOT / weights
    model = build_probe(model_config, str(weights)).to(device)
    if height % model.backbone.patch_size or width % model.backbone.patch_size:
        raise ValueError("Input height and width must be divisible by the backbone patch size")
    report = trainable_parameter_report(model)
    adapter_enabled = is_adapter_finetune(model)
    unexpected_trainables = [
        name for name, value in model.backbone.named_parameters()
        if value.requires_grad and not any(part in {"lora_A", "lora_B"} for part in name.split("."))
    ]
    if report["head_trainable"] <= 0 or unexpected_trainables or (
        adapter_enabled and report["backbone_trainable"] <= 0
    ) or (not adapter_enabled and report["backbone_trainable"] != 0):
        raise RuntimeError(f"Invalid probe parameters: {report}, unexpected={unexpected_trainables}")
    linear = spec["type"] in {"legacy_linear", "multilayer_linear", "spatial_linear"}
    protocol = "task07_frozen_linear_probe" if linear else "task07_frozen_nonlinear_probe"
    if adapter_enabled:
        protocol = "task07_lora_finetune"
    metadata = {"protocol": protocol, "probe_spec": spec, "adapter_spec": model.adapter_spec}
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)
    save_json(output_dir / "split_and_audit.json", {
        "protocol": "paired DR identifier; 20% test, 10% of remainder validation",
        "seed": seed, "splits": splits, "excluded": excluded,
    })
    print(json.dumps({
        **metadata, "device": str(device), **report,
        "train_pairs": len(datasets["train"]), "val_pairs": len(datasets["val"]),
        "test_pairs": len(datasets["test"]), "excluded_pairs": len(excluded),
        "output_dir": str(output_dir),
        "adapter_modules": getattr(model.backbone, "adapter_modules", []),
    }))

    optimizer, scheduler = build_downstream_optimizer(model, training)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_name = "best_linear_probe.pt" if spec["type"] == "legacy_linear" else f"best_{spec['type']}.pt"
    if adapter_enabled:
        checkpoint_name = "best_lora_finetune.pt" if spec["type"] == "legacy_linear" else f"best_lora_{spec['type']}.pt"
    checkpoint_path = output_dir / checkpoint_name
    early = training.get("early_stopping", {})
    early_enabled = bool(early.get("enabled", False))
    patience = int(early.get("patience", 15))
    min_delta = float(early.get("min_delta", 0.0))
    sigma = float(training.get("heatmap_sigma", 1.5))
    max_grad_norm = float(training.get("max_grad_norm", 0.0))
    best = float("inf")
    epochs_without_improvement = 0
    stopped_epoch = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        sample_count = 0
        for batch_index, batch in enumerate(loaders["train"]):
            ap_images = batch["ap_image"].to(device, non_blocking=True)
            lat_images = batch["lat_image"].to(device, non_blocking=True)
            ap_target = batch["ap_points"].to(device, non_blocking=True)
            lat_target = batch["lat_points"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(ap_images, lat_images)
                loss = gaussian_heatmap_loss(output["ap"], ap_target, sigma)
                loss = loss + gaussian_heatmap_loss(output["lat"], lat_target, sigma)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}, batch {batch_index}")
            scaler.scale(loss / accumulation_steps).backward()
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loaders["train"]):
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * ap_images.shape[0]
            sample_count += ap_images.shape[0]

        metrics, _ = evaluate(model, loaders["val"], device)
        metrics.update(**metadata, split="val", epoch=epoch, train_loss=running_loss / max(sample_count, 1))
        print(json.dumps(metrics))
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        if metrics["mean_epsilon_dec"] < best - min_delta:
            best = metrics["mean_epsilon_dec"]
            epochs_without_improvement = 0
            payload = {
                "format_version": 2 if adapter_enabled else 1, **metadata, "config": config,
                "val_metrics": metrics, "vertebrae": VERTEBRAE, "corner_order": CORNER_ORDER,
            }
            if adapter_enabled:
                payload["trainable_state"] = trainable_state_dict(model)
            else:
                payload["probe"] = model.probe_state_dict()
            torch.save(payload, checkpoint_path)
        else:
            epochs_without_improvement += 1
        if early_enabled and epochs_without_improvement >= patience:
            stopped_epoch = epoch
            break
        if scheduler is not None:
            scheduler.step()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restore_probe(model, checkpoint)
    metrics, predictions = evaluate(model, loaders["test"], device, include_predictions=True)
    metrics.update(**metadata, split="test", selected_by="lowest_val_mean_epsilon_dec")
    save_json(output_dir / "test_metrics.json", metrics)
    save_json(output_dir / "test_predictions.json", predictions)
    save_json(output_dir / "training_summary.json", {
        **metadata, **report, "max_epochs": epochs, "stopped_epoch": stopped_epoch,
        "early_stopping_enabled": early_enabled, "patience": patience, "min_delta": min_delta,
        "best_val_metrics": checkpoint["val_metrics"], "checkpoint": checkpoint_name,
    })
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
