from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Downstream"))

from dataset import LANDMARK_NAMES, LumbarLandmarkDataset, audit_images
from linear_probe_cli import load_linear_probe_config
from keypoint_linear_common import (
    build_downstream_optimizer,
    checkpoint_filename,
    gaussian_heatmap_loss,
    is_adapter_finetune,
    load_trainable_state_dict,
    save_json,
    seed_everything,
    soft_argmax_2d,
    trainable_parameter_report,
    trainable_state_dict,
)
from model import LumbarKeypointLinearProbe


def make_loader(dataset, training, shuffle):
    workers = int(training.get("workers", 4))
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def evaluate(model, loader, device, pixel_spacing_mm, include_predictions=False):
    model.eval()
    l1_sum = 0.0
    radial_sum = 0.0
    count = 0
    successful = {threshold: 0 for threshold in (2, 3, 4, 5)}
    records = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["points"].to(device, non_blocking=True)
        sizes = batch["original_size"].to(device, non_blocking=True)
        predicted = soft_argmax_2d(model(images))
        difference_mm = (predicted - target) * sizes[:, None, :] * pixel_spacing_mm
        radial = difference_mm.square().sum(dim=-1).sqrt()
        l1 = difference_mm.abs().sum(dim=-1)
        l1_sum += l1.sum().item()
        radial_sum += radial.sum().item()
        count += radial.numel()
        for threshold in successful:
            successful[threshold] += (radial <= threshold).sum().item()
        if include_predictions:
            for index, identifier in enumerate(batch["id"]):
                records.append({
                    "id": identifier,
                    "pred_normalized": predicted[index].cpu().tolist(),
                    "gt_normalized": target[index].cpu().tolist(),
                    "radial_error_mm": radial[index].cpu().tolist(),
                })
    metrics = {
        "MAE_mm": l1_sum / max(count, 1),
        "MRE_mm": radial_sum / max(count, 1),
        "landmarks": count,
    }
    for threshold, value in successful.items():
        metrics[f"SDR_{threshold}mm_percent"] = 100.0 * value / max(count, 1)
    return metrics, records


def main():
    config = load_linear_probe_config("Task 08 flexion-extension linear probe")
    training = config["training"]
    seed = int(training.get("seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = REPO_ROOT / "Downstream/data/task08"
    clean, excluded = audit_images(data_root)
    # Filenames retain the original case relationship: N is the neutral LAT
    # view, gqN is hyperflexion, and gsN is hyperextension. Split by N so no
    # view from one patient can leak across train/validation/test.
    grouped: dict[str, list[str]] = {}
    for identifier in clean:
        match = re.fullmatch(r"(?:gq|gs)?(\d+)", identifier, flags=re.IGNORECASE)
        case_id = match.group(1) if match else identifier
        grouped.setdefault(case_id, []).append(identifier)
    cases = sorted(grouped, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    random.Random(seed).shuffle(cases)
    test_cases = cases[: min(87, len(cases) - 2)]
    remainder = cases[len(test_cases):]
    validation_count = max(1, round(0.1 * len(remainder)))
    case_splits = {
        "test": test_cases,
        "val": remainder[:validation_count],
        "train": remainder[validation_count:],
    }
    splits = {
        split: sorted(
            [identifier for case_id in case_ids for identifier in grouped[case_id]]
        )
        for split, case_ids in case_splits.items()
    }
    output_dir = (REPO_ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "split_and_audit.json", {
        "protocol": "case-grouped split using N/gqN/gsN; 87 test cases as in paper",
        "seed": seed, "case_splits": case_splits, "splits": splits,
        "excluded": excluded,
    })

    height = int(config["model"].get("image_height", 896))
    width = int(config["model"].get("image_width", 448))
    datasets = {
        split: LumbarLandmarkDataset(
            data_root, identifiers, clean, height, width, split == "train"
        )
        for split, identifiers in splits.items()
    }
    loaders = {
        split: make_loader(dataset, training, split == "train")
        for split, dataset in datasets.items()
    }
    weights = config["model"].get("weights")
    if weights and not Path(weights).is_absolute():
        weights = str(REPO_ROOT / weights)
    model = LumbarKeypointLinearProbe(
        config["model"]["backbone"], weights,
        adapter=config["model"].get("adapter"),
    ).to(device)
    report = trainable_parameter_report(model)
    expected = 22 * (model.backbone.embedding_dim + 1)
    adapter_enabled = is_adapter_finetune(model)
    if report["head_trainable"] != expected or (
        adapter_enabled and report["backbone_trainable"] <= 0
    ) or (not adapter_enabled and report["backbone_trainable"] != 0):
        raise RuntimeError(f"Invalid downstream trainable parameters: {report}, head_expected={expected}")
    print(json.dumps({
        "protocol": "single-image dense LoRA 22-landmark fine-tune" if adapter_enabled else "single-image dense linear 22-landmark probe",
        "adapter_modules": getattr(model.backbone, "adapter_modules", []),
        "device": str(device), **report,
        "train_images": len(datasets["train"]), "val_images": len(datasets["val"]),
        "test_images": len(datasets["test"]), "excluded_images": len(excluded),
    }))

    optimizer, scheduler = build_downstream_optimizer(model, training)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best = float("inf")
    early_stopping = training.get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping.get("enabled", False))
    patience = int(early_stopping.get("patience", 15))
    min_delta = float(early_stopping.get("min_delta", 0.0))
    epochs_without_improvement = 0
    stopped_epoch = None
    sigma = float(training.get("heatmap_sigma", 1.5))
    spacing = float(config["evaluation"].get("pixel_spacing_mm", 0.143))
    checkpoint_path = output_dir / checkpoint_filename(model)
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    max_grad_norm = float(training.get("max_grad_norm", 0.0))

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loaders["train"]):
            images = batch["image"].to(device, non_blocking=True)
            target = batch["points"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = gaussian_heatmap_loss(logits, target, sigma)
            scaler.scale(loss / accumulation_steps).backward()
            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(loaders["train"])
            )
            if should_step:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * images.shape[0]
            sample_count += images.shape[0]

        metrics, _ = evaluate(model, loaders["val"], device, spacing)
        metrics.update(split="val", epoch=epoch, train_loss=running_loss / max(sample_count, 1))
        print(json.dumps(metrics, ensure_ascii=False))
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        if metrics["MRE_mm"] < best - min_delta:
            best = metrics["MRE_mm"]
            epochs_without_improvement = 0
            payload = {
                "config": config,
                "val_metrics": metrics, "landmark_names": LANDMARK_NAMES,
            }
            if adapter_enabled:
                payload["trainable_state"] = trainable_state_dict(model)
            else:
                payload["head"] = model.head.state_dict()
            torch.save(payload, checkpoint_path)
        else:
            epochs_without_improvement += 1
        if early_stopping_enabled and epochs_without_improvement >= patience:
            stopped_epoch = epoch
            print(json.dumps({
                "early_stopping": True, "stopped_epoch": epoch,
                "patience": patience, "best_val_MRE_mm": best,
            }, ensure_ascii=False))
            break
        if scheduler is not None:
            scheduler.step()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if adapter_enabled:
        load_trainable_state_dict(model, checkpoint["trainable_state"])
    else:
        model.head.load_state_dict(checkpoint["head"])
    metrics, predictions = evaluate(
        model, loaders["test"], device, spacing, include_predictions=True
    )
    metrics.update(split="test", selected_by="lowest_val_MRE_mm")
    save_json(output_dir / "test_metrics.json", metrics)
    save_json(output_dir / "test_predictions.json", predictions)
    save_json(output_dir / "training_summary.json", {
        "max_epochs": int(training["epochs"]),
        "stopped_epoch": stopped_epoch,
        "early_stopping_enabled": early_stopping_enabled,
        "patience": patience,
        "min_delta": min_delta,
        "best_val_metrics": checkpoint["val_metrics"],
    })
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
    is_adapter_finetune,
    load_trainable_state_dict,
