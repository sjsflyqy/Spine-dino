from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Downstream"))

from dataset import CORNER_ORDER, VERTEBRAE, PairedWholeSpineDataset, audit_pairs
from linear_probe_cli import load_linear_probe_config
from keypoint_linear_common import (
    build_downstream_optimizer,
    checkpoint_filename,
    deterministic_split,
    gaussian_heatmap_loss,
    is_adapter_finetune,
    load_trainable_state_dict,
    save_json,
    seed_everything,
    soft_argmax_2d,
    trainable_parameter_report,
    trainable_state_dict,
)
from model import PairedKeypointLinearProbe


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


def paper_view_metrics(predicted, target):
    difference = predicted - target
    scaled = difference * difference.new_tensor((512.0, 1024.0))
    return {
        "epsilon_dec": scaled.square().sum(dim=-1).sqrt().sum().item(),
        "epsilon_MAE": difference.abs().sum(dim=-1).sum().item(),
        "count": difference.shape[0] * difference.shape[1],
    }


@torch.no_grad()
def evaluate(model, loader, device, include_predictions=False):
    model.eval()
    totals = {
        "ap": {"epsilon_dec": 0.0, "epsilon_MAE": 0.0, "count": 0},
        "lat": {"epsilon_dec": 0.0, "epsilon_MAE": 0.0, "count": 0},
    }
    records = []
    for batch in loader:
        ap_images = batch["ap_image"].to(device, non_blocking=True)
        lat_images = batch["lat_image"].to(device, non_blocking=True)
        ap_target = batch["ap_points"].to(device, non_blocking=True)
        lat_target = batch["lat_points"].to(device, non_blocking=True)
        output = model(ap_images, lat_images)
        decoded = {"ap": soft_argmax_2d(output["ap"]), "lat": soft_argmax_2d(output["lat"])}
        for view, target in (("ap", ap_target), ("lat", lat_target)):
            partial = paper_view_metrics(decoded[view], target)
            for key in totals[view]:
                totals[view][key] += partial[key]
        if include_predictions:
            for index, identifier in enumerate(batch["id"]):
                records.append({
                    "id": identifier,
                    "ap_pred_normalized": decoded["ap"][index].cpu().tolist(),
                    "ap_gt_normalized": ap_target[index].cpu().tolist(),
                    "lat_pred_normalized": decoded["lat"][index].cpu().tolist(),
                    "lat_gt_normalized": lat_target[index].cpu().tolist(),
                })
    metrics = {}
    for view in ("ap", "lat"):
        count = max(totals[view]["count"], 1)
        metrics[f"{view}_epsilon_dec"] = totals[view]["epsilon_dec"] / count
        metrics[f"{view}_epsilon_MAE"] = totals[view]["epsilon_MAE"] / count
    metrics["mean_epsilon_dec"] = 0.5 * (
        metrics["ap_epsilon_dec"] + metrics["lat_epsilon_dec"]
    )
    metrics["mean_epsilon_MAE"] = 0.5 * (
        metrics["ap_epsilon_MAE"] + metrics["lat_epsilon_MAE"]
    )
    return metrics, records


def main():
    config = load_linear_probe_config("Task 07 whole-spine linear probe")
    training = config["training"]
    seed = int(training.get("seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = REPO_ROOT / "Downstream/data/task07"
    clean, excluded = audit_pairs(data_root)
    splits = deterministic_split(list(clean), seed)
    output_dir = (REPO_ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "split_and_audit.json", {
        "protocol": "paired DR identifier; 20% test, 10% of remainder validation",
        "seed": seed, "splits": splits, "excluded": excluded,
    })

    height = int(config["model"].get("image_height", 896))
    width = int(config["model"].get("image_width", 448))
    datasets = {
        split: PairedWholeSpineDataset(
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
    model = PairedKeypointLinearProbe(
        config["model"]["backbone"], weights,
        adapter=config["model"].get("adapter"),
    ).to(device)
    report = trainable_parameter_report(model)
    expected = 2 * 68 * (2 * model.backbone.embedding_dim + 1)
    adapter_enabled = is_adapter_finetune(model)
    if report["head_trainable"] != expected or (
        adapter_enabled and report["backbone_trainable"] <= 0
    ) or (not adapter_enabled and report["backbone_trainable"] != 0):
        raise RuntimeError(f"Invalid downstream trainable parameters: {report}, head_expected={expected}")
    print(json.dumps({
        "protocol": "paired AP/LAT dense LoRA fine-tune" if adapter_enabled else "paired AP/LAT dense linear landmark probe",
        "adapter_modules": getattr(model.backbone, "adapter_modules", []),
        "device": str(device), **report,
        "train_pairs": len(datasets["train"]), "val_pairs": len(datasets["val"]),
        "test_pairs": len(datasets["test"]), "excluded_pairs": len(excluded),
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
            ap_images = batch["ap_image"].to(device, non_blocking=True)
            lat_images = batch["lat_image"].to(device, non_blocking=True)
            ap_target = batch["ap_points"].to(device, non_blocking=True)
            lat_target = batch["lat_points"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(ap_images, lat_images)
                loss = gaussian_heatmap_loss(output["ap"], ap_target, sigma)
                loss = loss + gaussian_heatmap_loss(output["lat"], lat_target, sigma)
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
            running_loss += loss.item() * ap_images.shape[0]
            sample_count += ap_images.shape[0]

        metrics, _ = evaluate(model, loaders["val"], device)
        metrics.update(split="val", epoch=epoch, train_loss=running_loss / max(sample_count, 1))
        print(json.dumps(metrics, ensure_ascii=False))
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        if metrics["mean_epsilon_dec"] < best - min_delta:
            best = metrics["mean_epsilon_dec"]
            epochs_without_improvement = 0
            payload = {
                "config": config,
                "val_metrics": metrics, "vertebrae": VERTEBRAE,
                "corner_order": CORNER_ORDER,
            }
            if adapter_enabled:
                payload["trainable_state"] = trainable_state_dict(model)
            else:
                payload["probe"] = model.probe_state_dict()
            torch.save(payload, checkpoint_path)
        else:
            epochs_without_improvement += 1
        if early_stopping_enabled and epochs_without_improvement >= patience:
            stopped_epoch = epoch
            print(json.dumps({
                "early_stopping": True, "stopped_epoch": epoch,
                "patience": patience, "best_val_mean_epsilon_dec": best,
            }, ensure_ascii=False))
            break
        if scheduler is not None:
            scheduler.step()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if adapter_enabled:
        load_trainable_state_dict(model, checkpoint["trainable_state"])
    else:
        model.load_probe_state_dict(checkpoint["probe"])
    metrics, predictions = evaluate(model, loaders["test"], device, include_predictions=True)
    metrics.update(split="test", selected_by="lowest_val_mean_epsilon_dec")
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
