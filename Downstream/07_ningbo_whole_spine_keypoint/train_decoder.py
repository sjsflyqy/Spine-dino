"""Train the isolated paired structured decoder protocol for task 07."""

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
from decoder_model import PairedStructuredDecoder
from downstream_experiment_cli import load_experiment_config
from keypoint_decoder import (
    build_structured_targets,
    decode_structured_landmarks,
    structured_landmark_loss,
)
from keypoint_linear_common import (
    build_downstream_optimizer,
    deterministic_split,
    is_adapter_finetune,
    load_trainable_state_dict,
    save_json,
    seed_everything,
    trainable_parameter_report,
    trainable_state_dict,
)


def make_loader(dataset, training: dict, shuffle: bool) -> DataLoader:
    workers = int(training.get("workers", 4))
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def paper_view_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict:
    difference = predicted - target
    scaled = difference * difference.new_tensor((512.0, 1024.0))
    return {
        "epsilon_dec": scaled.square().sum(dim=-1).sqrt().sum().item(),
        "epsilon_MAE": difference.abs().sum(dim=-1).sum().item(),
        "count": difference.shape[0] * difference.shape[1],
    }


@torch.no_grad()
def evaluate(model, loader, device, include_predictions: bool = False):
    model.eval()
    totals = {
        "ap": {"epsilon_dec": 0.0, "epsilon_MAE": 0.0, "count": 0},
        "lat": {"epsilon_dec": 0.0, "epsilon_MAE": 0.0, "count": 0},
    }
    records = []
    for batch in loader:
        ap_images = batch["ap_image"].to(device, non_blocking=True)
        lat_images = batch["lat_image"].to(device, non_blocking=True)
        targets = {
            "ap": batch["ap_points"].to(device, non_blocking=True),
            "lat": batch["lat_points"].to(device, non_blocking=True),
        }
        output = model(ap_images, lat_images)
        decoded = {
            view: decode_structured_landmarks(
                output[view], object_count=17, task="task07"
            )
            for view in ("ap", "lat")
        }
        for view in ("ap", "lat"):
            partial = paper_view_metrics(decoded[view], targets[view])
            for key in totals[view]:
                totals[view][key] += partial[key]
        if include_predictions:
            for index, identifier in enumerate(batch["id"]):
                records.append({
                    "id": identifier,
                    "ap_pred_normalized": decoded["ap"][index].cpu().tolist(),
                    "ap_gt_normalized": targets["ap"][index].cpu().tolist(),
                    "lat_pred_normalized": decoded["lat"][index].cpu().tolist(),
                    "lat_gt_normalized": targets["lat"][index].cpu().tolist(),
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


def main() -> None:
    config = load_experiment_config("Task 07 paired structured landmark decoder")
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
        "protocol": "paired DR identifier; same split as the old probe",
        "seed": seed,
        "splits": splits,
        "excluded": excluded,
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
    weights = Path(config["model"]["weights"]).expanduser()
    if not weights.is_absolute():
        weights = REPO_ROOT / weights
    model = PairedStructuredDecoder(
        config["model"]["backbone"],
        str(weights),
        decoder=config["model"].get("decoder"),
        adapter=config["model"].get("adapter"),
    ).to(device)
    report = trainable_parameter_report(model)
    adapter_enabled = is_adapter_finetune(model)
    if report["head_trainable"] <= 0 or (
        adapter_enabled and report["backbone_trainable"] <= 0
    ) or (not adapter_enabled and report["backbone_trainable"] != 0):
        raise RuntimeError(f"Invalid structured decoder parameters: {report}")
    print(json.dumps({
        "protocol": "task07_structured_decoder_lora" if adapter_enabled
        else "task07_structured_decoder_frozen",
        "device": str(device),
        "adapter_modules": getattr(model.backbone, "adapter_modules", []),
        **report,
        "train_pairs": len(datasets["train"]),
        "val_pairs": len(datasets["val"]),
        "test_pairs": len(datasets["test"]),
        "excluded_pairs": len(excluded),
    }, ensure_ascii=False))

    optimizer, scheduler = build_downstream_optimizer(model, training)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    checkpoint_path = output_dir / (
        "best_lora_decoder.pt" if adapter_enabled else "best_decoder.pt"
    )
    best = float("inf")
    early = training.get("early_stopping", {})
    patience = int(early.get("patience", 20))
    min_delta = float(early.get("min_delta", 0.0))
    epochs_without_improvement = 0
    stopped_epoch = None
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    max_grad_norm = float(training.get("max_grad_norm", 0.0))
    decoder_config = config["model"].get("decoder", {})
    sigma = float(decoder_config.get("heatmap_sigma", 2.0))
    loss_weights = config.get("loss", {})

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        sample_count = 0
        for batch_index, batch in enumerate(loaders["train"]):
            ap_images = batch["ap_image"].to(device, non_blocking=True)
            lat_images = batch["lat_image"].to(device, non_blocking=True)
            points = {
                "ap": batch["ap_points"].to(device, non_blocking=True),
                "lat": batch["lat_points"].to(device, non_blocking=True),
            }
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(ap_images, lat_images)
                loss = output["ap"]["hm"].new_zeros(())
                for view in ("ap", "lat"):
                    targets = build_structured_targets(
                        points[view], output[view]["hm"].shape[-2:],
                        task="task07", heatmap_sigma=sigma,
                    )
                    view_loss, _ = structured_landmark_loss(
                        output[view], targets, loss_weights
                    )
                    loss = loss + view_loss
                loss = loss * 0.5
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
        metrics.update(
            split="val", epoch=epoch,
            train_loss=running_loss / max(sample_count, 1),
        )
        print(json.dumps(metrics, ensure_ascii=False))
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        if metrics["mean_epsilon_dec"] < best - min_delta:
            best = metrics["mean_epsilon_dec"]
            epochs_without_improvement = 0
            torch.save({
                "config": config,
                "trainable_state": trainable_state_dict(model),
                "val_metrics": metrics,
                "vertebrae": VERTEBRAE,
                "corner_order": CORNER_ORDER,
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
        if bool(early.get("enabled", True)) and epochs_without_improvement >= patience:
            stopped_epoch = epoch
            break
        if scheduler is not None:
            scheduler.step()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_trainable_state_dict(model, checkpoint["trainable_state"])
    metrics, predictions = evaluate(
        model, loaders["test"], device, include_predictions=True
    )
    metrics.update(split="test", selected_by="lowest_val_mean_epsilon_dec")
    save_json(output_dir / "test_metrics.json", metrics)
    save_json(output_dir / "test_predictions.json", predictions)
    save_json(output_dir / "training_summary.json", {
        "max_epochs": int(training["epochs"]),
        "stopped_epoch": stopped_epoch,
        "best_val_metrics": checkpoint["val_metrics"],
        "checkpoint": checkpoint_path.name,
    })
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
