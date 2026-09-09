"""Train the isolated structured decoder protocol for downstream task 08."""

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
from decoder_model import LumbarStructuredDecoder
from downstream_experiment_cli import load_experiment_config
from keypoint_decoder import (
    build_structured_targets,
    decode_structured_landmarks,
    structured_landmark_loss,
)
from keypoint_linear_common import (
    build_downstream_optimizer,
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


def case_grouped_split(clean: dict, seed: int) -> tuple[dict, dict]:
    grouped: dict[str, list[str]] = {}
    for identifier in clean:
        match = re.fullmatch(r"(?:gq|gs)?(\d+)", identifier, flags=re.IGNORECASE)
        case_id = match.group(1) if match else identifier
        grouped.setdefault(case_id, []).append(identifier)
    cases = sorted(
        grouped,
        key=lambda value: (
            not value.isdigit(), int(value) if value.isdigit() else value
        ),
    )
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
            identifier
            for case_id in case_ids
            for identifier in grouped[case_id]
        )
        for split, case_ids in case_splits.items()
    }
    return splits, case_splits


@torch.no_grad()
def evaluate(
    model, loader, device, pixel_spacing_mm: float, include_predictions: bool = False
):
    model.eval()
    l1_sum = radial_sum = 0.0
    count = 0
    successful = {threshold: 0 for threshold in (2, 3, 4, 5)}
    records = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["points"].to(device, non_blocking=True)
        sizes = batch["original_size"].to(device, non_blocking=True)
        predicted = decode_structured_landmarks(
            model(images), object_count=6, task="task08"
        )
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


def main() -> None:
    config = load_experiment_config("Task 08 structured landmark decoder")
    training = config["training"]
    seed = int(training.get("seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = REPO_ROOT / "Downstream/data/task08"
    clean, excluded = audit_images(data_root)
    splits, case_splits = case_grouped_split(clean, seed)
    output_dir = (REPO_ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "split_and_audit.json", {
        "protocol": "case-grouped N/gqN/gsN split; same split as the old probe",
        "seed": seed,
        "case_splits": case_splits,
        "splits": splits,
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
    weights = Path(config["model"]["weights"]).expanduser()
    if not weights.is_absolute():
        weights = REPO_ROOT / weights
    model = LumbarStructuredDecoder(
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
        "protocol": "task08_structured_decoder_lora" if adapter_enabled
        else "task08_structured_decoder_frozen",
        "device": str(device),
        "adapter_modules": getattr(model.backbone, "adapter_modules", []),
        **report,
        "train_images": len(datasets["train"]),
        "val_images": len(datasets["val"]),
        "test_images": len(datasets["test"]),
        "excluded_images": len(excluded),
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
    spacing = float(config["evaluation"].get("pixel_spacing_mm", 0.143))

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        sample_count = 0
        for batch_index, batch in enumerate(loaders["train"]):
            images = batch["image"].to(device, non_blocking=True)
            points = batch["points"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(images)
                targets = build_structured_targets(
                    points, output["hm"].shape[-2:], task="task08",
                    heatmap_sigma=sigma,
                )
                loss, _ = structured_landmark_loss(output, targets, loss_weights)
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
        metrics.update(
            split="val", epoch=epoch,
            train_loss=running_loss / max(sample_count, 1),
        )
        print(json.dumps(metrics, ensure_ascii=False))
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        if metrics["MRE_mm"] < best - min_delta:
            best = metrics["MRE_mm"]
            epochs_without_improvement = 0
            torch.save({
                "config": config,
                "trainable_state": trainable_state_dict(model),
                "val_metrics": metrics,
                "landmark_names": LANDMARK_NAMES,
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
        model, loaders["test"], device, spacing, include_predictions=True
    )
    metrics.update(split="test", selected_by="lowest_val_MRE_mm")
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
