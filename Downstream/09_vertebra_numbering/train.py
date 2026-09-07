from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Downstream"))

from dataset import VertebraCocoDataset, collate_vertebra_batch
from linear_probe_cli import load_linear_probe_config
from model import VertebraLinearProbe
from keypoint_linear_common import (
    build_downstream_optimizer,
    checkpoint_filename,
    is_adapter_finetune,
    load_trainable_state_dict,
    trainable_parameter_report,
    trainable_state_dict,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    batch["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True)
    batch["boxes"] = [boxes.to(device, non_blocking=True) for boxes in batch["boxes"]]
    batch["labels"] = [labels.to(device, non_blocking=True) for labels in batch["labels"]]
    return batch


def flatten_labels(labels):
    nonempty = [item for item in labels if len(item)]
    if not nonempty:
        return torch.empty(0, dtype=torch.long, device=labels[0].device)
    return torch.cat(nonempty)


@torch.no_grad()
def evaluate(model, loader, device, num_classes=18):
    """Paper metrics: instance-level IDR and image-level exact-match IRA."""
    model.eval()
    instance_correct = 0
    instance_total = 0
    image_correct = 0
    image_total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["pixel_values"], batch["boxes"])
        predictions = output["logits"].argmax(dim=1)
        labels = flatten_labels(batch["labels"])

        instance_correct += (predictions == labels).sum().item()
        instance_total += labels.numel()
        offset = 0
        for count in output["counts"]:
            predicted_image = predictions[offset : offset + count]
            target_image = labels[offset : offset + count]
            image_correct += int(
                count > 0 and torch.equal(predicted_image, target_image)
            )
            image_total += 1
            offset += count

        pairs = labels * num_classes + predictions
        confusion += torch.bincount(
            pairs.cpu(), minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)

    idr = 100.0 * instance_correct / max(instance_total, 1)
    ira = 100.0 * image_correct / max(image_total, 1)
    per_class_total = confusion.sum(dim=1)
    per_class_correct = confusion.diag()
    per_class_accuracy = [
        100.0 * int(correct) / int(total) if total else 0.0
        for correct, total in zip(per_class_correct, per_class_total)
    ]
    return {
        "IDR": idr,
        "IRA": ira,
        "instances": instance_total,
        "images": image_total,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
    }


def make_loader(dataset, config, shuffle):
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["training"].get("workers", 4),
        pin_memory=True,
        collate_fn=collate_vertebra_batch,
        persistent_workers=config["training"].get("workers", 4) > 0,
    )


def main():
    config = load_linear_probe_config("Task 09 vertebra-numbering linear probe")
    seed_everything(config["training"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = REPO_ROOT / "Downstream/data/task09/annotations"
    image_size = config["model"].get("image_size", 518)
    train_set = VertebraCocoDataset(data_root / "train.json", image_size, train=True)
    val_set = VertebraCocoDataset(data_root / "val.json", image_size, train=False)
    test_set = VertebraCocoDataset(data_root / "test.json", image_size, train=False)
    train_loader = make_loader(train_set, config, True)
    val_loader = make_loader(val_set, config, False)
    test_loader = make_loader(test_set, config, False)

    weights = config["model"].get("weights")
    if weights and not Path(weights).is_absolute():
        weights = str(REPO_ROOT / weights)
    model = VertebraLinearProbe(
        config["model"]["backbone"], weights,
        adapter=config["model"].get("adapter"),
    ).to(device)
    report = trainable_parameter_report(model)
    adapter_enabled = is_adapter_finetune(model)
    expected_head = 768 * 18 + 18
    if report["head_trainable"] != expected_head or (
        adapter_enabled and report["backbone_trainable"] <= 0
    ) or (not adapter_enabled and report["backbone_trainable"] != 0):
        raise RuntimeError(
            f"Invalid downstream trainable parameters: {report}, head_expected={expected_head}"
        )
    print(json.dumps({
        "protocol": "GT-box LoRA identification fine-tune" if adapter_enabled else "GT-box linear identification probe",
        "device": str(device),
        **report,
        "adapter_modules": getattr(model.backbone, "adapter_modules", []),
        "train_images": len(train_set),
        "val_images": len(val_set),
        "test_images": len(test_set),
    }))

    optimizer, scheduler = build_downstream_optimizer(model, config["training"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = (REPO_ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_key = (-1.0, -1.0)
    early_stopping = config["training"].get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping.get("enabled", False))
    patience = int(early_stopping.get("patience", 15))
    min_delta = float(early_stopping.get("min_delta", 0.0))
    epochs_without_improvement = 0
    stopped_epoch = None
    checkpoint_path = output_dir / checkpoint_filename(model)
    accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    max_grad_norm = float(config["training"].get("max_grad_norm", 0.0))

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        region_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            labels = flatten_labels(batch["labels"])
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(batch["pixel_values"], batch["boxes"])
                loss = F.cross_entropy(output["logits"], labels)
            scaler.scale(loss / accumulation_steps).backward()
            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * labels.numel()
            region_count += labels.numel()

        metrics = evaluate(model, val_loader, device)
        metrics.update(
            split="val", epoch=epoch,
            train_loss=running_loss / max(region_count, 1),
        )
        print(json.dumps(metrics, ensure_ascii=False))
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        current_key = (metrics["IRA"], metrics["IDR"])
        ira_improved = current_key[0] > best_key[0] + min_delta
        idr_improved = (
            abs(current_key[0] - best_key[0]) <= min_delta
            and current_key[1] > best_key[1] + min_delta
        )
        if ira_improved or idr_improved:
            best_key = current_key
            epochs_without_improvement = 0
            payload = {
                "config": config,
                "category_names": train_set.category_names,
                "val_metrics": metrics,
            }
            if adapter_enabled:
                payload["trainable_state"] = trainable_state_dict(model)
            else:
                payload["classifier"] = model.classifier.state_dict()
            torch.save(payload, checkpoint_path)
        else:
            epochs_without_improvement += 1
        if early_stopping_enabled and epochs_without_improvement >= patience:
            stopped_epoch = epoch
            print(json.dumps({
                "early_stopping": True, "stopped_epoch": epoch,
                "patience": patience, "best_val_IRA": best_key[0],
                "best_val_IDR": best_key[1],
            }, ensure_ascii=False))
            break
        if scheduler is not None:
            scheduler.step()

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if adapter_enabled:
        load_trainable_state_dict(model, checkpoint["trainable_state"])
    else:
        model.classifier.load_state_dict(checkpoint["classifier"])
    test_metrics = evaluate(model, test_loader, device)
    test_metrics.update(split="test", selected_by="val_IRA_then_IDR")
    print(json.dumps(test_metrics, ensure_ascii=False))
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps({
            "max_epochs": int(config["training"]["epochs"]),
            "stopped_epoch": stopped_epoch,
            "early_stopping_enabled": early_stopping_enabled,
            "patience": patience,
            "min_delta": min_delta,
            "best_val_metrics": checkpoint["val_metrics"],
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
