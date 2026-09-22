"""Train common automatic-localization helpers on the 200 training images only."""
from __future__ import annotations

import argparse
import json
import math

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from common import device_for, load_config, resolve, save_json, seed_everything, weight_fingerprint
from dataset import (CSXAStore, ClassificationDataset, DetectionDataset, PointDataset,
                     detection_collate)
from metrics import SpineFMEvaluator
from models.auxiliary import build_auxiliary


def train_aux_epoch(model, loader, optimizer, stage, device):
    model.train()
    total, count = 0., 0
    for inputs, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        if stage == "detector":
            inputs = [x.to(device) for x in inputs]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            loss = sum(model(inputs, targets).values())
        else:
            predictions = model(inputs.to(device))
            loss = F.cross_entropy(predictions, targets.to(device)) if stage == "classifier" else F.mse_loss(predictions, targets.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {stage} loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        total += float(loss.detach()) * len(inputs)
        count += len(inputs)
    return total / max(count, 1)


@torch.no_grad()
def validate_aux(model, loader, stage, device):
    model.eval()
    total, count = 0., 0
    evaluator = SpineFMEvaluator()
    for inputs, targets in loader:
        if stage == "detector":
            outputs = model([x.to(device) for x in inputs])
            for output, target in zip(outputs, targets):
                keep = output["scores"] > 0.6
                masks = output["masks"][keep, 0].cpu().numpy() > 0.5
                identifier = loader.dataset.ids[int(target["image_id"].item())]
                evaluator.add(identifier, masks, target["masks"].numpy())
        else:
            predictions = model(inputs.to(device))
            if stage == "classifier":
                total += int((predictions.argmax(1).cpu() == targets).sum())
            else:
                total -= float(F.mse_loss(predictions, targets.to(device), reduction="sum")) / 2
            count += len(inputs)
    if stage == "detector":
        metrics = evaluator.summary(loader.dataset.ids)
        return metrics["table"]["Avg"]["overall_dsc"], metrics
    score = total / max(count, 1)
    return score, {"accuracy" if stage == "classifier" else "negative_coordinate_mse": score}


def run_stage(stage, config, store, device, epochs_override=None):
    settings = config[stage]
    output = resolve(config["output_dir"])
    checkpoint_path = output / f"{stage}.pt"
    if checkpoint_path.exists() or (output / f"{stage}_metrics.jsonl").exists():
        raise FileExistsError(f"Existing {stage} run: choose a new auxiliary output directory")
    seed_everything(int(config.get("seed", 42)))
    datasets = {"detector": DetectionDataset, "classifier": ClassificationDataset, "point_predictor": PointDataset}
    loaders = {}
    for split in ("train", "val"):
        dataset = datasets[stage](store, split)
        workers = int(settings.get("workers", 4))
        loaders[split] = DataLoader(dataset, batch_size=int(settings.get("batch_size", 4)),
                                    shuffle=split == "train", num_workers=workers,
                                    collate_fn=detection_collate if stage == "detector" else None,
                                    persistent_workers=workers > 0, pin_memory=device.type == "cuda")
    model_config = settings.get("model", {})
    model = build_auxiliary(stage, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
                                  weight_decay=float(settings.get("weight_decay", 0.0001)))
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(stage=stage, model_config=model_config, data_fingerprint=store.fingerprint,
                      fit_split="train", selection_split="val", train_ids=store.splits["train"],
                      val_ids=store.splits["val"], seed=config.get("seed", 42),
                      initialization=(weight_fingerprint(model_config["initialization"])
                                      if model_config.get("initialization") else "random, fixed seed"))
    save_json(output / f"{stage}_config.json", dict(config=config, provenance=provenance,
                                                 epochs_override=epochs_override))
    epochs = int(epochs_override or settings["epochs"])
    best = -math.inf
    for epoch in range(1, epochs+1):
        loss = train_aux_epoch(model, loaders["train"], optimizer, stage, device)
        row = dict(epoch=epoch, train_loss=loss)
        if epoch % int(settings.get("validate_every", 1)) == 0 or epoch == epochs:
            score, metrics = validate_aux(model, loaders["val"], stage, device)
            row["validation"] = metrics
            if score > best:
                best = score
                state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(dict(state_dict=state, epoch=epoch, validation=metrics, **provenance), checkpoint_path)
        with (output / f"{stage}_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(stage, json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Downstream/10_csxa_segmentation/configs/auxiliary.yaml")
    parser.add_argument("--stage", choices=["all", "detector", "classifier", "point_predictor"], default="all")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    store = CSXAStore(**config.get("data", {}))
    device = device_for(args.device)
    stages = ["detector", "classifier", "point_predictor"] if args.stage == "all" else [args.stage]
    for stage in stages:
        run_stage(stage, config, store, device, args.epochs)


if __name__ == "__main__":
    main()
