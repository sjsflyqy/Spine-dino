"""Train a linear/SAM segmentation head with a frozen or LoRA-adapted encoder."""
from __future__ import annotations

import argparse
import json
import math

import torch
from torch.utils.data import DataLoader

from common import (device_for, load_config, parameter_counts, resolve, save_json,
                    seed_everything, trainable_state, weight_fingerprint)
from dataset import CSXAStore, SegmentationDataset
from evaluate import evaluate_split
from models.auxiliary import load_auxiliaries
from models.segmenter import Segmenter, segmentation_loss
from pipeline import SpineFMPipeline


def train_epoch(model, loader, optimizer, device, accumulation=1, max_grad_norm=1.0, mixed_precision=True):
    model.train()
    amp = device.type == "cuda" and mixed_precision
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
    total, examples, pending = 0., 0, 0
    optimizer.zero_grad(set_to_none=True)
    for index, batch in enumerate(loader):
        images, points, targets = (batch[k].to(device) for k in ("image", "point", "mask"))
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            logits, quality = model(images, points)
            loss = segmentation_loss(logits, targets, quality)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite segmentation loss")
        # Accumulate sums and divide by actual examples, including a partial final batch.
        scaler.scale(loss * len(images)).backward()
        pending += len(images)
        total += float(loss.detach()) * len(images)
        examples += len(images)
        if (index+1) % accumulation == 0 or index+1 == len(loader):
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(pending)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
    return total / max(examples, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", help="Override the evaluated backbone with a local checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.weights:
        config["model"]["weights"] = args.weights
    if args.output_dir:
        config["output_dir"] = args.output_dir
    output = resolve(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output directory: {output}")
    training = config["training"]
    seed_everything(int(training.get("seed", 42)))
    device = device_for(args.device)
    store = CSXAStore(**config.get("data", {}))
    auxiliaries, auxiliary_fingerprints = load_auxiliaries(config["auxiliaries"], store, device)
    # Auxiliary construction consumes RNG; reset for identical head initialization across runs.
    seed_everything(int(training.get("seed", 42)))
    model = Segmenter(config["model"]).to(device)
    backbone_fingerprint = weight_fingerprint(config["model"]["weights"])
    counts = parameter_counts(model)
    print(json.dumps(counts), flush=True)
    if not model.uses_lora and counts["backbone_trainable"] != 0:
        raise RuntimeError("Frozen-backbone protocol violated")
    if model.uses_lora and counts["backbone_trainable"] == 0:
        raise RuntimeError("No trainable LoRA modules found")
    dataset = SegmentationDataset(store, "train", model.input_size,
                                   config.get("pipeline", {}).get("patch_size", 300), model.target_size,
                                   repeats=int(training.get("patch_repeats", 1)))
    workers = int(training.get("workers", 4))
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True,
                        num_workers=workers, persistent_workers=workers > 0,
                        pin_memory=device.type == "cuda")
    groups = []
    for prefix, lr in [("backbone.", training.get("adapter_learning_rate", 1e-4)),
                       (None, training.get("learning_rate", 1e-3))]:
        parameters = [p for n, p in model.named_parameters() if p.requires_grad and
                      (n.startswith("backbone.") if prefix else not n.startswith("backbone."))]
        if parameters:
            groups.append(dict(params=parameters, lr=float(lr)))
    optimizer = torch.optim.AdamW(groups, weight_decay=float(training.get("weight_decay", 0.01)))
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pipeline = SpineFMPipeline(model, auxiliaries, device, config.get("pipeline"))
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", config)
    save_json(output / "split_and_audit.json", dict(splits=store.splits, data_fingerprint=store.fingerprint,
                                                   annotation_audit=json.loads((store.cache / "annotation_audit.json").read_text())))
    provenance = dict(data_fingerprint=store.fingerprint, backbone_fingerprint=backbone_fingerprint,
                      auxiliary_fingerprints=auxiliary_fingerprints, parameters=counts,
                      decoder_initialization=(weight_fingerprint(config["model"]["sam_initialization"])
                                              if config["model"].get("sam_initialization") else "random, fixed seed"))
    save_json(output / "provenance.json", provenance)
    best, stale = -math.inf, 0
    for epoch in range(1, epochs+1):
        loss = train_epoch(model, loader, optimizer, device,
                           int(training.get("gradient_accumulation_steps", 1)),
                           float(training.get("max_grad_norm", 1.0)),
                           bool(training.get("mixed_precision", True)))
        scheduler.step()
        row = dict(epoch=epoch, train_loss=loss)
        if epoch % int(training.get("validate_every", 5)) == 0 or epoch == epochs:
            validation = evaluate_split(pipeline, store, "val")
            row["validation"] = validation
            score = validation["table"]["Avg"]["overall_dsc"]
            if score > best:
                best, stale = score, 0
                torch.save(dict(state_dict=trainable_state(model), config=config, epoch=epoch,
                                validation=validation, fit_split="train", selection_split="val", **provenance),
                           output / "best_segmentation.pt")
                save_json(output / "best_validation_metrics.json", validation)
            else:
                stale += 1
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
        patience = int(training.get("patience", 0))
        if patience and stale >= patience:
            break
    print(f"Best validation Overall DSC: {best:.4f}. Run evaluate.py for the held-out test set.")


if __name__ == "__main__":
    main()
