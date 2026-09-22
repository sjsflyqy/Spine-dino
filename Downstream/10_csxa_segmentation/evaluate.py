"""Automatic full-image evaluation using the SpineFM table metrics."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from common import device_for, resolve, restore_trainable, save_json, seed_everything, weight_fingerprint
from dataset import CSXAStore
from metrics import SpineFMEvaluator, table_text
from models.auxiliary import load_auxiliaries
from models.segmenter import Segmenter
from pipeline import SpineFMPipeline


def evaluate_split(pipeline, store, split, output=None, save_masks=False):
    reference = SpineFMEvaluator("reference")
    unique = SpineFMEvaluator("one_to_one")
    traces = {}
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if save_masks:
            (output / "masks").mkdir(exist_ok=True)
    for index, identifier in enumerate(store.splits[split]):
        image, gt = store.load(identifier)
        # Ground truth is available only to the evaluators, never to pipeline.predict.
        masks, trace = pipeline.predict(image)
        reference.add(identifier, masks, gt["masks"])
        unique.add(identifier, masks, gt["masks"])
        traces[identifier] = trace
        if output is not None and save_masks:
            array = np.stack(masks) if masks else np.empty((0, image.height, image.width), dtype=bool)
            np.savez_compressed(output / "masks" / f"{identifier}.npz", masks=array)
        if (index+1) % 25 == 0:
            print(f"{split}: {index+1}/{len(store.splits[split])}", flush=True)
    result = reference.summary(store.splits[split])
    result.update(split=split, data_fingerprint=store.fingerprint,
                  mask_probability_threshold=pipeline.mask_threshold,
                  one_to_one_audit=unique.summary(store.splits[split]))
    result["pipeline_diagnostics"] = dict(
        zero_prediction_images=sum(r["predictions"] == 0 for r in reference.records.values()),
        fewer_than_three_detector_candidates=sum(t["candidates"] < 3 for t in traces.values()),
        stop_reasons=dict(Counter(reason for t in traces.values() for reason in t["stops"])))
    if output is not None:
        save_json(output / "metrics.json", result)
        save_json(output / "matches.json", dict(reference=reference.records, one_to_one=unique.records))
        save_json(output / "pipeline_traces.json", traces)
        (output / "table.md").write_text(table_text(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-masks", action="store_true")
    args = parser.parse_args()
    output = resolve(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new evaluation output directory: {output}")
    checkpoint = torch.load(resolve(args.checkpoint), map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed_everything(config["training"].get("seed", 42))
    device = device_for(args.device)
    data = dict(config.get("data", {}))
    if args.data_root:
        data["data_root"] = args.data_root
    if args.cache_root:
        data["cache_root"] = args.cache_root
    store = CSXAStore(**data)
    if checkpoint["data_fingerprint"] != store.fingerprint:
        raise ValueError("Checkpoint was trained with a different GT cache")
    if weight_fingerprint(config["model"]["weights"]) != checkpoint["backbone_fingerprint"]:
        raise ValueError("Frozen backbone weights changed since training")
    # Saved decoder state includes its initialization; no initializer file needed at evaluation.
    model_config = dict(config["model"])
    model_config.pop("sam_initialization", None)
    model = Segmenter(model_config).to(device)
    restore_trainable(model, checkpoint["state_dict"])
    auxiliary, fingerprints = load_auxiliaries(config["auxiliaries"], store, device)
    if fingerprints != checkpoint["auxiliary_fingerprints"]:
        raise ValueError("Auxiliary checkpoints changed since model selection")
    pipeline = SpineFMPipeline(model, auxiliary, device, config.get("pipeline"))
    result = evaluate_split(pipeline, store, args.split, output, args.save_masks)
    save_json(output / "evaluation_config.json", dict(checkpoint=str(resolve(args.checkpoint)), config=config))
    print(table_text(result))


if __name__ == "__main__":
    main()
