"""Re-evaluate saved attention and baseline masks; no model inference or training."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from methods.geotopo_dino.masking.spine_span import SpanConfig, generate_mask_comparison
from visualization.spine_masks.visualize import _save_case, _write_overview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--span-fraction", type=float, default=0.5)
    parser.add_argument("--band-mask-ratio", type=float, default=0.5)
    parser.add_argument("--context-rows", type=int, default=2)
    args = parser.parse_args()
    source, output = args.from_run.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = json.loads((source / "run_config.json").read_text())
    previous = json.loads((source / "summary.json").read_text())
    settings = {**manifest["span_config"], "span_fraction": args.span_fraction,
                "band_mask_ratio": args.band_mask_ratio, "context_rows": args.context_rows}
    config = SpanConfig(**settings)
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    new_manifest = {**manifest, "algorithm_version": 2, "span_config": asdict(config),
                    "source_run_arguments": manifest["arguments"],
                    "arguments": {**manifest["arguments"], "output_dir": str(output),
                                  "span_fraction": args.span_fraction,
                                  "band_mask_ratio": args.band_mask_ratio, "context_rows": args.context_rows},
                    "replayed_from": str(source), "model_inference_performed": False,
                    "replay_arguments": {key: str(value) if isinstance(value, Path) else value
                                         for key, value in vars(args).items()},
                    "training_integration": False,
                    "mask_display": "mask_steps.png: green=band, blue=span, purple=supplement, orange=visible context"}
    (output / "run_config.json").write_text(json.dumps(new_manifest, indent=2) + "\n")
    completed = []
    map_keys = ("attention_heads", "attention_patch_mass", "selected_attention", "mean_attention",
                "cosine", "selected_layer_cosine")
    for old_record in previous["completed"]:
        relative = Path(old_record["output"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Cached case output must be relative to its run")
        origin = source / relative
        with np.load(origin / "masks_and_maps.npz", allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        details = json.loads((origin / "metadata.json").read_text())
        valid = torch.from_numpy(arrays["valid_mask"])
        response = torch.from_numpy(arrays["response"])
        block = torch.from_numpy(arrays["mask_block"])
        result = generate_mask_comparison(response, valid, block, config=config, seed=details["seed"] + 2)
        with Image.open(origin / "anchor.png") as image:
            anchor = image.convert("RGB")
        _save_case(output / relative, anchor, {name: arrays[name] for name in map_keys}, response,
                   valid, result, details, layer=manifest["arguments"]["layer"],
                   heads=manifest["selected_heads_one_based"], source=manifest["arguments"]["response_source"],
                   title=f"{old_record['image_name']} | {old_record['mode']}", old_arrays=arrays)
        record = {key: old_record[key] for key in ("image_name", "image", "mode", "output")}
        record.update(result.diagnostics)
        completed.append(record)
        print(f"{record['image_name']} [{record['mode']}]: band={record['band_count']}, "
              f"span={record['span_count']}, band masked={record['band_mask_ratio_actual']:.1%}, "
              f"fallback={record['fallback_reasons']}", flush=True)
    summary = {"completed": completed, "failures": [], "success_count": len(completed), "failure_count": 0,
               "topology_usage_rate": sum(not r["used_fallback"] for r in completed) / max(len(completed), 1)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ("image_name", "mode", "used_fallback", "fallback_reasons", "coverage", "contrast", "border_mass",
              "valid_count", "target_count", "band_count", "span_count", "span_rows", "span_band_fraction",
              "span_selected_segment_fraction", "band_mask_ratio_actual", "span_total_mask_fraction",
              "context_count", "band_budget_adjusted", "output")
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(completed)
    _write_overview(output, completed)
    _write_overview(output, completed, "revision.png", "revision_overview.jpg")
    print(f"Replayed {len(completed)} cached cases to {output}", flush=True)


if __name__ == "__main__":
    main()
