"""Add whole-local cosine figures to an existing run using its exact saved inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from visualization.dino_spine_maps.model_loader import load_dino_model
from .analysis import extract_local_features, pooled_local_similarity
from .rendering import save_matrix_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing analysis root containing run.json")
    parser.add_argument("--device", default="cuda:0", help="Inference device; defaults to cuda:0")
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.local_batch_size < 1 or args.cpu_threads < 1:
        parser.error("Batch size and thread count must be positive")
    torch.set_num_threads(args.cpu_threads)
    root = args.run_dir.expanduser().resolve()
    manifest = json.loads((root / "run.json").read_text())
    directories = sorted(path.parent for path in root.glob("*/seed_*/geometry.json"))
    if not directories:
        raise ValueError(f"No completed crop geometry found in {root}")
    device = torch.device(args.device)
    options = manifest["arguments"]
    models = {}
    records = []
    for directory in directories:
        results = []
        devices_used = {}
        for name in ("before", "after"):
            checkpoint = manifest[f"{name}_checkpoint"]
            weights = Path(checkpoint["path"])
            stat = weights.stat()
            if stat.st_size != checkpoint["bytes"] or stat.st_mtime_ns != checkpoint["mtime_ns"]:
                raise ValueError(f"Checkpoint changed since the original run: {weights}. Start a new analysis run.")
            cached = directory / f"{name}_features.pt"
            if cached.is_file():
                tokens = torch.load(cached, map_location="cpu", weights_only=True)["local_patch_tokens"]
                devices_used[name] = "saved_features"
            else:
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable in this process; run on a GPU host with --device cuda:0")
                if name not in models:
                    print(f"Loading {name} backbone on {device}: {weights}", flush=True)
                    models[name] = load_dino_model(weights, architecture="dinov2", device=device)
                inputs = torch.load(directory / "inputs.pt", map_location="cpu", weights_only=True)
                tokens = extract_local_features(models[name], inputs["locals"], device, args.local_batch_size)
                devices_used[name] = str(device)
            results.append({"local_pooled_similarity": pooled_local_similarity(tokens).numpy()})
        save_matrix_comparison(directory, *results, pooled_only=True,
                               labels=options.get("labels", ["MAIRA2", "GCVD"]),
                               vmin=options.get("vmin", -1), vmax=options.get("vmax", 1),
                               delta_limit=options.get("delta_limit", 0.5))
        before, after = [result["local_pooled_similarity"] for result in results]
        np.savez_compressed(directory / "local_pooled_similarity.npz", before=before, after=after,
                            difference=after - before)
        off_diagonal = ~np.eye(8, dtype=bool)
        record = {
            "sample_dir": str(directory.relative_to(root)),
            "definition": "cosine of mean raw x_norm_patchtokens per local, normalized after pooling",
            "feature_source": devices_used,
            "before_pooled_cross_local_mean_cosine": float(before[off_diagonal].mean()),
            "after_pooled_cross_local_mean_cosine": float(after[off_diagonal].mean()),
            "difference_after_minus_before": float((after - before)[off_diagonal].mean()),
        }
        (directory / "local_pooled_metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        records.append(record)
        print(f"Saved {directory / 'local_pooled_similarity_8x8.png'}", flush=True)
    (root / "pooled_summary.json").write_text(json.dumps({"samples": records}, indent=2, allow_nan=False) + "\n")
    print(f"Added whole-local similarity for {len(records)} image/seed pairs", flush=True)


if __name__ == "__main__":
    main()
