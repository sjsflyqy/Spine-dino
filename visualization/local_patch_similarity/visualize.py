"""Compare eight local crops using original MAIRA2 and a trained GCVD backbone."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualization.dino_spine_maps.model_loader import load_dino_model
from visualization.local_patch_similarity.analysis import aggregate, analyze, extract_features, metric_differences
from visualization.local_patch_similarity.rendering import save_alignment, save_inputs, save_matrix_comparison, save_queries
from visualization.local_patch_similarity.views import make_views, select_queries


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, nargs="+", help="One or more source radiographs")
    inputs.add_argument("--image-dir", type=Path, help="Directory of source radiographs")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Maximum source images; 0 means all")
    parser.add_argument("--before-weights", type=Path,
                        default=Path("weights/initialization/rad_dino_maira2_dinov2_format.pth"))
    parser.add_argument("--after-weights", type=Path, required=True, help="Extracted trained teacher/backbone .pth")
    parser.add_argument("--labels", nargs=2, default=["MAIRA2", "GCVD"], metavar=("BEFORE", "AFTER"))
    parser.add_argument("--config", type=Path, help="Optional training YAML; reads crops and student.patch_size only")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are rejected")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--mode", choices=("clean", "train"), default="clean",
                        help="clean: no photo augmentation/flips; train: training photo transforms and flips")
    parser.add_argument("--anchor-size", type=int, help="Override config/default 518")
    parser.add_argument("--local-size", type=int, help="Override config/default 196")
    parser.add_argument("--local-scale", type=float, nargs=2, help="Override config/default 0.2 0.5")
    parser.add_argument("--query", type=int, nargs=3, action="append", metavar=("LOCAL", "ROW", "COL"),
                        help="Repeatable query patch, all zero-based; default: centre of L0")
    parser.add_argument("--query-xy", type=float, nargs=2, action="append", metavar=("X", "Y"),
                        help="Repeatable point in EXIF-corrected source; selects first covering local")
    parser.add_argument("--negative-distance", type=float, default=0.1,
                        help="Far negatives exceed this fraction of source-image diagonal (default 0.1)")
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--delta-limit", type=float, default=0.5,
                        help="Symmetric difference color limit; raw numbers are never clipped")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--save-features", action="store_true", help="Also save raw backbone patch features to .pt")
    return parser


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_value) + "\n")


def resolve_crop_settings(args):
    config = yaml.safe_load(args.config.read_text()) if args.config else {}
    config = config or {}
    crops = config.get("crops", {})
    if config.get("student", {}).get("patch_size", 14) != 14:
        raise ValueError("Only ViT-B/14 checkpoints are supported")
    if crops.get("local_crops_number", 8) != 8:
        raise ValueError("This analysis requires exactly eight local crops")
    settings = {
        "anchor_size": args.anchor_size if args.anchor_size is not None else crops.get("global_crops_size", 518),
        "local_size": args.local_size if args.local_size is not None else crops.get("local_crops_size", 196),
        "local_scale": args.local_scale if args.local_scale is not None else crops.get("local_crops_scale", [0.2, 0.5]),
        "crop_ratio": crops.get("crop_ratio", [0.75, 4 / 3]),
        "flip_probability": crops.get("horizontal_flip_probability", 0.5),
    }
    if any(size < 28 or size % 14 for size in (settings["anchor_size"], settings["local_size"])):
        raise ValueError("View sizes must be multiples of 14 and at least 28")
    low, high = settings["local_scale"]
    if not (0 < low <= high <= 1):
        raise ValueError("Local scale must satisfy 0 < minimum <= maximum <= 1")
    low, high = settings["crop_ratio"]
    if not (0 < low <= high and np.isfinite(high)):
        raise ValueError("Crop ratio must be positive, finite and ordered")
    if not 0 <= settings["flip_probability"] <= 1:
        raise ValueError("Horizontal flip probability must lie in [0, 1]")
    return settings


def _inputs(args):
    if args.image:
        paths = [path.expanduser().resolve() for path in args.image]
    else:
        directory = args.image_dir.expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        if args.output_dir.resolve().is_relative_to(directory):
            raise ValueError("--output-dir must be outside --image-dir")
        suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
        paths = sorted(p.resolve() for p in (directory.rglob("*") if args.recursive else directory.iterdir())
                       if p.is_file() and p.suffix.lower() in suffixes)
    paths = list(dict.fromkeys(paths))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise ValueError("No input images found")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _checkpoint_record(path):
    path = path.expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def main():
    args = build_parser().parse_args()
    if args.cpu_threads < 1 or args.local_batch_size < 1 or args.limit < 0:
        raise ValueError("Threads/batch size must be positive and limit nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed > 2**32 - 1 for seed in args.seeds):
        raise ValueError("Seeds must be unique integers in [0, 2**32 - 1]")
    if not (0 < args.negative_distance < 1 and -1 <= args.vmin < args.vmax <= 1
            and 0 < args.delta_limit <= 2):
        raise ValueError("Require 0 < negative-distance < 1, -1 <= vmin < vmax <= 1, 0 < delta-limit <= 2")
    settings = resolve_crop_settings(args)
    paths = _inputs(args)
    before_record = _checkpoint_record(args.before_weights)
    after_record = _checkpoint_record(args.after_weights)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    _write_json(output / "run.json", {
        "arguments": vars(args), "resolved_crop_settings": settings, "device": str(device),
        "before_checkpoint": before_record, "after_checkpoint": after_record,
        "torch_version": str(torch.__version__), "images": paths,
        "feature_source": "final backbone x_norm_patchtokens; L2 normalized for cosine; no projection head",
        "aggregation": "equal weight per image/seed; directed patch pairs are not independent samples",
        "scope": "all real-image patches, including background; no automatic spine segmentation",
        "mode_note": "geometry sampled once, photo transforms replayed separately; train mode is not exact dataloader RNG replay",
    })
    print(f"Loading two ViT-B/14 backbones on {device}", flush=True)
    before_model = load_dino_model(args.before_weights, architecture="dinov2", device=device)
    after_model = load_dino_model(args.after_weights, architecture="dinov2", device=device)
    records = []
    render_options = {"labels": args.labels, "vmin": args.vmin, "vmax": args.vmax}
    for path in paths:
        with Image.open(path) as image:
            source = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256(str(path).encode()).hexdigest()[:8]
        for seed in args.seeds:
            print(f"Analyzing {path.name}, seed={seed}", flush=True)
            views = make_views(source, seed=seed, mode=args.mode, **settings)
            queries = select_queries(views, args.query, args.query_xy)
            destination = output / f"{path.stem}__{digest}" / f"seed_{seed}"
            destination.mkdir(parents=True)
            _write_json(destination / "geometry.json", {
                "image": path, "source_size_wh": source.size, "seed": seed, "mode": args.mode,
                "anchor": views.anchor_geometry, "locals": views.local_geometry,
                "patch_size": views.patch_size, "queries": queries,
                "negative_distance_source_diagonal_fraction": args.negative_distance,
                "coordinates": "continuous pixel-edge xy, EXIF-corrected source; row-major patch order",
            })
            torch.save({"anchor": views.anchor, "locals": views.locals}, destination / "inputs.pt")
            save_inputs(destination, views, queries)
            results, metrics = [], []
            for name, model in (("before", before_model), ("after", after_model)):
                local, anchor = extract_features(model, views, device, args.local_batch_size)
                metric, arrays = analyze(local, anchor, views, negative_distance=args.negative_distance)
                np.savez_compressed(destination / f"{name}_arrays.npz", **arrays,
                                    local_source_centres_xy=views.source_centres.numpy())
                if args.save_features:
                    torch.save({"local_patch_tokens": local, "anchor_patch_tokens": anchor}, destination / f"{name}_features.pt")
                results.append(arrays)
                metrics.append(metric)
            save_matrix_comparison(destination, *results, delta_limit=args.delta_limit, **render_options)
            np.savez_compressed(destination / "local_pooled_similarity.npz",
                                before=results[0]["local_pooled_similarity"], after=results[1]["local_pooled_similarity"],
                                difference=results[1]["local_pooled_similarity"] - results[0]["local_pooled_similarity"])
            save_queries(destination, views, queries, *results, **render_options)
            save_alignment(destination, views, *results, **render_options)
            record = {"image": str(path), "seed": seed, "sample_dir": str(destination.relative_to(output)),
                      "before": metrics[0], "after": metrics[1],
                      "difference_after_minus_before": metric_differences(*metrics)}
            _write_json(destination / "metrics.json", record)
            records.append(record)
            _write_json(output / "summary.json", {"completed_image_seed_pairs": len(records),
                                                   "aggregate": aggregate(records), "samples": records})
            # Persist completed samples even if a later input fails.
            rows = [{"image": r["image"], "seed": r["seed"], "sample_dir": r["sample_dir"],
                     **{f"{section}.{key}": value for section in ("before", "after", "difference_after_minus_before")
                        for key, value in r[section].items()}} for r in records]
            with (output / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            gap = record["difference_after_minus_before"]["paired_discrimination_gap"]
            print(f"Saved {destination}; discrimination-gap change={gap}", flush=True)
    print(f"Finished {len(records)} image/seed pairs. Summary: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
