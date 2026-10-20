"""Mine all valid queries and export a distinct-image shortlist without retraining."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from visualization.dino_spine_maps.model_loader import load_dino_model
from .geometry import common_queries, image_quality, mask_at, query_texture, read_source, sample_pair
from .matching import Summary, extract_pair, measurements, prepared_features, rank_key, ranking
from .rendering import encode_image, export_bundle


SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, default=_json_default, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(path, loaded):
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path), "patch_size": loaded.patch_size,
            "feature_dimension": loaded.embedding_dim, "architecture": loaded.architecture}


def _inputs(args):
    if args.start_index < 1 or (args.end_index is not None and args.end_index < args.start_index):
        raise ValueError("Use 1-based indices with 1 <= start-index <= end-index")
    if args.image:
        paths = [path.expanduser().resolve() for path in args.image]
    else:
        directory = args.image_dir.expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        if args.output_dir.expanduser().resolve().is_relative_to(directory):
            raise ValueError("Output must be outside the scanned input directory")
        paths = sorted(p.resolve() for p in (directory.rglob("*") if args.recursive else directory.iterdir())
                       if p.is_file() and p.suffix.lower() in SUFFIXES)
    paths = list(dict.fromkeys(paths))
    total = len(paths)
    paths = paths[args.start_index - 1:args.end_index]
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise ValueError(f"No images in the selected range; total discovered images: {total}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _record(image_id, path, seed, queries, i, baseline, ours, ranks, thresholds):
    record = {"image_id": image_id, "source_image": str(path), "seed": seed,
              "query_index": int(queries["indices"][i]),
              "query_a_x": float(queries["xy_a"][i, 0]), "query_a_y": float(queries["xy_a"][i, 1]),
              "source_x": float(queries["source_xy"][i, 0]), "source_y": float(queries["source_xy"][i, 1]),
              "gt_b_x": float(queries["gt_b"][i, 0]), "gt_b_y": float(queries["gt_b"][i, 1])}
    for name, values in (("baseline", baseline), ("ours", ours)):
        record.update({f"{name}_{key}": value[i].item() for key, value in values.items()})
        record.update({f"{name}_pck_{threshold:g}px": bool(values["error_px"][i] <= threshold) for threshold in thresholds})
    record.update({key: value[i].item() for key, value in ranks.items()})
    return record


def _bundle(metadata, source, a, b, baseline_heat, ours_heat, valid_b):
    return {"metadata": np.asarray(json.dumps(metadata, default=_json_default, allow_nan=False)),
            "source_png": encode_image(source), "a_png": encode_image(a.image), "b_png": encode_image(b.image),
            "input_a": a.tensor.numpy(), "input_b": b.tensor.numpy(),
            "baseline_heatmap": baseline_heat, "ours_heatmap": ours_heat,
            "baseline_valid_b": valid_b[0].cpu().numpy(), "ours_valid_b": valid_b[1].cpu().numpy()}


def _summary(global_summary, image_summaries, shortlisted, processed, rejected, model_info):
    per_image = [part["statistics"] for part in image_summaries if part["statistics"]["query_count"]]
    image_weighted = {}
    for name in ("baseline", "ours"):
        if per_image:
            image_weighted[name] = {
                "mean_error_px": float(np.mean([part["models"][name]["mean_error_px"] for part in per_image])),
                "pck": {key: float(np.mean([part["models"][name]["pck"][key] for part in per_image]))
                        for key in per_image[0]["models"][name]["pck"]}}
    return {"status": "complete", "processed_images": processed, "images_with_queries": len(per_image), "rejected_items": rejected,
            "all_valid_queries": global_summary.report(), "image_weighted": image_weighted,
            "per_image": image_summaries, "selected_case_count": len(shortlisted), "models": model_info,
            "interpretation": "Illustrations are selected by improvement; aggregate statistics include ALL valid queries, "
                              "including failures and low-texture queries. Correlated crops/queries are not independent samples."}


def mine(args):
    if args.limit < 0 or args.top_k < 1 or args.cpu_threads < 1 or args.chunk_size < 1:
        raise ValueError("Invalid limit/top-k/threads/chunk-size")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError("Seeds must be unique and nonnegative")
    if not (0 < args.min_iou < args.max_iou < 1 and 0 <= args.flip_probability <= 1 and args.border_patches >= 1):
        raise ValueError("Invalid IoU/flip/border configuration; border must be at least one patch")
    positive = (args.baseline_min_patches, args.ours_max_patches, args.min_gain_patches,
                args.peak_separation_patches, args.near_patches, args.temperature, args.peak_delta)
    if any(not math.isfinite(x) or x <= 0 for x in positive) or not 0 <= args.min_query_gradient <= 1:
        raise ValueError("Distance, temperature and peak parameters must be positive/finite")
    if not -1 <= args.min_peak_similarity <= 1:
        raise ValueError("Minimum peak cosine must be in [-1, 1]")
    paths = _inputs(args)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select an available GPU. No silent CPU fallback is performed.")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"Loading backbone checkpoints on {device}; no training", flush=True)
    models = [load_dino_model(path, device=device) for path in (args.baseline_weights, args.ours_weights)]
    patches = [model.patch_size for model in models]
    if args.view_size % math.lcm(*patches) or args.view_size < 8 * max(patches):
        raise ValueError(f"View size must be divisible by {math.lcm(*patches)} and >= {8 * max(patches)}")
    thresholds = args.pck_px or [patches[0] * value for value in (1, 2, 4)]
    if any(not math.isfinite(t) or t <= 0 for t in thresholds) or len(set(thresholds)) != len(thresholds):
        raise ValueError("PCK pixel thresholds must be positive, finite and unique")
    model_info = {name: _checkpoint(path, model) for name, path, model in
                  zip(("baseline", "ours"), (args.baseline_weights, args.ours_weights), models)}
    output.mkdir(parents=True)
    (output / "case_cache").mkdir()
    manifest = {"arguments": vars(args), "models": model_info, "images": paths,
                "input_order": "Resolved file paths sorted lexicographically for image-dir; supplied order for --image",
                "range_convention": "1-based inclusive start/end, before --limit and quality filtering",
                "torch_version": str(torch.__version__), "device": str(device), "status": "running",
                "precision": "FP32 backbone/features; TF32 disabled; no autocast",
                "query_lattice": f"Baseline A patch centres ({patches[0]} px); both models sample the same physical xy",
                "features": "final backbone x_norm_patchtokens only; sample raw query tokens then L2 normalize",
                "pck_thresholds_px": thresholds, "fixed_heatmap_limits": [args.vmin, args.vmax],
                "selection_note": "Gradient/intensity/centrality are ranking heuristics, not anatomical segmentation; "
                                  "source completeness and vertebral visibility require visual review."}
    write_json(output / "run.json", manifest)
    global_summary = Summary(thresholds)
    image_summaries, shortlist, seen_images = [], [], set()
    rejected, processed, writer, best_writer = 0, 0, None, None
    with gzip.open(output / "all_queries.csv.gz", "wt", newline="", encoding="utf-8", compresslevel=4) as query_stream, \
            (output / "rejections.jsonl").open("w", encoding="utf-8") as rejection_stream, \
            (output / "image_statistics.jsonl").open("w", encoding="utf-8") as image_stream, \
            (output / "image_best.csv").open("w", newline="", encoding="utf-8") as best_stream:
        def reject(path, seed, reason):
            nonlocal rejected
            rejected += 1
            rejection_stream.write(json.dumps({"image": str(path), "seed": seed, "reason": reason}) + "\n")
            rejection_stream.flush()

        for image_number, path in enumerate(paths, 1):
            print(f"[{image_number}/{len(paths)}] {path.name}", flush=True)
            try:
                source = read_source(path)
            except (OSError, ValueError) as exc:
                reject(path, None, f"image_read: {exc}")
                continue
            if source.width < 128 or source.height < 256:
                reject(path, None, "source_resolution_below_128x256")
                continue
            source_quality = image_quality(source)
            if source_quality["std"] < 0.04 or source_quality["gradient"] < 0.002:
                reject(path, None, "source_low_contrast_or_texture")
                continue
            decoded_hash = hashlib.sha256(str(source.size).encode() + source.tobytes()).hexdigest()
            if decoded_hash in seen_images:
                reject(path, None, "duplicate_decoded_source_pixels")
                continue
            seen_images.add(decoded_hash)
            source_hash = sha256_file(path)
            image_id = path.stem + "__" + decoded_hash[:10]
            per_image = Summary(thresholds)
            image_best, best_bundle = None, None
            for seed in args.seeds:
                effective_seed = int.from_bytes(hashlib.sha256(f"{decoded_hash}:{seed}".encode()).digest()[:8], "little")
                try:
                    a, b = sample_pair(source, seed=effective_seed, size=args.view_size, min_iou=args.min_iou,
                                       max_iou=args.max_iou, flip_probability=args.flip_probability)
                    crop_quality = [image_quality(source.crop(view.metadata["box_xyxy"])) for view in (a, b)]
                    if any(q["std"] < 0.04 or q["gradient"] < 0.002 for q in crop_quality):
                        raise ValueError("low_quality_crop")
                    queries = common_queries(a, b, patches, args.border_patches)
                    if not len(queries["indices"]):
                        raise ValueError("no_complete_common_query_footprints")
                except ValueError as exc:
                    reject(path, seed, str(exc))
                    continue
                features = [extract_pair(model, a, b, device) for model in models]
                prepared = [prepared_features(feature, queries, a, b, patch, args.border_patches)
                            for feature, patch in zip(features, patches)]
                feature_valid = torch.ones(len(queries["indices"]), dtype=torch.bool)
                for item, patch in zip(prepared, patches):
                    feature_valid &= item[2]
                    feature_valid &= mask_at(item[3].cpu().reshape(args.view_size // patch, -1), queries["gt_b"], args.view_size)
                if not feature_valid.any():
                    reject(path, seed, "no_common_nonzero_query_and_gt_features")
                    continue
                queries = {key: value[feature_valid] for key, value in queries.items()}
                values, heatmaps = [], []
                for item, patch in zip(prepared, patches):
                    measured, heat = measurements(item[0][feature_valid.to(device)], item[1], item[3], queries["gt_b"],
                                                  size=args.view_size, patch_size=patch,
                                                  near_radius=args.near_patches * patches[0],
                                                  far_radius=args.baseline_min_patches * patches[0],
                                                  peak_separation=args.peak_separation_patches * patches[0],
                                                  peak_delta=args.peak_delta, min_peak_similarity=args.min_peak_similarity,
                                                  temperature=args.temperature, chunk_size=args.chunk_size)
                    values.append(measured)
                    heatmaps.append(heat)
                ranks = ranking(*values, queries, query_texture(a, queries["xy_a"], patches[0]), source.size,
                                source_quality["score"], size=args.view_size, reference_patch=patches[0],
                                baseline_min_patches=args.baseline_min_patches, ours_max_patches=args.ours_max_patches,
                                min_gain_patches=args.min_gain_patches, min_query_gradient=args.min_query_gradient,
                                min_match_similarity=args.min_peak_similarity,
                                require_ambiguity=args.require_ambiguity)
                per_image.add(*values)
                global_summary.add(*values)
                records = [_record(image_id, path, seed, queries, i, *values, ranks, thresholds)
                           for i in range(len(queries["indices"]))]
                if writer is None:
                    writer = csv.DictWriter(query_stream, fieldnames=list(records[0]))
                    writer.writeheader()
                writer.writerows(records)
                chosen = max(range(len(records)), key=lambda i: rank_key(records[i]))
                record = records[chosen]
                if image_best is None or rank_key(record) > rank_key(image_best):
                    image_best = record
                    if record["eligible"]:
                        query = {"a_xy": queries["xy_a"][chosen].tolist(), "source_xy": queries["source_xy"][chosen].tolist(),
                                 "gt_b_xy": queries["gt_b"][chosen].tolist(),
                                 **{f"{name}_pred_b_xy": [record[f"{name}_pred_x"], record[f"{name}_pred_y"]]
                                    for name in ("baseline", "ours")},
                                 **{f"{name}_{unit}": record[f"{name}_{unit}"] for name in ("baseline", "ours")
                                    for unit in ("error_px", "error_patch")}}
                        metadata = {"image_id": image_id, "source_image": str(path), "source_sha256": source_hash,
                                    "decoded_source_sha256": decoded_hash, "source_size_wh": source.size,
                                    "orientation": "EXIF corrected before all cropping and coordinate transforms",
                                    "seed": seed, "effective_seed": effective_seed, "query_index": record["query_index"],
                                    "models": model_info, "views": {"a": a.metadata, "b": b.metadata}, "query": query,
                                    "selection": record, "source_quality": source_quality,
                                    "full_source_anatomical_completeness": "not automatically established",
                                    "query_anatomical_identity": "not annotated; texture heuristic only"}
                        best_bundle = _bundle(metadata, source, a, b, heatmaps[0][chosen], heatmaps[1][chosen],
                                              [item[3].reshape(args.view_size // patch, -1) for item, patch in zip(prepared, patches)])
                del features, prepared
            processed += 1
            image_summaries.append({"image_id": image_id, "source_image": str(path), "statistics": per_image.report()})
            image_stream.write(json.dumps(image_summaries[-1]) + "\n")
            image_stream.flush()
            if image_best is not None:
                if best_writer is None:
                    best_writer = csv.DictWriter(best_stream, fieldnames=list(image_best))
                    best_writer.writeheader()
                best_writer.writerow(image_best)
                best_stream.flush()
            if image_best is not None and image_best["eligible"]:
                contender = {**image_best, "bundle": f"case_cache/{image_id}.npz"}
                proposed = sorted(shortlist + [contender], key=rank_key, reverse=True)[:args.top_k]
                if contender in proposed:
                    np.savez_compressed(output / contender["bundle"], **best_bundle)
                shortlist = proposed
            query_stream.flush()
            write_json(output / "selected_cases.json", [{"rank": i + 1, **record} for i, record in enumerate(shortlist)])
            # Constant-size progress avoids rewriting an ever-growing dataset
            # report after each image. Full image-weighted statistics are final.
            write_json(output / "summary.json", {
                "status": "in_progress", "processed_images": processed, "rejected_items": rejected,
                "all_valid_queries": global_summary.report(), "selected_case_count": len(shortlist),
                "per_image_statistics_file": "image_statistics.jsonl"})
    # Figure export happens only after all image/seed candidates have been evaluated.
    for rank, case in enumerate(shortlist, 1):
        export_bundle(output / case["bundle"], output / "figures" / f"case_{rank:02d}_{case['image_id']}",
                      vmin=args.vmin, vmax=args.vmax, alpha=args.alpha, cmap=args.cmap)
    write_json(output / "selected_cases.json", [{"rank": i + 1, **record} for i, record in enumerate(shortlist)])
    write_json(output / "summary.json", _summary(global_summary, image_summaries, shortlist, processed, rejected, model_info))
    manifest["status"] = "complete"
    manifest["selected_count"] = len(shortlist)
    manifest["target_count"] = args.top_k
    write_json(output / "run.json", manifest)
    print(f"Finished: {global_summary.count} valid queries; {len(shortlist)}/{args.top_k} distinct qualified images. "
          "No unqualified cases are added to fill the quota.", flush=True)


def reexport(args):
    if args.case:
        path = args.case.expanduser().resolve()
    else:
        if args.run_dir is None:
            raise ValueError("Provide --case or --run-dir")
        root = args.run_dir.expanduser().resolve()
        selected = json.loads((root / "selected_cases.json").read_text())
        if args.image_id:
            selected = [case for case in selected if case["image_id"] == args.image_id]
        else:
            selected = [case for case in selected if case["rank"] == args.rank]
        if len(selected) != 1:
            raise ValueError("Requested case is not uniquely present in selected_cases.json")
        path = root / selected[0]["bundle"]
    export_bundle(path, args.output_dir.expanduser().resolve(), vmin=args.vmin, vmax=args.vmax, alpha=args.alpha, cmap=args.cmap)
    print(f"Re-exported actual cached correspondence: {path} -> {args.output_dir}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("mine", help="Run actual backbone inference and scan every valid query")
    images = search.add_mutually_exclusive_group(required=True)
    images.add_argument("--image-dir", type=Path)
    images.add_argument("--image", type=Path, nargs="+")
    search.add_argument("--recursive", action="store_true")
    search.add_argument("--start-index", type=int, default=1, help="First image in the discovered list, 1-based inclusive")
    search.add_argument("--end-index", type=int, help="Last image, 1-based inclusive; default: through the end")
    search.add_argument("--limit", type=int, default=0, help="Maximum images after range selection; 0 means no additional limit")
    search.add_argument("--baseline-weights", type=Path, default=Path("weights/initialization/rad_dino_maira2_dinov2_format.pth"))
    search.add_argument("--ours-weights", type=Path, required=True)
    search.add_argument("--device", default="cuda:0")
    search.add_argument("--cpu-threads", type=int, default=4)
    search.add_argument("--view-size", type=int, default=448)
    search.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--min-iou", type=float, default=0.35)
    search.add_argument("--max-iou", type=float, default=0.78)
    search.add_argument("--flip-probability", type=float, default=0)
    search.add_argument("--border-patches", type=int, default=1)
    search.add_argument("--baseline-min-patches", type=float, default=6)
    search.add_argument("--ours-max-patches", type=float, default=1.5)
    search.add_argument("--min-gain-patches", type=float, default=4)
    search.add_argument("--near-patches", type=float, default=2)
    search.add_argument("--peak-separation-patches", type=float, default=4)
    search.add_argument("--peak-delta", type=float, default=0.05)
    search.add_argument("--min-peak-similarity", type=float, default=0.3)
    search.add_argument("--temperature", type=float, default=0.07, help="Ranking diagnostic only; never used for displayed heatmaps")
    search.add_argument("--min-query-gradient", type=float, default=0.004)
    search.add_argument("--require-ambiguity", action="store_true", help="Require spatial-ambiguity improvement, not just prioritize it")
    search.add_argument("--pck-px", type=float, nargs="+", help="Shared physical thresholds; default 1/2/4 baseline patch widths")
    search.add_argument("--chunk-size", type=int, default=128)
    export = commands.add_parser("export", help="Re-export a specified cached case without inference")
    source = export.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", type=Path, help="Portable .npz case bundle")
    source.add_argument("--run-dir", type=Path, help="Search output root containing selected_cases.json")
    selector = export.add_mutually_exclusive_group()
    selector.add_argument("--rank", type=int, default=1)
    selector.add_argument("--image-id", help="Exact image_id in selected_cases.json")
    for command in (search, export):
        command.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are rejected")
        command.add_argument("--vmin", type=float, default=-1)
        command.add_argument("--vmax", type=float, default=1)
        command.add_argument("--alpha", type=float, default=0.45)
        command.add_argument("--cmap", default="turbo")
    return parser


def main():
    args = build_parser().parse_args()
    from matplotlib import colormaps
    colormaps[args.cmap]  # Validate before spending time on dataset inference.
    if not (-1 <= args.vmin < args.vmax <= 1 and 0 <= args.alpha <= 0.75):
        raise ValueError("Require -1 <= vmin < vmax <= 1 and 0 <= alpha <= 0.75")
    if args.command == "mine":
        with torch.inference_mode():
            mine(args)
    else:
        reexport(args)
