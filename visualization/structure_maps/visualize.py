"""Visualize local teacher-token differences in the exact GeoTopo anchor geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.geotopo_dino.masking.feature_structure_score import compute_structure_score
from visualization.dino_spine_maps.model_loader import load_dino_model
from visualization.dino_spine_maps.visualize import _device, _layers, _safe_name
from visualization.spine_masks.visualize import _inputs, _metadata_value, make_anchor


@torch.inference_mode()
def extract_patch_features(loaded, image, selected_layers):
    """Use the native intermediate-token API, without computing CLS attention."""
    if loaded.generation != 2 or loaded.patch_size != 14:
        raise ValueError("This anchor diagnostic supports native DINOv2 ViT-B/14 checkpoints")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    indices = sorted(set(selected_layers))
    if image.ndim != 4 or image.shape[0] != 1 or not indices:
        raise ValueError("expected one BCHW image and nonempty layers")
    if min(indices) < 0 or max(indices) >= loaded.num_layers:
        raise ValueError("selected layers exceed model depth")
    outputs = loaded.model.get_intermediate_layers(image, n=indices, norm=True)
    expected = (image.shape[-2] // 14) * (image.shape[-1] // 14)
    if len(outputs) != len(indices) or any(tokens.shape[1] != expected for tokens in outputs):
        raise RuntimeError("native patch features do not match the anchor grid")
    return {index + 1: tokens[0].detach().float().cpu() for index, tokens in zip(indices, outputs)}


@torch.no_grad()
def pixel_gradient(tensor, valid, patch_size=14):
    """Mean absolute grayscale finite differences; exclude padding transitions."""
    gray = tensor.float().mean(0)
    pixel_valid = valid.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
    if gray.shape != pixel_valid.shape:
        raise ValueError("input and validity geometry differ")
    total, count = torch.zeros_like(gray), torch.zeros_like(gray)
    for dy, dx in ((0, 1), (1, 0)):
        center = (slice(0, gray.shape[0] - dy), slice(0, gray.shape[1] - dx))
        neighbor = (slice(dy, None), slice(dx, None))
        pair = pixel_valid[center] & pixel_valid[neighbor]
        total[center] += (gray[center] - gray[neighbor]).abs() * pair
        count[center] += pair
    total = F.avg_pool2d(total[None, None], patch_size, patch_size)[0, 0]
    count = F.avg_pool2d(count[None, None], patch_size, patch_size)[0, 0]
    raw = total / count.clamp_min(1e-8) * valid
    values = raw[valid]
    normalized = torch.zeros_like(raw)
    if values.numel():
        low, high = torch.quantile(values, torch.tensor([0.05, 0.95]))
        if high - low > 1e-8:
            normalized = ((raw - low) / (high - low)).clamp(0, 1) * valid
    return raw, normalized


def _correlation(a, b, valid):
    x, y = a[valid].double(), b[valid].double()
    if x.numel() < 2:
        return None
    x, y = x - x.mean(), y - y.mean()
    denominator = x.norm() * y.norm()
    return float((x @ y) / denominator) if denominator > 1e-12 else None


def _aligned_stability(clean, train, layers):
    """Undo the recorded horizontal flip before comparing identical geometry."""
    flipped = bool(train["geometry"]["flipped"])
    train_valid = train["valid"].flip(-1) if flipped else train["valid"]
    valid = clean["valid"] & train_valid
    result = {}
    for layer in layers:
        train_map = train["scores"][layer].difference
        if flipped:
            train_map = train_map.flip(-1)
        result[str(layer)] = {
            "raw_difference_pearson": _correlation(clean["scores"][layer].difference, train_map, valid),
            "compared_patch_count": int(valid.sum()),
            "flip_undone": flipped,
        }
    return result


def _render_case(path, views, layers, alpha):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Share raw D scale across this image's selected layers and clean/train views.
    raw_values = np.concatenate([
        view["scores"][layer].difference[view["valid"]].numpy()
        for view in views.values() for layer in layers
        if view["valid"].any()
    ]) if any(view["valid"].any() for view in views.values()) else np.array([0.0])
    raw_max = max(float(np.percentile(raw_values, 98)), 1e-5)
    figure, axes = plt.subplots(len(views) * len(layers), 5,
                               figsize=(17, 4.2 * len(views) * len(layers)), squeeze=False)
    row = 0
    for mode, view in views.items():
        for layer in layers:
            score = view["scores"][layer]
            valid = view["valid"].numpy()
            image = np.asarray(view["image"])
            maps = (score.difference.numpy(), score.score.numpy(), score.score.numpy(), view["gradient_norm"].numpy())
            axes[row, 0].imshow(image)
            axes[row, 0].set_title(f"{mode} | layer {layer}\ninput; valid={int(valid.sum())}")
            titles = (f"Raw D; mean={score.metrics['raw_mean']:.4f}",
                      "Smoothed normalized S [0,1]", "S overlay", "Pixel-gradient overlay [0,1]")
            for col, (grid, title) in enumerate(zip(maps, titles), start=1):
                ax = axes[row, col]
                if col < 3:
                    artist = ax.imshow(np.ma.array(grid, mask=~valid), cmap="magma", vmin=0,
                                       vmax=raw_max if col == 1 else 1, interpolation="nearest")
                    figure.colorbar(artist, ax=ax, fraction=0.045, pad=0.02)
                else:
                    ax.imshow(image)
                    expanded = np.repeat(np.repeat(grid, 14, 0), 14, 1)
                    visibility = np.repeat(np.repeat(valid, 14, 0), 14, 1)
                    ax.imshow(np.ma.array(expanded, mask=~visibility), cmap="magma", vmin=0,
                              vmax=1, alpha=alpha, interpolation="nearest")
                ax.set_title(title + ("\nFLAT: normalization suppressed" if score.metrics["near_constant"] else ""))
            for ax in axes[row]:
                ax.axis("off")
            row += 1
    figure.suptitle(f"{path.parent.name} | {path.name}\nLocal feature variation is not spine probability; raw scale shared within this figure.")
    figure.tight_layout()
    figure.savefig(path / "structure.png", dpi=140, bbox_inches="tight")
    plt.close(figure)
    return {"raw_vmin": 0.0, "raw_vmax_p98": raw_max,
            "raw_scale_scope": "this image/checkpoint, all selected layers and modes",
            "normalized_vmin": 0.0, "normalized_vmax": 1.0}


def _overview(root, records):
    # Paginate to keep a large diagnosis set reviewable and memory bounded.
    for start in range(0, len(records), 6):
        thumbnails = []
        for record in records[start:start + 6]:
            with Image.open(root / record["output"] / "structure.png") as source:
                panel = source.convert("RGB")
                panel.thumbnail((1400, 850))
                thumbnails.append(panel.copy())
        width = max(panel.width for panel in thumbnails)
        height = sum(panel.height for panel in thumbnails)
        sheet = Image.new("RGB", (width, height), "white")
        top = 0
        for panel in thumbnails:
            sheet.paste(panel, (0, top))
            top += panel.height
        sheet.save(root / f"overview_{start // 6 + 1:03d}.jpg", quality=90)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--demo", action="store_true", help="Six existing source X-rays from different datasets")
    inputs.add_argument("--image", type=Path, action="append", help="Repeat to use multiple original X-rays")
    inputs.add_argument("--image-dir", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Maximum input count; 0 means all")
    parser.add_argument("--sample-count", type=int, default=0, help="Seeded random subset before --limit; 0 means all")
    parser.add_argument("--weights", type=Path, action="append", help="Repeat to compare checkpoints; default is original MAIRA-2")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing output is never overwritten")
    parser.add_argument("--layers", default="12", help="One-based: 12, 8,12 or all")
    parser.add_argument("--anchor-size", type=int, default=518)
    parser.add_argument("--anchor-mode", choices=("clean", "train", "both"), default="both")
    parser.add_argument("--radius", type=int, default=1, help="Local neighborhood radius; 1 means eight neighbors")
    parser.add_argument("--smooth-kernel", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.limit < 0 or args.sample_count < 0 or args.cpu_threads < 1 or not 0 <= args.alpha <= 1:
        raise ValueError("invalid limit, sample-count, cpu-threads or alpha")
    if args.anchor_size < 14 or args.anchor_size % 14 or args.radius < 1 or args.smooth_kernel < 1 or args.smooth_kernel % 2 != 1:
        raise ValueError("anchor-size must be divisible by 14; radius positive; smooth-kernel positive and odd")
    root = args.output_dir.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {root}")
    # Reuse existing discovery, including output/input separation and collision-safe keys.
    limit = args.limit
    args.limit = 0
    entries = _inputs(args)
    args.limit = limit
    if args.sample_count and args.sample_count < len(entries):
        entries = sorted(random.Random(args.seed).sample(entries, args.sample_count))
    if limit:
        entries = entries[:limit]
    weights = [p.expanduser().resolve() for p in (args.weights or [REPO_ROOT / "weights/initialization/rad_dino_maira2_dinov2_format.pth"])]
    if len(set(weights)) != len(weights):
        raise ValueError("duplicate weights")
    for path in weights:
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(args.cpu_threads)
    device = _device(args.device)
    root.mkdir(parents=True)
    manifest = {"arguments": vars(args), "weights": weights, "inputs": [str(p) for p, _ in entries],
                "training_integration": False, "formula": "D=mean_valid_neighbors(1-cos(L2(tokens))); S=valid_smooth(clamp((D-q05)/(q95-q05),0,1))",
                "normalization_quantiles": [0.05, 0.95], "flat_spread_threshold": 1e-5,
                "note": "Heuristic structure variation, not anatomy probability. Train is one seeded augmentation realization."}
    (root / "run_config.json").write_text(json.dumps(manifest, indent=2, default=_metadata_value) + "\n")
    completed, failures, rows = [], [], []

    def save_summary():
        (root / "summary.json").write_text(json.dumps({"completed": completed, "failures": failures,
            "success_count": len(completed), "failure_count": len(failures)}, indent=2) + "\n")
        if rows:
            with (root / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    for checkpoint in weights:
        label = _safe_name(checkpoint.parent.name + "__" + checkpoint.stem) + "__" + hashlib.sha256(str(checkpoint).encode()).hexdigest()[:8]
        loaded = load_dino_model(checkpoint, architecture="dinov2", device=device)
        layers = [index + 1 for index in _layers(args.layers, loaded.num_layers)]
        print(f"Loaded {checkpoint.name}; device={device}; layers={layers}; images={len(entries)}", flush=True)
        for image_path, key in entries:
            destination = root / key / label
            print(f"  {image_path.name}", flush=True)
            try:
                with Image.open(image_path) as opened:
                    original = ImageOps.exif_transpose(opened).convert("RGB")
                views = {}
                modes = ("clean", "train") if args.anchor_mode == "both" else (args.anchor_mode,)
                seed = (args.seed + int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16)) % (2**32)
                for mode in modes:
                    tensor, geometry, anchor = make_anchor(original, size=args.anchor_size, mode=mode, seed=seed)
                    features = extract_patch_features(loaded, tensor.to(device), [layer - 1 for layer in layers])
                    valid = geometry["valid_mask"]
                    scores = {layer: compute_structure_score(features[layer], valid, radius=args.radius,
                        smooth_kernel=args.smooth_kernel) for layer in layers}
                    gradient, gradient_norm = pixel_gradient(tensor, valid)
                    views[mode] = {"geometry": geometry, "image": anchor, "valid": valid,
                                   "scores": scores, "gradient": gradient, "gradient_norm": gradient_norm}
                destination.mkdir(parents=True)
                original.save(destination / "original.png")
                arrays = {}
                case_rows = []
                metrics = {}
                for mode, view in views.items():
                    view["image"].save(destination / f"input_{mode}.png")
                    arrays[f"{mode}_valid"] = view["valid"].numpy()
                    arrays[f"{mode}_pixel_gradient"] = view["gradient"].numpy()
                    arrays[f"{mode}_pixel_gradient_normalized"] = view["gradient_norm"].numpy()
                    for layer, score in view["scores"].items():
                        prefix = f"{mode}_layer_{layer:02d}"
                        for name, value in (("difference", score.difference), ("normalized", score.normalized),
                                            ("score", score.score), ("neighbor_count", score.neighbor_count)):
                            arrays[f"{prefix}_{name}"] = value.numpy()
                        stats = {**score.metrics, "gradient_pearson": _correlation(score.difference, view["gradient"], view["valid"])}
                        metrics[prefix] = stats
                        case_rows.append({"image": str(image_path), "weights": str(checkpoint), "mode": mode,
                                     "layer": layer, **stats, "output": str(destination.relative_to(root))})
                scale = _render_case(destination, views, layers, args.alpha)
                stability = _aligned_stability(views["clean"], views["train"], layers) if len(views) == 2 else None
                metadata = {"image": image_path, "weights": checkpoint, "architecture": loaded.architecture,
                            "layers": layers, "seed": seed, "radius": args.radius, "smooth_kernel": args.smooth_kernel,
                            "geometry": {mode: view["geometry"] for mode, view in views.items()},
                            "metrics": metrics, "clean_train_stability": stability, "display_scale": scale}
                np.savez_compressed(destination / "raw_maps.npz", **arrays)
                (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_metadata_value) + "\n")
                rows.extend(case_rows)
                completed.append({"image": str(image_path), "weights": str(checkpoint), "output": str(destination.relative_to(root))})
            except Exception as exc:
                failures.append({"image": str(image_path), "weights": str(checkpoint), "error": f"{type(exc).__name__}: {exc}"})
                print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
                save_summary()
                if args.fail_fast:
                    raise
            save_summary()
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if completed:
        _overview(root, completed)
    print(f"Saved {len(completed)} image/checkpoint cases; failed {len(failures)}; {root}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
