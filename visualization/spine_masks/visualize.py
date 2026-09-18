"""Generate training-geometry spine masks without changing or running training."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageOps
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.geotopo_dino.masking.spine_span import (
    SpanConfig, fuse_attention, generate_mask_comparison,
)
from visualization.dino_spine_maps.map_extractor import extract_layer_maps
from visualization.dino_spine_maps.model_loader import load_dino_model


DEMO_NAMES = (
    "buu2000_ap_00000002.jpg", "csxa_00000629.png", "nanning_00000706.png",
    "nhanes2_00009665.png", "ningbo_00000356.jpg", "vindr_train_00006336.png",
)
SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@contextmanager
def seeded_cpu_random(seed):
    """Reuse training augmentations without leaking changes to caller RNG state."""
    state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        random.setstate(state)


def make_anchor(source: Image.Image, *, size: int, mode: str, seed: int):
    # Import only in this offline process. Existing augmentation code stays intact.
    upstream = str(REPO_ROOT / "upstream" / "dinov2-main")
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from methods.geotopo_dino.data.augmentations import DataAugmentationGeoTopoDINO

    if mode not in {"clean", "train"}:
        raise ValueError("mode must be clean or train")
    augmentation = DataAugmentationGeoTopoDINO(
        (0.5, 1.0), (0.2, 0.5), 8, global_crops_size=size,
        patch_size=14, horizontal_flip_probability=0.5 if mode == "train" else 0.0,
    )
    if mode == "clean":
        augmentation.anchor_photometric = torch.nn.Identity()
    with seeded_cpu_random(seed):
        tensor, geometry = augmentation._full_fov_anchor(source)
    mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    pixels = ((tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return tensor, geometry, Image.fromarray(pixels)


def _metadata_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def _inputs(args):
    if args.demo:
        base = REPO_ROOT / "SpinePretrain-v1/spine_dino_dataset/train/spine"
        paths = [base / name for name in DEMO_NAMES]
        entries = [(path, Path(path.name)) for path in paths]
    elif args.image:
        paths = [path.expanduser().resolve() for path in args.image]
        entries = [(path, Path(path.name + "__" + hashlib.sha256(str(path).encode()).hexdigest()[:8]))
                   for path in paths]
    else:
        base = args.image_dir.expanduser().resolve()
        if not base.is_dir():
            raise NotADirectoryError(base)
        if args.output_dir.resolve().is_relative_to(base):
            raise ValueError("Output directory must be outside --image-dir to prevent reading generated images")
        paths = sorted(path for path in (base.rglob("*") if args.recursive else base.iterdir())
                       if path.is_file() and path.suffix.lower() in SUFFIXES)
        entries = [(path, path.relative_to(base)) for path in paths]
    if args.limit:
        entries = entries[:args.limit]
    if not entries:
        raise ValueError("No input images found")
    for path, _ in entries:
        if not path.is_file():
            raise FileNotFoundError(path)
    return entries


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, action="append", help="Repeat for multiple source images")
    inputs.add_argument("--image-dir", type=Path)
    inputs.add_argument("--demo", action="store_true", help="The six source images reviewed in outputs/visualization")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 processes all selected inputs")
    parser.add_argument("--weights", type=Path, required=True, help="Extracted DINOv2 ViT-B/14 backbone")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing runs are never overwritten")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--anchor-size", type=int, default=518)
    parser.add_argument("--anchor-mode", choices=("clean", "train", "both"), default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=8, help="One-based attention layer")
    parser.add_argument("--heads", default="4,5,8", help="One-based heads, comma-separated, or all; a hypothesis, not calibrated")
    parser.add_argument("--response-source", choices=("selected_attention", "mean_attention", "cosine"),
                        default="selected_attention", help="cosine uses the final layer")
    parser.add_argument("--mask-ratio", type=float, default=0.4, help="Fraction of valid image patches, not the padded square")
    parser.add_argument("--span-fraction", type=float, default=0.5)
    parser.add_argument("--band-mask-ratio", type=float, default=0.5)
    parser.add_argument("--context-rows", type=int, default=2)
    parser.add_argument("--band-half-width-fraction", type=float, default=0.12)
    parser.add_argument("--min-coverage", type=float, default=0.25)
    parser.add_argument("--min-contrast", type=float, default=1.4)
    parser.add_argument("--max-border-mass", type=float, default=0.55)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _render(output, anchor, maps, result, valid, *, layer, heads, source, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_np = valid.numpy()
    size = anchor.width
    grid_size = valid.shape[0]

    def image_axis(ax, label):
        ax.imshow(anchor)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    def response_axis(ax, array, label, vmin=None, vmax=None, cmap="magma"):
        array = np.asarray(array).copy()
        array[~valid_np] = np.nan
        image_axis(ax, label)
        if vmax is None:
            finite = array[np.isfinite(array)]
            vmax = max(float(np.percentile(finite, 98)), 1e-8) if finite.size else 1.0
        heat = ax.imshow(array, extent=(-0.5, size - 0.5, size - 0.5, -0.5),
                         interpolation="nearest", alpha=0.6, cmap=cmap, vmin=vmin, vmax=vmax)
        return heat

    def mask_axis(ax, mask, label, color=(0.08, 0.45, 1.0, 0.85)):
        image_axis(ax, label)
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[mask.numpy()] = color
        ax.imshow(rgba, extent=(-0.5, size - 0.5, size - 0.5, -0.5), interpolation="nearest")

    figure, axes = plt.subplots(2, 5, figsize=(18, 8), constrained_layout=True)
    image_axis(axes[0, 0], f"{anchor.width} x {anchor.height} teacher anchor")
    limit = max(float(np.percentile(np.concatenate([
        maps["selected_attention"][valid_np], maps["mean_attention"][valid_np]]), 98)), 1e-8)
    heat = response_axis(axes[0, 1], maps["selected_attention"], f"L{layer} heads {heads}", 0, limit)
    response_axis(axes[0, 2], maps["mean_attention"], f"L{layer} all heads (same scale)", 0, limit)
    figure.colorbar(heat, ax=[axes[0, 1], axes[0, 2]], shrink=0.55)
    heat = response_axis(axes[0, 3], maps["cosine"], "Final CLS-patch cosine", -1, 1, "coolwarm")
    figure.colorbar(heat, ax=axes[0, 3], shrink=0.55)
    mask_axis(axes[0, 4], result.band, f"Candidate band ({source})", (0.1, 0.8, 0.3, 0.45))
    rows = (result.centerline >= 0).nonzero().flatten()
    if rows.numel():
        scale = size / grid_size
        line = result.centerline.numpy().astype(float)
        line[line < 0] = np.nan  # Do not draw a connection across unsupported segments.
        axes[0, 4].plot((line + 0.5) * scale - 0.5,
                        (np.arange(len(line)) + 0.5) * scale - 0.5, color="lime", linewidth=1.4)
    labels = {"block": "Baseline block", "vertical_span": "Vertical interval + fill",
              "band_random": "Band random + same outside fill", "topology_span": "Topology span + quota fill"}
    for axis, name in zip(axes[1, :4], labels):
        info = result.diagnostics["masks"][name]
        fallback = (name in {"band_random", "topology_span"} and result.diagnostics["used_fallback"])
        fallback |= name == "vertical_span" and bool(result.diagnostics["vertical_fallback_reason"])
        label = labels[name] + (" [BLOCK FALLBACK]" if fallback else "")
        mask_axis(axis, result.masks[name], f"{label}\n{info['masked_count']} patches; {info['valid_mask_ratio']:.1%} valid")
    mask_axis(axes[1, 4], result.span, f"Span only (before fill)\n{result.diagnostics['span_count']} patches")
    diagnostics = result.diagnostics
    status = ("FALLBACK: " + ", ".join(diagnostics["fallback_reasons"]) if diagnostics["used_fallback"]
              else "PROPOSAL ACCEPTED (heuristic; not anatomical ground truth)")
    figure.suptitle(f"{title}\n{status}\ncoverage={diagnostics['coverage']:.2f}, "
                   f"contrast={diagnostics['contrast']:.2f}, border mass={diagnostics['border_mass']:.2f}; "
                   "green = candidate region; blue = token locations to mask", fontsize=11)
    figure.savefig(output / "comparison.png", dpi=130)
    plt.close(figure)

    head_maps = maps["attention_heads"]
    figure, axes = plt.subplots(3, 4, figsize=(12, 10), constrained_layout=True)
    head_limit = max(float(np.percentile(head_maps[:, valid_np], 98)), 1e-8)
    for index, axis in enumerate(axes.flat):
        response_axis(axis, head_maps[index], f"Head {index + 1}" + (" [selected]" if index + 1 in heads else ""),
                      0, head_limit)
    figure.suptitle(f"{title} | layer {layer}, raw attention; shared scale", fontsize=11)
    figure.savefig(output / "heads.png", dpi=110)
    plt.close(figure)


def _render_steps(output, anchor, result, title, old_arrays=None):
    """A reader-facing decomposition; colors have a single meaning throughout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.exists() else FontProperties()
    colors = {"band": (0.1, 0.8, 0.3, 0.4), "span": (0.08, 0.45, 1.0, 0.88),
              "supplement": (0.7, 0.2, 0.85, 0.8), "context": (1.0, 0.65, 0.05, 0.7)}
    extent = (-0.5, anchor.width - 0.5, anchor.height - 0.5, -0.5)

    def panel(ax, label, layers=()):
        ax.imshow(anchor)
        for mask, color in layers:
            values = np.asarray(mask, dtype=bool)
            rgba = np.zeros((*values.shape, 4), dtype=np.float32)
            rgba[values] = colors[color]
            ax.imshow(rgba, extent=extent, interpolation="nearest")
        ax.set_title(label, fontproperties=font, fontsize=12)
        ax.axis("off")

    d = result.diagnostics
    figure, axes = plt.subplots(2, 3, figsize=(12, 10), constrained_layout=True)
    panel(axes[0, 0], "① 原始输入视图")
    panel(axes[0, 1], f"② 候选带：绿色仅表示候选区域\n{d['band_count']} 个 patch；不是解剖分割",
          [(result.band, "band")])
    panel(axes[0, 2], f"③ 连续遮挡段（蓝）+ 保留上下文（橙）\n连续段 {d['span_count']} 个 patch",
          [(result.span, "span"), (result.context, "context")])
    panel(axes[1, 0], f"④ 补充遮挡（紫）\n{d['supplement_count']} 个 patch，可在远处带内或带外",
          [(result.supplement, "supplement"), (result.context, "context")])
    final_layers = ([(result.masks['topology_span'], "span")] if d['used_fallback'] else
                    [(result.span, "span"), (result.supplement, "supplement"), (result.context, "context")])
    panel(axes[1, 1], f"⑤ 最终遮挡：蓝 + 紫；橙色仍可见\n带内遮挡 {d['band_mask_ratio_actual']:.1%}；总计 {d['target_count']} patch",
          final_layers)
    panel(axes[1, 2], "⑥ 原训练 block 对照（同总预算）", [(result.masks['block'], "span")])
    status = "回退到原始 block：" + ", ".join(d['fallback_reasons']) if d['used_fallback'] else "离线候选方案；尚未接入训练"
    figure.suptitle(f"{title}\n{status}", fontproperties=font, fontsize=14)
    figure.savefig(output / "mask_steps.png", dpi=140)
    plt.close(figure)
    if old_arrays is not None:
        figure, axes = plt.subplots(1, 4, figsize=(15, 6), constrained_layout=True)
        old_band = old_arrays['band'].astype(bool)
        old_ratio = np.count_nonzero(old_arrays['mask_topology_span'] & old_band) / max(old_band.sum(), 1)
        panel(axes[0], "旧版候选带", [(old_band, "band")])
        panel(axes[1], f"旧版最终掩码：带内 {old_ratio:.1%}",
              [(old_arrays['span'], "span"), (old_arrays['background'], "supplement")])
        panel(axes[2], "新版候选带", [(result.band, "band")])
        panel(axes[3], f"新版最终掩码：带内 {d['band_mask_ratio_actual']:.1%}", final_layers)
        figure.suptitle(f"{title}\n相同输入、注意力响应和总掩码数；蓝=连续段，紫=补充，橙=保留上下文",
                       fontproperties=font, fontsize=13)
        figure.savefig(output / "revision.png", dpi=140)
        plt.close(figure)


def _save_case(destination, anchor, maps, response, valid, result, details, *, layer, heads, source, title,
               old_arrays=None):
    destination.mkdir(parents=True)
    anchor.save(destination / "anchor.png")
    Image.fromarray(valid.numpy().astype(np.uint8) * 255).save(destination / "valid_grid.png")
    arrays = {**maps, "valid_mask": valid.numpy(), "response": response.numpy(),
              "smoothed_response": result.response.numpy(), "centerline": result.centerline.numpy()}
    for name in ("band", "span", "background", "supplement", "context"):
        arrays[name] = getattr(result, name).numpy()
    for name, mask in result.masks.items():
        arrays[f"mask_{name}"] = mask.numpy()
        Image.fromarray(mask.numpy().astype(np.uint8) * 255).save(destination / f"mask_{name}.png")
    np.savez_compressed(destination / "masks_and_maps.npz", **arrays)
    details = {**details, "diagnostics": result.diagnostics}
    (destination / "metadata.json").write_text(json.dumps(details, indent=2, default=_metadata_value) + "\n")
    _render(destination, anchor, maps, result, valid, layer=layer, heads=heads, source=source, title=title)
    _render_steps(destination, anchor, result, title, old_arrays)


def _write_overview(root, completed, filename="comparison.png", output_name="overview.jpg"):
    from PIL import ImageDraw
    if not completed:
        return
    columns = 2
    thumb_width, thumb_height = 1000, 480
    sheet = Image.new("RGB", (columns * thumb_width, ((len(completed) + columns - 1) // columns) * thumb_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, entry in enumerate(completed):
        with Image.open(root / entry["output"] / filename) as opened:
            thumb = ImageOps.contain(opened.convert("RGB"), (thumb_width, thumb_height - 20))
        x, y = index % columns * thumb_width, index // columns * thumb_height
        draw.text((x + 5, y + 3), f"{entry['image_name']} / {entry['mode']}", fill="black")
        sheet.paste(thumb, (x, y + 20))
    sheet.save(root / output_name, quality=90)


def main():
    args = build_parser().parse_args()
    if not 0 <= args.mask_ratio <= 1 or args.anchor_size < 14 or args.anchor_size % 14:
        raise ValueError("mask-ratio must be in [0,1]; anchor-size must be a positive multiple of 14")
    if args.cpu_threads < 1 or args.limit < 0:
        raise ValueError("cpu-threads must be positive and limit nonnegative")
    config = SpanConfig(span_fraction=args.span_fraction, band_mask_ratio=args.band_mask_ratio,
                        context_rows=args.context_rows, band_half_width_fraction=args.band_half_width_fraction,
                        min_coverage=args.min_coverage, min_contrast=args.min_contrast, max_border_mass=args.max_border_mass)
    entries = _inputs(args)
    root = args.output_dir.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Use a new --output-dir; refusing to overwrite {root}")
    torch.set_num_threads(args.cpu_threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    loaded = load_dino_model(args.weights, architecture="dinov2", device=device)
    if not 1 <= args.layer <= loaded.num_layers:
        raise ValueError(f"layer must be in 1..{loaded.num_layers}")
    heads = (list(range(1, loaded.num_heads + 1)) if args.heads == "all"
             else sorted(set(int(value) for value in args.heads.split(","))))
    if not heads or min(heads) < 1 or max(heads) > loaded.num_heads:
        raise ValueError(f"heads must be one-based indices in 1..{loaded.num_heads}")
    from methods.geotopo_dino.data.collate import _block_mask_within_valid

    root.mkdir(parents=True)
    manifest = {"arguments": vars(args), "span_config": asdict(config), "device": str(device),
                "weights": str(loaded.weight_path), "checkpoint_bytes": loaded.weight_path.stat().st_size,
                "checkpoint_mtime_ns": loaded.weight_path.stat().st_mtime_ns,
                "torch_version": torch.__version__, "selected_heads_one_based": heads,
                "algorithm_version": 2,
                "mask_display": "Green=band, blue=span, purple=supplement, orange=visible context in mask_steps.png.",
                "training_integration": False}
    (root / "run_config.json").write_text(json.dumps(manifest, indent=2, default=_metadata_value) + "\n")
    completed, failures = [], []
    modes = ("clean", "train") if args.anchor_mode == "both" else (args.anchor_mode,)
    print(f"Loaded {loaded.weight_path}; {len(entries)} images x {len(modes)} modes on {device}", flush=True)
    for path, key in entries:
        for mode in modes:
            print(f"Processing {path.name} [{mode}]", flush=True)
            try:
                sample_seed = (args.seed + int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16)) % (2**32)
                with Image.open(path) as opened:
                    original = ImageOps.exif_transpose(opened).convert("RGB")
                tensor, geometry, anchor = make_anchor(original, size=args.anchor_size, mode=mode, seed=sample_seed)
                valid = geometry["valid_mask"]
                extracted = extract_layer_maps(loaded, tensor.to(device), layers=sorted({args.layer - 1, loaded.num_layers - 1}))
                layer_maps = next(item for item in extracted.layers if item.layer == args.layer)
                final_maps = next(item for item in extracted.layers if item.layer == loaded.num_layers)
                all_attention = torch.from_numpy(layer_maps.attention_heads)
                selected = fuse_attention(all_attention[[index - 1 for index in heads]], valid)
                mean_attention = fuse_attention(all_attention, valid)
                maps = {"attention_heads": layer_maps.attention_heads,
                        "attention_patch_mass": layer_maps.attention_patch_mass,
                        "selected_attention": selected.numpy(), "mean_attention": mean_attention.numpy(),
                        "cosine": final_maps.cosine, "selected_layer_cosine": layer_maps.cosine}
                response = torch.from_numpy(maps[args.response_source]).clamp(min=0)
                target = int(int(valid.sum()) * args.mask_ratio)
                with seeded_cpu_random(sample_seed + 1):
                    block = _block_mask_within_valid(valid, target)
                result = generate_mask_comparison(response, valid, block, config=config, seed=sample_seed + 2)
                destination = root / key / mode
                details = {"image": str(path.resolve()), "mode": mode, "seed": sample_seed,
                           "geometry": geometry, "diagnostics": result.diagnostics,
                           "valid_attention_mass_per_head": (all_attention * valid).sum((-2, -1)).tolist()}
                _save_case(destination, anchor, maps, response, valid, result, details,
                           layer=args.layer, heads=heads, source=args.response_source,
                           title=f"{path.name} | {mode}")
                record = {"image_name": path.name, "image": str(path.resolve()), "mode": mode,
                          "output": str(destination.relative_to(root)), **result.diagnostics}
                completed.append(record)
                print(f"  target={target}; span={result.diagnostics['span_count']}; "
                      f"fallback={result.diagnostics['fallback_reasons']}", flush=True)
            except Exception as exc:
                failures.append({"image": str(path), "mode": mode, "error": f"{type(exc).__name__}: {exc}"})
                print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
                if args.fail_fast:
                    (root / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
                    raise
    summary = {"completed": completed, "failures": failures, "success_count": len(completed),
               "failure_count": len(failures),
               "topology_usage_rate": sum(not entry["used_fallback"] for entry in completed) / max(len(completed), 1)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ("image_name", "mode", "used_fallback", "fallback_reasons", "coverage", "contrast", "border_mass",
              "valid_count", "target_count", "band_count", "span_count", "span_rows", "span_band_fraction", "band_mask_ratio_actual", "span_total_mask_fraction",
              "context_count", "output")
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(completed)
    _write_overview(root, completed)
    print(f"Saved {len(completed)} comparisons, {len(failures)} failures to {root}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
