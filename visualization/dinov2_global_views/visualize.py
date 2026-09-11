"""Batch CLI for attention/cosine maps on native DINOv2 global views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import torch


if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT))
    from visualization.dino_spine_maps.map_extractor import extract_layer_maps
    from visualization.dino_spine_maps.model_loader import load_dino_model
    from visualization.dino_spine_maps.rendering import render_result
    from visualization.dinov2_global_views.global_views import (
        GLOBAL_CROP_SIZE,
        create_dinov2_global_views,
    )
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    from visualization.dino_spine_maps.map_extractor import extract_layer_maps
    from visualization.dino_spine_maps.model_loader import load_dino_model
    from visualization.dino_spine_maps.rendering import render_result
    from .global_views import GLOBAL_CROP_SIZE, create_dinov2_global_views


SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _layers(value: str, depth: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(depth))
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending layer range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    invalid = sorted(layer for layer in selected if layer < 1 or layer > depth)
    if invalid:
        raise ValueError(f"Layers must be between 1 and {depth}; got {invalid}")
    if not selected:
        raise ValueError("No layers selected")
    return [layer - 1 for layer in sorted(selected)]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def _collect_images(
    image: Path | None, image_dir: Path | None, recursive: bool
) -> tuple[list[tuple[Path, Path]], Path | None]:
    if image is not None:
        path = image.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return [(path, Path(path.stem))], None
    assert image_dir is not None
    root = image_dir.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    iterator = root.rglob("*") if recursive else root.glob("*")
    images = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No supported raster images found in {root}")
    return [(path, path.relative_to(root)) for path in images], root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the two native DINOv2 global crops, then visualize layer-wise "
            "CLS attention and CLS/patch cosine similarity."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path)
    inputs.add_argument("--image-dir", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--layers", default="all", help="all, 12, 5-12, or 3,6,9,12")
    parser.add_argument(
        "--maps",
        nargs="+",
        choices=("attention", "cosine"),
        default=("attention", "cosine"),
    )
    parser.add_argument("--save-heads", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--global-crop-scale", type=float, nargs=2, default=(0.5, 1.0), metavar=("MIN", "MAX")
    )
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "visualization" / "dinov2_global_views",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be in [0, 1]")
    device = _device(args.device)
    loaded = load_dino_model(args.weights, architecture="auto", device=device)
    if loaded.generation != 2 or loaded.patch_size != 14:
        raise ValueError("This visualizer accepts only native DINOv2 ViT-B/14 checkpoints")
    selected_layers = _layers(args.layers, loaded.num_layers)
    images, input_root = _collect_images(args.image, args.image_dir, args.recursive)
    scale = tuple(float(value) for value in args.global_crop_scale)
    checkpoint_name = (
        f"{_safe_name(loaded.weight_path.parent.name)}__{_safe_name(loaded.weight_path.stem)}"
    )
    output_root = args.output_dir.expanduser().resolve()
    completed: list[dict] = []
    failures: list[dict] = []

    print(
        f"Loaded {loaded.architecture}: {loaded.weight_path}\n"
        f"DINOv2 global views: 2 x {GLOBAL_CROP_SIZE}x{GLOBAL_CROP_SIZE}, scale={scale}\n"
        f"Found {len(images)} source image(s); output root: {output_root}"
    )
    for image_index, (image_path, relative_path) in enumerate(images, start=1):
        image_seed = args.seed + image_index - 1
        destination = output_root / relative_path / checkpoint_name
        print(f"[{image_index}/{len(images)}] {image_path} (seed={image_seed})", flush=True)
        try:
            global_views = create_dinov2_global_views(
                image_path,
                repo_root=REPO_ROOT,
                patch_size=loaded.patch_size,
                seed=image_seed,
                global_crop_scale=scale,
            )
            destination.mkdir(parents=True, exist_ok=True)
            global_views.source_image.save(destination / "source_image.png")
            view_records = []
            for view_index, prepared in enumerate(global_views.views, start=1):
                view_output = destination / f"global_view_{view_index}"
                image_tensor = prepared.tensor.to(
                    device, non_blocking=device.type == "cuda"
                )
                result = extract_layer_maps(
                    loaded, image_tensor, layers=selected_layers
                )
                view_metadata = {
                    "source_image": str(image_path),
                    "global_view": view_index,
                    "augmentation": "upstream DINOv2 DataAugmentationDINO",
                    "seed": image_seed,
                    "global_crop_size": GLOBAL_CROP_SIZE,
                    "global_crop_scale": list(scale),
                    "horizontal_flip_probability": 0.5,
                    "weights": str(loaded.weight_path),
                    "architecture": loaded.architecture,
                    "patch_size": loaded.patch_size,
                    "patch_grid": [
                        prepared.geometry.grid_height,
                        prepared.geometry.grid_width,
                    ],
                    "selected_layers": [index + 1 for index in selected_layers],
                    "maps": list(args.maps),
                    "attention_patch_mass": {
                        f"layer_{layer.layer:02d}": layer.attention_patch_mass.tolist()
                        for layer in result.layers
                    },
                }
                render_result(
                    view_output,
                    prepared,
                    result,
                    maps=set(args.maps),
                    save_heads=args.save_heads,
                    alpha=args.alpha,
                    metadata=view_metadata,
                )
                prepared.original.save(view_output / "global_view.png")
                view_records.append(str(view_output.resolve()))
            completed.append(
                {"image": str(image_path), "seed": image_seed, "views": view_records}
            )
        except Exception as exc:
            failure = {
                "image": str(image_path),
                "seed": image_seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(
                f"  FAILED ({failure['error_type']}): {failure['error']}",
                file=sys.stderr,
                flush=True,
            )
            if args.fail_fast:
                raise

    output_root.mkdir(parents=True, exist_ok=True)
    summary = output_root / f"batch_summary__{checkpoint_name}.json"
    with summary.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "input": str(input_root or images[0][0]),
                "weights": str(loaded.weight_path),
                "architecture": loaded.architecture,
                "augmentation": "upstream DINOv2 DataAugmentationDINO",
                "global_crop_size": GLOBAL_CROP_SIZE,
                "global_crop_scale": list(scale),
                "base_seed": args.seed,
                "images_found": len(images),
                "images_completed": len(completed),
                "images_failed": len(failures),
                "completed": completed,
                "failures": failures,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    print(
        f"Finished: {len(completed)} succeeded, {len(failures)} failed. Summary: {summary}"
    )
    if not completed:
        raise RuntimeError(f"Every source image failed; see {summary}")


if __name__ == "__main__":
    main()
