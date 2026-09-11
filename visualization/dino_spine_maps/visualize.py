"""CLI for layer-wise DINO attention and CLS/patch cosine visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch


if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT))
    from visualization.dino_spine_maps.map_extractor import extract_layer_maps
    from visualization.dino_spine_maps.model_loader import load_dino_model
    from visualization.dino_spine_maps.preprocessing import prepare_image
    from visualization.dino_spine_maps.rendering import render_result
else:
    from .map_extractor import extract_layer_maps
    from .model_loader import load_dino_model
    from .preprocessing import prepare_image
    from .rendering import render_result


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


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


SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _collect_images(
    *, image: Path | None, image_dir: Path | None, recursive: bool
) -> tuple[list[tuple[Path, Path]], Path | None]:
    """Return ``(absolute image, relative output key)`` pairs."""

    if image is not None:
        absolute = image.expanduser().resolve()
        if not absolute.is_file():
            raise FileNotFoundError(f"Input image does not exist: {absolute}")
        return [(absolute, Path(absolute.stem))], None

    assert image_dir is not None
    root = image_dir.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input image directory does not exist: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    images = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not images:
        mode = "recursively" if recursive else "at its top level"
        suffixes = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise FileNotFoundError(
            f"No supported images found {mode} in {root}. Supported suffixes: {suffixes}"
        )
    # Keep the filename extension as part of the output directory name so
    # ``case.jpg`` and ``case.png`` can never overwrite one another.
    return [(path, path.relative_to(root)) for path in images], root


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Visualize every DINO ViT block's CLS attention and CLS/patch cosine map."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, help="One input raster radiograph")
    inputs.add_argument("--image-dir", type=Path, help="A directory containing raster radiographs")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan --image-dir and preserve its relative directory structure",
    )
    parser.add_argument("--weights", type=Path, required=True, help="Extracted teacher/backbone .pth")
    parser.add_argument(
        "--architecture", choices=("auto", "dinov2", "dinov3"), default="auto"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--long-side", type=int, default=896, help="Aspect-preserving resized long side"
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.5,
        help="Minimum real-image fraction for a patch to be visualized",
    )
    parser.add_argument("--layers", default="all", help="all, 12, 5-12, or 3,6,9,12")
    parser.add_argument(
        "--maps",
        nargs="+",
        choices=("attention", "cosine"),
        default=("attention", "cosine"),
    )
    parser.add_argument("--save-heads", action="store_true", help="Save a 12-head sheet per layer")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed image instead of completing the batch",
    )
    parser.add_argument("--alpha", type=float, default=0.55, help="Overlay opacity in [0,1]")
    parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "outputs" / "visualization"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be in [0, 1]")
    device = _device(args.device)
    loaded = load_dino_model(
        args.weights, architecture=args.architecture, device=device
    )
    selected_layers = _layers(args.layers, loaded.num_layers)
    image_entries, input_root = _collect_images(
        image=args.image, image_dir=args.image_dir, recursive=args.recursive
    )
    checkpoint_group = _safe_name(loaded.weight_path.parent.name)
    checkpoint_name = _safe_name(loaded.weight_path.stem)
    checkpoint_output_name = f"{checkpoint_group}__{checkpoint_name}"
    output_root = args.output_dir.expanduser().resolve()
    completed: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    print(
        f"Loaded {loaded.architecture} from {loaded.weight_path}\n"
        f"Found {len(image_entries)} image(s); output root: {output_root}"
    )
    for image_index, (image_path, relative_key) in enumerate(image_entries, start=1):
        print(f"[{image_index}/{len(image_entries)}] {image_path}", flush=True)
        destination = output_root / relative_key / checkpoint_output_name
        try:
            prepared = prepare_image(
                image_path,
                patch_size=loaded.patch_size,
                long_side=args.long_side,
                min_valid_fraction=args.min_valid_fraction,
            )
            image_tensor = prepared.tensor.to(
                device, non_blocking=device.type == "cuda"
            )
            result = extract_layer_maps(
                loaded, image_tensor, layers=selected_layers
            )
            metadata = {
                "image": str(image_path),
                "input_root": str(input_root) if input_root is not None else None,
                "relative_image": str(relative_key),
                "weights": str(loaded.weight_path),
                "architecture": loaded.architecture,
                "generation": loaded.generation,
                "device": str(device),
                "patch_size": loaded.patch_size,
                "embedding_dim": loaded.embedding_dim,
                "num_layers": loaded.num_layers,
                "num_heads": loaded.num_heads,
                "num_extra_tokens": loaded.num_extra_tokens,
                "selected_layers": [index + 1 for index in selected_layers],
                "maps": list(args.maps),
                "geometry": prepared.geometry.as_dict(),
                "valid_patch_fraction": float(np.mean(prepared.valid_patch_mask)),
                "attention_patch_mass": {
                    f"layer_{layer.layer:02d}": layer.attention_patch_mass.tolist()
                    for layer in result.layers
                },
            }
            render_result(
                destination,
                prepared,
                result,
                maps=set(args.maps),
                save_heads=args.save_heads,
                alpha=args.alpha,
                metadata=metadata,
            )
            completed.append(
                {"image": str(image_path), "output": str(destination.resolve())}
            )
        except Exception as exc:
            failure = {
                "image": str(image_path),
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
    summary_path = output_root / f"batch_summary__{checkpoint_output_name}.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "input": str(input_root or image_entries[0][0]),
                "weights": str(loaded.weight_path),
                "architecture": loaded.architecture,
                "images_found": len(image_entries),
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
        f"Finished: {len(completed)} succeeded, {len(failures)} failed. "
        f"Summary: {summary_path}"
    )
    if not completed:
        raise RuntimeError(f"Visualization failed for every input image; see {summary_path}")


if __name__ == "__main__":
    main()
