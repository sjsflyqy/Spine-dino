"""Compare a selected spine patch with every patch across DINO checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visualization.dino_spine_maps.map_extractor import extract_layer_maps
from visualization.dino_spine_maps.model_loader import load_dino_model
from visualization.dino_spine_maps.preprocessing import PreparedImage, prepare_image
from visualization.dino_spine_maps.rendering import _save_inputs, save_raw_maps
from visualization.dino_spine_maps.visualize import _device, _layers, _safe_name


@dataclass(frozen=True)
class QueryPatch:
    row: int
    column: int
    index: int
    original_xy: tuple[float, float]
    original_box_xyxy: tuple[float, float, float, float]


def select_query_patch(
    prepared: PreparedImage,
    *,
    xy: tuple[float, float] | None = None,
    relative_xy: tuple[float, float] | None = None,
    patch_rc: tuple[int, int] | None = None,
) -> QueryPatch:
    """Map an original-image point to a patch using actual rounded resize sizes.

    Coordinates refer to the EXIF-corrected original image. Patch indices are
    zero-based and include the model-input padding.
    """
    if sum(value is not None for value in (xy, relative_xy, patch_rc)) != 1:
        raise ValueError("Choose exactly one of xy, relative_xy, patch_rc")
    g = prepared.geometry
    sx, sy = g.resized_width / g.original_width, g.resized_height / g.original_height
    if relative_xy is not None:
        u, v = relative_xy
        if not (0 <= u < 1 and 0 <= v < 1):
            raise ValueError("Relative query coordinates must be in [0, 1)")
        xy = (u * g.original_width, v * g.original_height)
    if xy is not None:
        x, y = xy
        if not (0 <= x < g.original_width and 0 <= y < g.original_height):
            raise ValueError(f"Query {xy} outside original image {prepared.original.size}")
        column = int((x * sx + g.pad_left) // g.patch_size)
        row = int((y * sy + g.pad_top) // g.patch_size)
    else:
        row, column = patch_rc
    if not (0 <= row < g.grid_height and 0 <= column < g.grid_width):
        raise ValueError(f"Query patch {(row, column)} outside grid {(g.grid_height, g.grid_width)}")
    if not prepared.valid_patch_mask[row, column]:
        raise ValueError("Query patch is padding or has insufficient real-image area")
    left = max(0.0, (column * g.patch_size - g.pad_left) / sx)
    top = max(0.0, (row * g.patch_size - g.pad_top) / sy)
    right = min(float(g.original_width), ((column + 1) * g.patch_size - g.pad_left) / sx)
    bottom = min(float(g.original_height), ((row + 1) * g.patch_size - g.pad_top) / sy)
    if xy is None:
        xy = ((left + right) / 2, (top + bottom) / 2)
    return QueryPatch(row, column, row * g.grid_width + column, tuple(xy), (left, top, right, bottom))


def _mark_query(axis, query: QueryPatch, *, grid: bool = False) -> None:
    if grid:
        left, top, width, height = query.column - 0.5, query.row - 0.5, 1, 1
    else:
        left, top, right, bottom = query.original_box_xyxy
        width, height = right - left, bottom - top
    axis.add_patch(Rectangle((left, top), width, height, fill=False, edgecolor="red", linewidth=1.5))


def save_comparison(
    path: Path,
    entries: list[tuple[str, PreparedImage, QueryPatch, np.ndarray]],
    *,
    title: str,
    overlay: bool,
    vmin: float,
    vmax: float,
    alpha: float,
) -> None:
    """Render original + maps; raw mode keeps the unsmoothed patch grid."""
    figure, axes = plt.subplots(
        1, len(entries) + 1, figsize=(3.5 * (len(entries) + 1), 7),
        squeeze=False, constrained_layout=True,
    )
    axes = axes[0]
    axes[0].imshow(entries[0][1].original)
    # Different patch sizes can produce different reference footprints.
    for _, _, query, _ in entries:
        _mark_query(axes[0], query)
    axes[0].set_title("Image / query patch")
    axes[0].axis("off")
    for axis, (label, prepared, query, values) in zip(axes[1:], entries):
        grid = np.where(prepared.valid_patch_mask, values, np.nan)
        if overlay:
            axis.imshow(prepared.original)
            mapped = prepared.geometry.grid_to_original(np.nan_to_num(grid), nearest=True)
            valid = prepared.geometry.grid_to_original(prepared.valid_patch_mask, nearest=True) > 0.5
            mapped = np.where(valid, mapped, np.nan)
            artist = axis.imshow(mapped, cmap="viridis", vmin=vmin, vmax=vmax, alpha=alpha, interpolation="nearest")
        else:
            artist = axis.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        _mark_query(axis, query, grid=not overlay)
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(title)
    figure.colorbar(artist, ax=axes[1:].tolist(), fraction=0.025, pad=0.02, label="Cosine similarity")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, nargs="+", required=True, help="One or more extracted backbone checkpoints, in display order")
    parser.add_argument("--labels", nargs="+", help="Display labels, one per checkpoint")
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--query-xy", type=float, nargs=2, metavar=("X", "Y"), help="Spine point in original-image pixels (x,y)")
    query.add_argument("--query-relative", type=float, nargs=2, metavar=("U", "V"), help="Original-image fractions in [0,1)")
    query.add_argument("--query-patch", type=int, nargs=2, metavar=("ROW", "COL"), help="Zero-based patch row/column; comparisons require identical grids")
    parser.add_argument("--layers", default="last", help="last, all, 12, 5-12, or 3,6,9,12")
    parser.add_argument("--architecture", choices=("auto", "dinov2", "dinov3"), default="auto")
    parser.add_argument("--long-side", type=int, default=896)
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--vmin", type=float, default=-1.0, help="Shared color minimum; raw similarities are never rescaled")
    parser.add_argument("--vmax", type=float, default=1.0, help="Shared color maximum")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/patch_similarity"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.weights):
        parser.error("--labels must contain one label per --weights checkpoint")
    if not (np.isfinite(args.vmin) and np.isfinite(args.vmax) and -1 <= args.vmin < args.vmax <= 1):
        parser.error("Color limits must satisfy -1 <= vmin < vmax <= 1")
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be in [0,1]")
    device = _device(args.device)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparison: dict[int, list] = {}
    records = []
    first_grid = None
    for number, weights in enumerate(args.weights):
        print(f"[{number + 1}/{len(args.weights)}] Loading {weights}", flush=True)
        loaded = load_dino_model(weights, architecture=args.architecture, device=device)
        prepared = prepare_image(
            args.image, patch_size=loaded.patch_size, long_side=args.long_side,
            min_valid_fraction=args.min_valid_fraction,
        )
        grid_spec = (loaded.patch_size, prepared.geometry.grid_height, prepared.geometry.grid_width)
        if args.query_patch is not None and first_grid is not None and grid_spec != first_grid:
            raise ValueError("--query-patch comparisons require identical patch sizes/grids; use --query-xy or --query-relative")
        first_grid = grid_spec
        query = select_query_patch(
            prepared, xy=args.query_xy, relative_xy=args.query_relative,
            patch_rc=args.query_patch,
        )
        layers = [loaded.num_layers - 1] if args.layers == "last" else _layers(args.layers, loaded.num_layers)
        result = extract_layer_maps(
            loaded, prepared.tensor.to(device), layers=layers,
            query_patch=(query.row, query.column),
        )
        label = args.labels[number] if args.labels else f"{weights.parent.name}/{weights.stem}"
        destination = output / f"{number + 1:02d}__{_safe_name(label)}"
        destination.mkdir(parents=True, exist_ok=True)
        _save_inputs(destination, prepared)
        save_raw_maps(destination, prepared, result)
        record = {
            "image": str(args.image.expanduser().resolve()), "weights": str(loaded.weight_path),
            "label": label, "architecture": loaded.architecture,
            "feature_space": "backbone block output followed by model.norm (no GCVD projection head)",
            "query": asdict(query), "geometry": prepared.geometry.as_dict(),
            "selected_layers": [layer.layer for layer in result.layers],
            "color_limits": [args.vmin, args.vmax], "output": str(destination),
        }
        (destination / "metadata.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        records.append(record)
        for layer in result.layers:
            values = layer.patch_cosine
            if values is None or not np.isfinite(values).all():
                raise RuntimeError("Missing or non-finite patch similarities")
            if values.min() < -1.0001 or values.max() > 1.0001:
                raise RuntimeError("Patch cosine similarity outside [-1,1]")
            entries = [(label, prepared, query, values)]
            comparison.setdefault(layer.layer, []).extend(entries)
            for overlay in (False, True):
                suffix = "overlay" if overlay else "grid"
                save_comparison(
                    destination / f"layer_{layer.layer:02d}_{suffix}.png", entries,
                    title=f"Layer {layer.layer}: query patch to all patches",
                    overlay=overlay, vmin=args.vmin, vmax=args.vmax, alpha=args.alpha,
                )
        print(f"  Query patch (row,col)=({query.row},{query.column}); saved {destination}", flush=True)
        # Release each model before loading the next checkpoint.
        del result, loaded
    for layer, entries in comparison.items():
        for overlay in (False, True):
            suffix = "overlay" if overlay else "grid"
            save_comparison(
                output / f"comparison_layer_{layer:02d}_{suffix}.png", entries,
                title=f"Layer {layer}: query patch to all patches (shared color scale)",
                overlay=overlay, vmin=args.vmin, vmax=args.vmax, alpha=args.alpha,
            )
    (output / "comparison_metadata.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Saved comparison to {output}")


if __name__ == "__main__":
    main()
