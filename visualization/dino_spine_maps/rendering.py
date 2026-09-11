"""Render patch-grid response maps on their original radiograph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .map_extractor import ExtractionResult, LayerMaps
from .preprocessing import PreparedImage


def _finite_limits(arrays: Sequence[np.ndarray], *, positive: bool = False) -> tuple[float, float]:
    values = np.concatenate([np.asarray(array)[np.isfinite(array)] for array in arrays])
    if values.size == 0:
        return (0.0, 1.0)
    lower = 0.0 if positive else float(np.percentile(values, 2.0))
    upper = float(np.percentile(values, 98.0))
    if upper <= lower:
        upper = lower + max(abs(lower) * 1e-6, 1e-6)
    return lower, upper


def _map_to_original(prepared: PreparedImage, grid: np.ndarray) -> np.ndarray:
    masked = np.asarray(grid, dtype=np.float32).copy()
    masked[~prepared.valid_patch_mask] = np.nan
    # PIL cannot reliably interpolate NaNs. Interpolate data and validity
    # separately, then restore invalid pixels after inverse geometry mapping.
    values = np.nan_to_num(masked, nan=0.0)
    validity = np.isfinite(masked).astype(np.float32)
    numerator = prepared.geometry.grid_to_original(values)
    denominator = prepared.geometry.grid_to_original(validity)
    result = numerator / np.maximum(denominator, 1e-6)
    result[denominator < 0.5] = np.nan
    return result


def _save_overlay(
    path: Path,
    prepared: PreparedImage,
    grid: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    alpha: float,
) -> None:
    original_map = _map_to_original(prepared, grid)
    figure, axis = plt.subplots(figsize=(8, 12), constrained_layout=True)
    axis.imshow(prepared.original)
    overlay = axis.imshow(
        original_map,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=np.where(np.isfinite(original_map), alpha, 0.0),
    )
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(overlay, ax=axis, fraction=0.035, pad=0.02)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_contact_sheet(
    path: Path,
    prepared: PreparedImage,
    layers: Sequence[LayerMaps],
    selector: Callable[[LayerMaps], np.ndarray],
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    alpha: float,
) -> None:
    count = len(layers)
    columns = min(4, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 5.2 * rows), squeeze=False
    )
    last_overlay = None
    for axis, layer in zip(axes.flat, layers):
        axis.imshow(prepared.original)
        mapped = _map_to_original(prepared, selector(layer))
        last_overlay = axis.imshow(
            mapped,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=np.where(np.isfinite(mapped), alpha, 0.0),
        )
        axis.set_title(f"Layer {layer.layer}")
        axis.axis("off")
    for axis in axes.flat[count:]:
        axis.axis("off")
    figure.suptitle(title, fontsize=15)
    if last_overlay is not None:
        figure.colorbar(last_overlay, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_head_sheet(
    path: Path,
    prepared: PreparedImage,
    layer: LayerMaps,
    *,
    alpha: float,
) -> None:
    heads = layer.attention_heads
    valid_heads = [np.where(prepared.valid_patch_mask, head, np.nan) for head in heads]
    vmin, vmax = _finite_limits(valid_heads, positive=True)
    columns = 4
    rows = int(np.ceil(heads.shape[0] / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 5.2 * rows), squeeze=False)
    last_overlay = None
    for head_index, (axis, grid) in enumerate(zip(axes.flat, heads)):
        axis.imshow(prepared.original)
        mapped = _map_to_original(prepared, grid)
        last_overlay = axis.imshow(
            mapped,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            alpha=np.where(np.isfinite(mapped), alpha, 0.0),
        )
        axis.set_title(
            f"Head {head_index + 1} | patch mass={layer.attention_patch_mass[head_index]:.3f}"
        )
        axis.axis("off")
    for axis in axes.flat[heads.shape[0] :]:
        axis.axis("off")
    figure.suptitle(f"Layer {layer.layer}: CLS attention heads", fontsize=15)
    if last_overlay is not None:
        figure.colorbar(last_overlay, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _save_inputs(output_dir: Path, prepared: PreparedImage) -> None:
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    prepared.original.save(input_dir / "original.png")
    prepared.model_image.save(input_dir / "model_input.png")

    pixel_valid = np.zeros(
        (prepared.geometry.model_height, prepared.geometry.model_width), dtype=np.uint8
    )
    geometry = prepared.geometry
    pixel_valid[
        geometry.pad_top : geometry.pad_top + geometry.resized_height,
        geometry.pad_left : geometry.pad_left + geometry.resized_width,
    ] = 255
    Image.fromarray(pixel_valid, mode="L").save(input_dir / "valid_region.png")


def save_raw_maps(
    output_dir: Path, prepared: PreparedImage, result: ExtractionResult
) -> None:
    payload: dict[str, np.ndarray] = {
        "valid_patch_mask": prepared.valid_patch_mask.astype(np.bool_, copy=False)
    }
    for layer in result.layers:
        prefix = f"layer_{layer.layer:02d}"
        payload[f"{prefix}_attention_heads"] = layer.attention_heads
        payload[f"{prefix}_attention_mean"] = layer.attention_mean
        payload[f"{prefix}_attention_max"] = layer.attention_max
        payload[f"{prefix}_attention_patch_mass"] = layer.attention_patch_mass
        payload[f"{prefix}_cosine"] = layer.cosine
    np.savez_compressed(output_dir / "raw_maps.npz", **payload)


def render_result(
    output_dir: str | Path,
    prepared: PreparedImage,
    result: ExtractionResult,
    *,
    maps: set[str],
    save_heads: bool = False,
    alpha: float = 0.55,
    metadata: dict | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _save_inputs(output, prepared)
    save_raw_maps(output, prepared, result)

    layers = list(result.layers)
    if "attention" in maps:
        attention_dir = output / "attention"
        attention_dir.mkdir(exist_ok=True)
        limits = _finite_limits(
            [
                np.where(prepared.valid_patch_mask, layer.attention_mean, np.nan)
                for layer in layers
            ],
            positive=True,
        )
        for layer in layers:
            _save_overlay(
                attention_dir / f"layer_{layer.layer:02d}_mean.png",
                prepared,
                layer.attention_mean,
                title=f"Layer {layer.layer}: mean CLS-to-patch attention",
                cmap="magma",
                vmin=limits[0],
                vmax=limits[1],
                alpha=alpha,
            )
            if save_heads:
                _save_head_sheet(
                    attention_dir / f"layer_{layer.layer:02d}_heads.png",
                    prepared,
                    layer,
                    alpha=alpha,
                )
        _save_contact_sheet(
            attention_dir / "all_layers.png",
            prepared,
            layers,
            lambda layer: layer.attention_mean,
            title="Mean CLS-to-patch attention (shared color scale)",
            cmap="magma",
            vmin=limits[0],
            vmax=limits[1],
            alpha=alpha,
        )

    if "cosine" in maps:
        cosine_dir = output / "cosine"
        cosine_dir.mkdir(exist_ok=True)
        limits = _finite_limits(
            [
                np.where(prepared.valid_patch_mask, layer.cosine, np.nan)
                for layer in layers
            ],
            positive=False,
        )
        for layer in layers:
            _save_overlay(
                cosine_dir / f"layer_{layer.layer:02d}.png",
                prepared,
                layer.cosine,
                title=f"Layer {layer.layer}: CLS-to-patch cosine similarity",
                cmap="turbo",
                vmin=limits[0],
                vmax=limits[1],
                alpha=alpha,
            )
        _save_contact_sheet(
            cosine_dir / "all_layers.png",
            prepared,
            layers,
            lambda layer: layer.cosine,
            title="CLS-to-patch cosine similarity (shared color scale)",
            cmap="turbo",
            vmin=limits[0],
            vmax=limits[1],
            alpha=alpha,
        )

    if metadata is not None:
        with (output / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
    return output
