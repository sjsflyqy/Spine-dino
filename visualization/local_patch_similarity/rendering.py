"""Static figures with shared cosine scales and explicit crop/query geometry."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

from methods.geotopo_dino.geometry.transforms import transform_points
from .views import display_tensor


def _image(axis, pixels):
    height, width = pixels.shape[:2]
    axis.imshow(pixels, extent=(0, width, height, 0))
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.axis("off")


def _mark_query(axis, views, query, local):
    point = transform_points(views.transforms[local], torch.tensor(query["source_xy"]))
    size = views.locals.shape[-1]
    if ((point >= 0) & (point < size)).all():
        axis.plot(*point.tolist(), marker="+", color="lime", markersize=10, markeredgewidth=1.5)
    if local == query["local"]:
        axis.add_patch(Rectangle((query["column"] * views.patch_size, query["row"] * views.patch_size),
                                 views.patch_size, views.patch_size, fill=False, edgecolor="red", linewidth=1.5))


def save_inputs(output, views, queries):
    inputs = output / "inputs"
    inputs.mkdir()
    views.source.save(inputs / "source.png")
    Image.fromarray(display_tensor(views.anchor)).save(inputs / "anchor.png")
    for index, tensor in enumerate(views.locals):
        Image.fromarray(display_tensor(tensor)).save(inputs / f"local_{index}.png")
    figure = plt.figure(figsize=(17, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 5, width_ratios=[1.5, 1, 1, 1, 1])
    source_axis = figure.add_subplot(grid[:, 0])
    _image(source_axis, np.asarray(views.source))
    source_axis.set_title("Source / eight crop boxes")
    for index, geometry in enumerate(views.local_geometry):
        left, top, right, bottom = geometry["box_xyxy"].tolist()
        color = plt.get_cmap("tab10")(index)
        source_axis.add_patch(Rectangle((left, top), right - left, bottom - top,
                                        fill=False, edgecolor=color, linewidth=1.5))
        source_axis.text(left, top, f"L{index}", color="white", fontsize=9,
                         bbox={"facecolor": color, "alpha": 0.8, "pad": 1})
        axis = figure.add_subplot(grid[index // 4, index % 4 + 1])
        _image(axis, display_tensor(views.locals[index]))
        for query in queries:
            _mark_query(axis, views, query, index)
        axis.set_title(f"L{index}" + (" (flipped)" if geometry["flipped"] else ""), color=color)
    for number, query in enumerate(queries):
        source_axis.plot(*query["source_xy"], marker="+", color="red")
        source_axis.annotate(f"Q{number}", query["source_xy"], color="red")
    figure.suptitle("Red box = query patch; green + = same source location when covered")
    figure.savefig(output / "crop_layout.png", dpi=160)
    plt.close(figure)


def save_matrix_comparison(output, before, after, *, labels, vmin, vmax, delta_limit, pooled_only=False):
    matrices = [
        ("patch_similarity", "patch_similarity.png", "Patch cosine (row-major patches within L0 ... L7)"),
        ("local_pair_mean", "local_similarity_8x8.png", "Mean patch-pair cosine; self-patch terms excluded on diagonal"),
        ("local_pooled_similarity", "local_pooled_similarity_8x8.png",
         "Whole-local cosine: mean-pool all raw patch tokens, then L2 normalize; self-similarity = 1"),
    ]
    if pooled_only:
        matrices = matrices[-1:]
    for key, filename, title in matrices:
        values = [before[key].copy(), after[key].copy(), after[key] - before[key]]
        if key == "patch_similarity":
            for value in values:
                np.fill_diagonal(value, np.nan)
        figure, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
        for index, (axis, value) in enumerate(zip(axes, values)):
            delta = index == 2
            artist = axis.imshow(value, interpolation="nearest", cmap="RdBu_r" if delta else "viridis",
                                  vmin=-delta_limit if delta else vmin,
                                  vmax=delta_limit if delta else vmax)
            if key == "patch_similarity":
                n = value.shape[0] // 8
                axis.set_xticks((np.arange(8) + 0.5) * n - 0.5, [f"L{i}" for i in range(8)])
                axis.set_yticks((np.arange(8) + 0.5) * n - 0.5, [f"L{i}" for i in range(8)])
                for boundary in np.arange(1, 8) * n - 0.5:
                    axis.axhline(boundary, color="white", linewidth=0.5)
                    axis.axvline(boundary, color="white", linewidth=0.5)
            else:
                axis.set_xticks(range(8), [f"L{i}" for i in range(8)])
                axis.set_yticks(range(8), [f"L{i}" for i in range(8)])
                for row in range(8):
                    for col in range(8):
                        axis.text(col, row, f"{value[row, col]:.2f}", ha="center", va="center", fontsize=7,
                                  color="black" if delta else "white")
            axis.set_title("After - before" if delta else labels[index])
            figure.colorbar(artist, ax=axis, shrink=0.7, label="Cosine difference" if delta else "Cosine")
        figure.suptitle(title)
        figure.savefig(output / filename, dpi=180)
        plt.close(figure)


def save_queries(output, views, queries, before, after, *, labels, vmin, vmax):
    for number, query in enumerate(queries):
        figure, axes = plt.subplots(3, 8, figsize=(22, 9), constrained_layout=True)
        flat_index = query["local"] * views.grid ** 2 + query["index"]
        for local in range(8):
            pixels = display_tensor(views.locals[local])
            for row in range(3):
                _image(axes[row, local], pixels)
                if row:
                    result = (before, after)[row - 1]
                    heat = result["patch_similarity"][flat_index].reshape(8, views.grid, views.grid)[local]
                    artist = axes[row, local].imshow(heat, extent=(0, pixels.shape[1], pixels.shape[0], 0),
                                                     vmin=vmin, vmax=vmax, cmap="viridis", alpha=0.65,
                                                     interpolation="nearest")
                _mark_query(axes[row, local], views, query, local)
                prefix = ("Input", *labels)[row]
                axes[row, local].set_title(f"{prefix} / L{local}", fontsize=9)
        figure.colorbar(artist, ax=axes[1:].flatten().tolist(), shrink=0.65, label="Query-to-patch cosine")
        figure.suptitle(f"Q{number}: L{query['local']} patch ({query['row']}, {query['column']}); "
                       "red = query patch; green + = same source location\n"
                       "Default centre queries are geometric points, not spine annotations")
        figure.savefig(output / f"query_{number:02d}.png", dpi=150)
        plt.close(figure)


def save_alignment(output, views, before, after, *, labels, vmin, vmax):
    for key, filename, low, high, cmap, caption in (
        ("local_anchor_aligned_cosine", "local_anchor_alignment.png", vmin, vmax, "viridis",
         "Cosine with geometrically aligned anchor token"),
        ("anchor_retrieval_error_patches", "anchor_retrieval_error.png", 0, 8, "magma",
         "Anchor retrieval error (anchor patch units; color clipped at 8)"),
    ):
        figure, axes = plt.subplots(2, 8, figsize=(22, 6), constrained_layout=True)
        for row, result in enumerate((before, after)):
            for local in range(8):
                pixels = display_tensor(views.locals[local])
                _image(axes[row, local], pixels)
                heat = result[key][local].reshape(views.grid, views.grid)
                artist = axes[row, local].imshow(heat, extent=(0, pixels.shape[1], pixels.shape[0], 0),
                                                 vmin=low, vmax=high, cmap=cmap, alpha=0.7,
                                                 interpolation="nearest")
                axes[row, local].set_title(f"{labels[row]} / L{local}", fontsize=9)
        figure.colorbar(artist, ax=axes.flatten().tolist(), shrink=0.7, label=caption)
        figure.suptitle(caption + "; gray/transparent patches have no valid anchor correspondence")
        figure.savefig(output / filename, dpi=150)
        plt.close(figure)
