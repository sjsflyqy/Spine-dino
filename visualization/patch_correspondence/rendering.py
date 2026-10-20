"""Publication export from immutable cached cosine maps and actual match coordinates."""

from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


BLUE, ORANGE, QUERY = "#2389ff", "#ff9e24", "#ff28b0"


def encode_image(image):
    stream = BytesIO()
    image.save(stream, format="PNG")
    return np.frombuffer(stream.getvalue(), dtype=np.uint8)


def decode_image(array):
    with Image.open(BytesIO(array.tobytes())) as image:
        return image.convert("RGB")


def load_bundle(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
        metadata = json.loads(str(archive["metadata"]))
    a_transform = np.asarray(metadata["views"]["a"]["transform_source_to_view"])
    b_transform = np.asarray(metadata["views"]["b"]["transform_source_to_view"])
    source_point = np.linalg.inv(a_transform) @ np.r_[metadata["query"]["a_xy"], 1]
    source_point /= source_point[2]
    expected_b = b_transform @ source_point
    expected_b /= expected_b[2]
    if not np.allclose(source_point[:2], metadata["query"]["source_xy"], atol=0.02, rtol=0):
        raise ValueError("Cached source Query disagrees with crop transform")
    if not np.allclose(expected_b[:2], metadata["query"]["gt_b_xy"], atol=1e-3, rtol=0):
        raise ValueError("Cached geometric GT disagrees with crop transforms")
    for name in ("baseline", "ours"):
        heat, valid = arrays[f"{name}_heatmap"], arrays[f"{name}_valid_b"]
        if heat.shape != valid.shape or not np.isfinite(heat).all():
            raise ValueError(f"Invalid cached {name} heatmap")
        flat = np.where(valid, heat, -np.inf).argmax()
        side, patch = heat.shape[1], metadata["models"][name]["patch_size"]
        predicted = np.asarray(((flat % side + 0.5) * patch, (flat // side + 0.5) * patch))
        recorded = np.asarray(metadata["query"][f"{name}_pred_b_xy"])
        if not np.allclose(predicted, recorded, atol=1e-4, rtol=0):
            raise ValueError(f"Cached {name} prediction disagrees with cosine argmax")
        error = float(np.linalg.norm(recorded - metadata["query"]["gt_b_xy"]))
        if not np.isclose(error, metadata["query"][f"{name}_error_px"], atol=1e-3, rtol=1e-5):
            raise ValueError(f"Cached {name} error disagrees with geometric GT")
    return metadata, arrays


def _background(axis, image):
    axis.imshow(image, extent=(0, image.width, image.height, 0))
    axis.set_xlim(0, image.width)
    axis.set_ylim(image.height, 0)
    axis.axis("off")


def _original(axis, image, metadata):
    _background(axis, image)
    for name, color in (("a", BLUE), ("b", ORANGE)):
        left, top, right, bottom = metadata["views"][name]["box_xyxy"]
        axis.add_patch(Rectangle((left, top), right - left, bottom - top,
                                 fill=False, edgecolor=color, linewidth=1.1))
    axis.plot(*metadata["query"]["source_xy"], marker="+", color=QUERY, markersize=8, markeredgewidth=1.5)


def _view_a(axis, image, metadata):
    _background(axis, image)
    x, y = metadata["query"]["a_xy"]
    patch = metadata["models"]["baseline"]["patch_size"]
    axis.add_patch(Rectangle((x - patch / 2, y - patch / 2), patch, patch,
                             fill=False, edgecolor=QUERY, linewidth=1.2))


def _heat(axis, image, metadata, arrays, name, norm, cmap, alpha):
    _background(axis, image)
    heat = np.ma.masked_where(~arrays[f"{name}_valid_b"], arrays[f"{name}_heatmap"])
    artist = axis.imshow(heat, extent=(0, image.width, image.height, 0), interpolation="nearest",
                         cmap=cmap, norm=norm, alpha=alpha)
    gt = metadata["query"]["gt_b_xy"]
    pred = metadata["query"][f"{name}_pred_b_xy"]
    axis.plot([gt[0], pred[0]], [gt[1], pred[1]], color="#fff066", linewidth=0.7)
    axis.plot(*gt, marker="+", color="#00ff63", markersize=9, markeredgewidth=1.5)
    axis.plot(*pred, marker="o", color="#ff2020", markersize=4, markeredgecolor="white", markeredgewidth=0.5)
    return artist


def _save_formats(figure, path):
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(path.with_suffix("." + suffix), dpi=300, facecolor="white")


def export_bundle(path, output, *, vmin=-1, vmax=1, alpha=0.45, cmap="turbo"):
    if not (-1 <= vmin < vmax <= 1 and 0 <= alpha <= 0.75):
        raise ValueError("Require -1 <= vmin < vmax <= 1 and 0 <= alpha <= 0.75")
    metadata, arrays = load_bundle(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    images = {name: decode_image(arrays[f"{name}_png"]) for name in ("source", "a", "b")}
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    # Editable vector text/markers around raster X-ray and exact patch heatmap.
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7,
                         "svg.fonttype": "none", "pdf.fonttype": 42}):
        figure = plt.figure(figsize=(7.2, 3.9), layout="constrained")
        grid = figure.add_gridspec(3, 4, width_ratios=(0.8, 1.15, 1.4, 1.4), height_ratios=(1, 0.075, 0.25))
        axes = [figure.add_subplot(grid[0, column]) for column in range(4)]
        _original(axes[0], images["source"], metadata)
        _view_a(axes[1], images["a"], metadata)
        _heat(axes[2], images["b"], metadata, arrays, "baseline", norm, cmap, alpha)
        artist = _heat(axes[3], images["b"], metadata, arrays, "ours", norm, cmap, alpha)
        for axis, title in zip(axes, ("(a) Original", "View A / Query", "(b) Baseline: MAIRA2", "(c) Ours: GCVD")):
            axis.set_title(title, fontsize=7.5, pad=3)
        color_axis = figure.add_subplot(grid[1, 2:])
        figure.colorbar(artist, cax=color_axis, orientation="horizontal", label="Patch feature cosine similarity")
        legend_axis = figure.add_subplot(grid[1:, :2])
        legend_axis.axis("off")
        handles = [Line2D([], [], color=BLUE, label="View A crop"), Line2D([], [], color=ORANGE, label="View B crop"),
                   Line2D([], [], color="#00cf50", marker="+", linestyle="none", label="Geometric GT"),
                   Line2D([], [], color="red", marker="o", linestyle="none", markersize=3, label="Predicted match")]
        legend_axis.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.5, ncol=2,
                           handlelength=1.2, columnspacing=0.7)
        legend_axis.text(0, 0.02, "GT = same source position across crops.\nNo vertebral identity annotation is assumed.",
                         fontsize=6, transform=legend_axis.transAxes)
        quantitative = figure.add_subplot(grid[2, 2:])
        quantitative.axis("off")
        query = metadata["query"]
        baseline, ours = query["baseline_error_px"], query["ours_error_px"]
        quantitative.text(0, 0.78, "(d) Matching error in View B", fontsize=7, transform=quantitative.transAxes)
        quantitative.text(0, 0.40, f"Baseline: {baseline:.1f} px", fontsize=8, transform=quantitative.transAxes)
        quantitative.text(0.56, 0.40, f"Ours: {ours:.1f} px", fontsize=8, transform=quantitative.transAxes)
        quantitative.text(0, 0.02, f"Error reduction: {baseline - ours:.1f} px", fontsize=8,
                          transform=quantitative.transAxes)
        _save_formats(figure, output / "fig1")
        plt.close(figure)

        panels = output / "panels"
        panels.mkdir()
        for name, image in (("original", images["source"]), ("view_a", images["a"]),
                            ("baseline", images["b"]), ("ours", images["b"])):
            figure = plt.figure(figsize=(3 * image.width / image.height, 3))
            axis = figure.add_axes((0, 0, 1, 1))
            if name == "original":
                _original(axis, image, metadata)
            elif name == "view_a":
                _view_a(axis, image, metadata)
            else:
                _heat(axis, image, metadata, arrays, name, norm, cmap, alpha)
            _save_formats(figure, panels / name)
            plt.close(figure)
        for name, image in images.items():
            image.save(panels / f"{name}_input.png", dpi=(300, 300))
    exported = {**metadata, "rendering": {"vmin": vmin, "vmax": vmax, "alpha": alpha, "cmap": cmap,
                                          "dpi": 300, "cache_file": str(Path(path).resolve())}}
    (output / "case.json").write_text(json.dumps(exported, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return metadata
