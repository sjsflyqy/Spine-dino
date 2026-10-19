"""Draw task 07 saved predictions: green GT, red predictions, yellow error lines.

Uses only NumPy and Pillow. No backbone, checkpoint, torch, or GPU is needed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
VIEWS = ("ap", "lat")
CORNER_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_points(value, name):
    points = np.asarray(value, dtype=np.float64)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise ValueError(f"{name}: expected 68 finite normalized (x,y) points; got {points.shape}")
    if ((points < -1e-6) | (points > 1 + 1e-6)).any():
        raise ValueError(f"{name}: normalized coordinates must be in [0, 1]")
    return points


def load_predictions(path):
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Prediction JSON must be a nonempty list of task 07 records")
    indexed = {}
    for record in records:
        identifier = record.get("id") if isinstance(record, dict) else None
        if not isinstance(identifier, str) or not re.fullmatch(r"DR_\d+", identifier):
            raise ValueError(f"Expected a task 07 DR_<number> identifier; got {identifier!r}")
        if identifier in indexed:
            raise ValueError(f"Duplicate prediction ID: {identifier}")
        for view in VIEWS:
            for kind in ("pred", "gt"):
                key = f"{view}_{kind}_normalized"
                if key not in record:
                    raise ValueError(f"{identifier}: missing {key}")
                validate_points(record[key], f"{identifier}.{key}")
        indexed[identifier] = record
    return indexed


def select_records(indexed, identifiers=None, limit=0):
    if limit < 0:
        raise ValueError("--limit must be nonnegative")
    identifiers = list(dict.fromkeys(identifiers)) if identifiers else list(indexed)
    missing = [identifier for identifier in identifiers if identifier not in indexed]
    if missing:
        raise ValueError(f"IDs not present in this model's saved predictions: {', '.join(missing)}")
    return [indexed[identifier] for identifier in (identifiers[:limit] if limit else identifiers)]


def find_image(data_root, view, identifier):
    directory = data_root / view / "images"
    matches = sorted(path for path in directory.glob(identifier + ".*")
                     if path.is_file() and path.suffix.lower() in SUFFIXES)
    if not matches:
        raise FileNotFoundError(f"No source image for {identifier}: {directory}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous source images for {identifier}: {matches}")
    return matches[0]


def check_annotation(path, size, saved_gt):
    """Match training's geometric corner ordering without importing its torch code."""
    if not path.is_file():
        return "not_available; GT taken from prediction JSON"
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotated_size = (payload["imageWidth"], payload["imageHeight"])
    if annotated_size != size:
        raise ValueError(f"Annotation/image dimensions disagree: {path}: {annotated_size} vs {size}")
    shapes = payload.get("shapes", [])
    expected_labels = Counter({str(i): 4 for i in range(1, 18)})
    if len(shapes) != 68 or Counter(str(shape.get("label", "")) for shape in shapes) != expected_labels:
        raise ValueError(f"Annotation must contain four corners for each of 17 vertebrae: {path}")
    ordered = []
    for label in range(1, 18):
        points = [tuple(map(float, shape["points"][0])) for shape in shapes if str(shape["label"]) == str(label)]
        by_y = sorted(points, key=lambda point: (point[1], point[0]))
        ordered.extend(sorted(by_y[:2], key=lambda point: point[0]) + sorted(by_y[2:], key=lambda point: point[0]))
    normalized = np.clip(np.asarray(ordered, dtype=np.float64) / np.asarray(size), 0, 1)
    if not np.isfinite(normalized).all() or not np.allclose(normalized, saved_gt, atol=1e-6, rtol=0):
        raise ValueError(f"Saved GT differs from current annotation or point ordering: {path}")
    return "dimensions_and_semantic_point_order_verified"


def original_coordinates(normalized, size):
    # The dataset stores x/imageWidth and y/imageHeight (not width-1/height-1).
    return np.asarray(normalized, dtype=np.float64) * np.asarray(size, dtype=np.float64)


def draw_overlay(source, gt, prediction, *, point_radius=4.0, line_width=1.5):
    """Draw on an original-resolution copy; clip only display coordinates at edges."""
    image = source.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    scale = image.width / 1000.0
    radius = max(1.0, point_radius * scale)
    width = max(1, round(line_width * scale))
    maximum = np.asarray((image.width - 1, image.height - 1))
    gt_display = np.clip(gt, 0, maximum)
    pred_display = np.clip(prediction, 0, maximum)
    # Correspondence is by semantic index, never nearest-neighbour matching.
    for target, predicted in zip(gt_display, pred_display):
        draw.line([tuple(target), tuple(predicted)], fill=YELLOW, width=width)
    for points, color in ((gt_display, GREEN), (pred_display, RED)):
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return image


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _fit_font(draw, text, width, size):
    font = _font(size)
    while size > 10 and draw.textbbox((0, 0), text, font=font)[2] > width:
        size -= 1
        font = _font(size)
    return font


def make_pair(overlays, metrics, *, model_name, identifier, panel_height=1600):
    """Make an AP/LAT overview with equal panel height and preserved aspect ratios."""
    resized = [overlays[view].resize((max(1, round(overlays[view].width * panel_height / overlays[view].height)),
                                     panel_height), Image.Resampling.LANCZOS) for view in VIEWS]
    gap, margin = 16, 16
    top, label_height, footer_height = 60, 44, 70
    canvas_width = max(620, sum(image.width for image in resized) + 2 * margin + gap)
    canvas = Image.new("RGB", (canvas_width, top + label_height + panel_height + footer_height), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    title = f"{model_name} | {identifier}"
    draw.text((margin, 12), title, fill="white", font=_fit_font(draw, title, canvas_width - 2 * margin, 28))
    x = (canvas_width - sum(image.width for image in resized) - gap) // 2
    for view, image in zip(VIEWS, resized):
        caption = f"{view.upper()} | mean error {metrics[view]['mean_error_px']:.2f} px"
        draw.text((x, top + 8), caption, fill="white", font=_fit_font(draw, caption, image.width, 22))
        canvas.paste(image, (x, top + label_height))
        x += image.width + gap
    legend_y = top + label_height + panel_height + 15
    x = margin
    for color, text in ((GREEN, "Ground truth"), (RED, "Prediction"), (YELLOW, "Corresponding-point error")):
        if color == YELLOW:
            draw.line((x, legend_y + 9, x + 18, legend_y + 9), fill=color, width=2)
        else:
            draw.ellipse((x + 5, legend_y + 4, x + 15, legend_y + 14), fill=color)
        font = _font(18)
        draw.text((x + 24, legend_y), text, fill="white", font=font)
        x += 40 + draw.textbbox((0, 0), text, font=font)[2]
    draw.text((margin, legend_y + 28), "Distances: original-image pixels; each view retains its aspect ratio.",
              fill=(190, 190, 190), font=_font(15))
    return canvas


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True, help="One model's directory containing test_predictions.json")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "Downstream/data/task07")
    parser.add_argument("--output-dir", type=Path, help="New directory for overlays and coordinate exports")
    parser.add_argument("--ids", nargs="+", help="DR IDs to show; default: all saved test predictions")
    parser.add_argument("--limit", type=int, default=0, help="Maximum selected cases; 0 means all")
    parser.add_argument("--list-ids", action="store_true", help="List available case IDs without rendering")
    parser.add_argument("--model-name", help="Display name in the paired overview")
    parser.add_argument("--point-radius", type=float, default=4.0, help="Dot radius in pixels at image width 1000")
    parser.add_argument("--line-width", type=float, default=1.5, help="Line width in pixels at image width 1000")
    parser.add_argument("--panel-height", type=int, default=1600, help="Height of each image in the AP/LAT overview")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not (np.isfinite(args.point_radius) and args.point_radius > 0
            and np.isfinite(args.line_width) and args.line_width > 0 and args.panel_height >= 100):
        parser.error("Dot radius and line width must be positive/finite; panel height must be >= 100")
    result_dir = args.result_dir.expanduser().resolve()
    prediction_path = result_dir / "test_predictions.json"
    indexed = load_predictions(prediction_path)
    selected = select_records(indexed, args.ids, args.limit)
    if args.list_ids:
        print("\n".join(record["id"] for record in selected))
        print(f"{len(selected)} cases", flush=True)
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless using --list-ids")
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    model_name = args.model_name or (result_dir.parent.name + "/" + result_dir.name
                                     if result_dir.name.startswith(("training_", "teacher_")) else result_dir.name)
    write_json(output / "manifest.json", {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "prediction_file": str(prediction_path),
        "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "data_root": str(data_root), "model_name": model_name, "selected_ids": [record["id"] for record in selected],
        "coordinate_mapping": "original xy = normalized xy * (image width, image height)",
        "orientation": "PIL stored orientation, unchanged from task07 dataset; no EXIF transpose",
        "point_order": {"vertebra_labels": list(range(1, 18)), "corners_per_vertebra": CORNER_ORDER},
        "colors": {"ground_truth": "green", "prediction": "red", "correspondence": "yellow"},
        "source": "saved task07 predictions; no model inference",
    })
    rows = []
    for record in selected:
        identifier = record["id"]
        destination = output / identifier
        destination.mkdir()
        overlays, view_metrics, coordinates = {}, {}, {}
        for view in VIEWS:
            path = find_image(data_root, view, identifier)
            # Match the training dataset exactly: do not auto-rotate by EXIF.
            with Image.open(path) as image:
                source = image.convert("RGB")
            gt_normalized = validate_points(record[f"{view}_gt_normalized"], f"{identifier}.{view}.GT")
            pred_normalized = validate_points(record[f"{view}_pred_normalized"], f"{identifier}.{view}.prediction")
            annotation_status = check_annotation(data_root / view / "labels" / f"{identifier}.json", source.size, gt_normalized)
            gt = original_coordinates(gt_normalized, source.size)
            predicted = original_coordinates(pred_normalized, source.size)
            error = np.linalg.norm(predicted - gt, axis=1)
            difference = pred_normalized - gt_normalized
            metrics = {
                "mean_error_px": float(error.mean()), "median_error_px": float(np.median(error)),
                "max_error_px": float(error.max()),
                "epsilon_dec_1024x512": float(np.linalg.norm(difference * (512, 1024), axis=1).mean()),
                "epsilon_MAE_normalized": float(np.abs(difference).sum(axis=1).mean()),
            }
            overlays[view] = draw_overlay(source, gt, predicted, point_radius=args.point_radius, line_width=args.line_width)
            overlays[view].save(destination / f"{view}.png")
            view_metrics[view] = metrics
            coordinates[view] = {
                "source_image": str(path), "size_wh": list(source.size), "annotation_check": annotation_status,
                "gt_normalized": gt_normalized.tolist(), "pred_normalized": pred_normalized.tolist(),
                "gt_original_xy": gt.tolist(), "pred_original_xy": predicted.tolist(),
                "error_px": error.tolist(), "metrics": metrics,
            }
            rows.append({"id": identifier, "view": view, "model_name": model_name, **metrics})
        make_pair(overlays, view_metrics, model_name=model_name, identifier=identifier,
                  panel_height=args.panel_height).save(destination / "pair.png")
        write_json(destination / "coordinates.json", {"id": identifier, "model_name": model_name, "views": coordinates})
        # Persist completed cases so a later missing image does not lose their report.
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {identifier}: AP {view_metrics['ap']['mean_error_px']:.2f} px, "
              f"LAT {view_metrics['lat']['mean_error_px']:.2f} px -> {destination}", flush=True)
    print(f"Finished {len(selected)} cases. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
