"""Rasterize the fixed SpineFM CSXA split once, with auditable conversions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common import LEVELS, TASK, read_splits, resolve, save_json, sha256

VERSION = "csxa-corners-v1"
POSITIONS = ("top left", "top right", "bottom right", "bottom left")


def cross(a, b, c):
    u, v = np.asarray(b)-a, np.asarray(c)-a
    return float(u[0]*v[1] - u[1]*v[0])


def convex_hull(points):
    pts = sorted(set(map(tuple, points)))
    lower, upper = [], []
    for p in pts:
        while len(lower) > 1 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) > 1 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def annotation_to_masks(annotation, size, max_click_spread=2.0):
    width, height = size
    if size != (annotation["imageWidth"], annotation["imageHeight"]):
        raise ValueError("Image/annotation size mismatch")
    points, audit = {}, []
    for shape in annotation["shapes"]:
        name = shape["label"].strip()
        if name in points:
            raise ValueError(f"Duplicate corner label: {name}")
        coordinates = np.asarray(shape["points"], dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not len(coordinates) or not np.isfinite(coordinates).all():
            raise ValueError(f"Invalid point: {name}")
        if (coordinates < 0).any() or (coordinates[:, 0] >= width).any() or (coordinates[:, 1] >= height).any():
            raise ValueError(f"Out of bounds: {name}")
        spread = float(np.linalg.norm(coordinates[:, None]-coordinates[None, :], axis=-1).max())
        if spread > max_click_spread:
            raise ValueError(f"Ambiguous multi-click corner: {name}, spread={spread}")
        points[name] = coordinates.mean(axis=0)
        if len(coordinates) > 1:
            audit.append(dict(action="mean_repeated_clicks", label=name, original=coordinates.tolist(),
                              result=points[name].tolist(), spread_pixels=spread))
    masks, corners, centers, boxes = [], [], [], []
    for level in LEVELS:
        polygon = np.asarray([points[f"{level} {position}"] for position in POSITIONS])
        hull = convex_hull(polygon)
        if len(hull) != 4:
            raise ValueError(f"Four distinct convex vertices required: {level}")
        turns = [cross(polygon[i], polygon[(i+1) % 4], polygon[(i+2) % 4]) for i in range(4)]
        if not (all(v > 0 for v in turns) or all(v < 0 for v in turns)):
            audit.append(dict(action="reorder_vertices_only", level=level,
                              original=polygon.tolist(), result=hull.tolist()))
            polygon = hull
        raster = Image.new("L", size, 0)
        vertices = [tuple(map(int, p)) for p in np.rint(polygon)]
        ImageDraw.Draw(raster).polygon(vertices, fill=1)
        mask = np.asarray(raster, dtype=np.uint8)
        y, x = np.nonzero(mask)
        if not len(x):
            raise ValueError(f"Empty mask: {level}")
        masks.append(mask)
        corners.append(polygon)
        centers.append((x.mean(), y.mean()))
        boxes.append((x.min(), y.min(), x.max()+1, y.max()+1))
    return dict(masks=np.stack(masks), corners=np.asarray(corners, np.float32),
                centroids=np.asarray(centers, np.float32), boxes=np.asarray(boxes, np.float32),
                levels=np.asarray(LEVELS), size=np.asarray(size, np.int32)), audit


def make_overlay(image, payload, path):
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    colors = ("red", "lime", "cyan", "yellow", "magenta")
    for level, points, center, color in zip(LEVELS, payload["corners"], payload["centroids"], colors):
        vertices = [tuple(map(float, p)) for p in points]
        draw.line(vertices + vertices[:1], fill=color, width=3)
        draw.text(tuple(map(float, center)), level, fill=color)
    overlay.convert("RGB").save(path)


def prepare(data_root, output=None):
    data_root = resolve(data_root)
    output = resolve(output) if output else data_root / "masks_gt"
    splits = read_splits()
    ids = [i for group in splits.values() for i in group]
    image_ids = {p.stem for p in (data_root / "datasets").glob("*.png")}
    json_ids = {p.stem for p in (data_root / "dataset_json").glob("*.json")}
    if image_ids != set(ids) or json_ids != set(ids):
        raise ValueError("Images and JSONs must match the 600 official split IDs exactly")
    signature = {"version": VERSION, "rounding": "numpy.rint ties-to-even; PIL polygon inclusive boundary",
                 "max_click_spread_pixels": 2.0, "splits": splits,
                 "split_hashes": {s: sha256(TASK / "splits" / f"{s}.txt") for s in splits},
                 "sources": {i: {"image": sha256(data_root / "datasets" / f"{i}.png"),
                                  "json": sha256(data_root / "dataset_json" / f"{i}.json")} for i in ids}}
    manifest_path = output / "manifest.json"
    if output.exists():
        if not manifest_path.exists():
            raise ValueError(f"Incomplete/existing output; choose a new --output directory: {output}")
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != signature:
            raise ValueError("Source/recipe changed; use a new --output directory to preserve the old GT version")
        for i, row in manifest["samples"].items():
            if sha256(output / row["file"]) != row["sha256"]:
                raise ValueError(f"Cached mask corrupted: {i}")
        print(f"Verified existing GT cache: {len(ids)} images, 3000 instances; no rasterization")
        return manifest
    output.mkdir(parents=True)
    previews = output / "audit_overlays"
    previews.mkdir()
    samples, audits = {}, {}
    for split, split_ids in splits.items():
        for i in split_ids:
            image_path = data_root / "datasets" / f"{i}.png"
            annotation = json.loads((data_root / "dataset_json" / f"{i}.json").read_bytes())
            with Image.open(image_path) as image:
                payload, audit = annotation_to_masks(annotation, image.size)
                if audit or i == split_ids[0]:
                    make_overlay(image, payload, previews / f"{i}.jpg")
            target = output / f"{i}.npz"
            np.savez_compressed(target, **payload)
            samples[i] = dict(file=target.name, sha256=sha256(target), split=split)
            if audit:
                audits[i] = audit
    manifest = dict(signature=signature, samples=samples, images=len(ids), instances=len(ids)*5)
    save_json(output / "annotation_audit.json", audits)
    save_json(manifest_path, manifest)
    print(f"Generated {len(ids)} caches / {len(ids)*5} instances; {len(audits)} audited images in {output}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="Downstream/data/task10")
    parser.add_argument("--output")
    args = parser.parse_args()
    prepare(args.data_root, args.output)
