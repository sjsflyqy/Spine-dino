from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from keypoint_linear_common import build_keypoint_transform


VERTEBRAE = tuple(str(index) for index in range(1, 18))
CORNER_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")


def _find_image(root: Path, view: str, identifier: str) -> Path | None:
    for suffix in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        candidate = root / view / "images" / f"{identifier}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _read_clean_points(annotation_path: Path) -> torch.Tensor | None:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    shapes = payload.get("shapes", [])
    labels = [str(shape.get("label", "")) for shape in shapes]
    if len(shapes) != 68 or Counter(labels) != Counter({label: 4 for label in VERTEBRAE}):
        return None

    width = float(payload["imageWidth"])
    height = float(payload["imageHeight"])
    landmarks = []
    for vertebra in VERTEBRAE:
        points = [
            tuple(map(float, shape["points"][0]))
            for shape in shapes
            if str(shape["label"]) == vertebra
        ]
        # LabelMe only stores the vertebral identity. Canonicalize its four
        # corners geometrically so each output channel has fixed semantics.
        top, bottom = sorted(points, key=lambda point: (point[1], point[0]))[:2], sorted(
            points, key=lambda point: (point[1], point[0])
        )[2:]
        ordered = sorted(top, key=lambda point: point[0]) + sorted(
            bottom, key=lambda point: point[0]
        )
        landmarks.extend((x / width, y / height) for x, y in ordered)
    return torch.tensor(landmarks, dtype=torch.float32).clamp(0.0, 1.0)


def audit_pairs(root: str | Path):
    root = Path(root)
    clean: dict[str, dict[str, torch.Tensor]] = {}
    excluded: dict[str, list[str]] = {}
    ap_labels = {path.stem: path for path in (root / "ap/labels").glob("*.json")}
    lat_labels = {path.stem: path for path in (root / "lat/labels").glob("*.json")}
    for identifier in sorted(set(ap_labels) | set(lat_labels)):
        reasons = []
        if identifier not in ap_labels or identifier not in lat_labels:
            reasons.append("missing_paired_annotation")
        if reasons:
            excluded[identifier] = reasons
            continue
        ap_points = _read_clean_points(ap_labels[identifier])
        lat_points = _read_clean_points(lat_labels[identifier])
        if ap_points is None:
            reasons.append("invalid_ap_landmarks")
        if lat_points is None:
            reasons.append("invalid_lat_landmarks")
        ap_image = _find_image(root, "ap", identifier)
        lat_image = _find_image(root, "lat", identifier)
        if ap_image is None or lat_image is None:
            reasons.append("missing_paired_image")
        if reasons:
            excluded[identifier] = reasons
        else:
            clean[identifier] = {
                "ap": ap_points, "lat": lat_points,
                "ap_path": ap_image, "lat_path": lat_image,
            }
    return clean, excluded


class PairedWholeSpineDataset(Dataset):
    def __init__(
        self, root: str | Path, identifiers: list[str], point_cache,
        image_height: int, image_width: int, train: bool,
    ) -> None:
        self.root = Path(root)
        self.identifiers = identifiers
        self.point_cache = point_cache
        self.transform = build_keypoint_transform(image_height, image_width, train)

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, index):
        identifier = self.identifiers[index]
        ap = Image.open(self.point_cache[identifier]["ap_path"])
        lat = Image.open(self.point_cache[identifier]["lat_path"])
        return {
            "id": identifier,
            "ap_image": self.transform(ap),
            "lat_image": self.transform(lat),
            "ap_points": self.point_cache[identifier]["ap"],
            "lat_points": self.point_cache[identifier]["lat"],
        }
