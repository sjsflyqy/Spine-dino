from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from keypoint_linear_common import build_keypoint_transform


LANDMARK_NAMES = tuple(
    [f"L{level}-{corner}" for level in range(1, 6) for corner in range(1, 5)]
    + ["S1-1", "S1-2"]
)


def audit_images(root: str | Path):
    root = Path(root)
    clean: dict[str, dict] = {}
    excluded: dict[str, list[str]] = {}
    expected = Counter(LANDMARK_NAMES)
    for annotation_path in sorted((root / "labels").glob("*.json")):
        identifier = annotation_path.stem
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        shapes = payload.get("shapes", [])
        labels = [str(shape.get("label", "")) for shape in shapes]
        reasons = []
        if len(shapes) != 22 or Counter(labels) != expected:
            reasons.append("invalid_landmark_labels")
        image_path = root / "images" / f"{identifier}.png"
        if not image_path.is_file():
            reasons.append("missing_image")
        if reasons:
            excluded[identifier] = reasons
            continue
        width = float(payload["imageWidth"])
        height = float(payload["imageHeight"])
        point_by_label = {
            str(shape["label"]): tuple(map(float, shape["points"][0]))
            for shape in shapes
        }
        points = torch.tensor(
            [
                (point_by_label[label][0] / width, point_by_label[label][1] / height)
                for label in LANDMARK_NAMES
            ],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        clean[identifier] = {"points": points, "width": width, "height": height}
    return clean, excluded


class LumbarLandmarkDataset(Dataset):
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
        image = Image.open(self.root / "images" / f"{identifier}.png")
        record = self.point_cache[identifier]
        return {
            "id": identifier,
            "image": self.transform(image),
            "points": record["points"],
            "original_size": torch.tensor(
                (record["width"], record["height"]), dtype=torch.float32
            ),
        }

