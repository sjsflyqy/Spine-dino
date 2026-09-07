from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from backbone.preprocessing import build_transform


class VertebraCocoDataset(Dataset):
    """COCO images with GT regions and T1--S1 classification labels."""

    def __init__(self, annotation: str | Path, image_size: int = 518, train=False):
        self.annotation = Path(annotation)
        self.root = self.annotation.parent.parent
        payload = json.loads(self.annotation.read_text(encoding="utf-8"))
        self.images = payload["images"]
        self.annotations_by_image: dict[str | int, list[dict]] = {}
        for annotation_item in payload["annotations"]:
            self.annotations_by_image.setdefault(
                annotation_item["image_id"], []
            ).append(annotation_item)
        categories = sorted(payload["categories"], key=lambda item: item["id"])
        self.category_ids = [item["id"] for item in categories]
        self.category_names = [item["name"] for item in categories]
        self.category_to_label = {
            category_id: label for label, category_id in enumerate(self.category_ids)
        }
        self.transform = build_transform(image_size=image_size, train=train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        record = self.images[index]
        image_path = (self.root / record["file_name"]).resolve()
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        regions = []
        for annotation_item in self.annotations_by_image.get(record["id"], []):
            x, y, box_width, box_height = map(float, annotation_item["bbox"])
            regions.append(
                (
                    self.category_to_label[annotation_item["category_id"]],
                    [
                        x / width,
                        y / height,
                        (x + box_width) / width,
                        (y + box_height) / height,
                    ],
                )
            )
        # Anatomical order makes IRA diagnostics deterministic while the
        # shared classifier itself receives no slot or order identifier.
        regions.sort(key=lambda item: item[0])
        labels = torch.tensor([item[0] for item in regions], dtype=torch.long)
        boxes = torch.tensor([item[1] for item in regions], dtype=torch.float32)
        if not regions:
            boxes = torch.empty((0, 4), dtype=torch.float32)

        return {
            "pixel_values": self.transform(image),
            "boxes": boxes.clamp(0, 1),
            "labels": labels,
            "image_id": record["id"],
            "path": str(image_path),
        }


def collate_vertebra_batch(samples):
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
        "boxes": [sample["boxes"] for sample in samples],
        "labels": [sample["labels"] for sample in samples],
        "image_id": [sample["image_id"] for sample in samples],
        "path": [sample["path"] for sample in samples],
    }
