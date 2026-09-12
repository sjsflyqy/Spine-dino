"""Save anchor/local crops and projected local patch centres for inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TVF

from methods.geotopo_dino.data.augmentations import DataAugmentationGeoTopoDINO
from methods.geotopo_dino.geometry.transforms import transform_points


MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def _to_pil(normalized: torch.Tensor) -> Image.Image:
    return TVF.to_pil_image((normalized.cpu() * STD + MEAN).clamp(0.0, 1.0))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to one source radiograph")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    import random

    random.seed(args.seed)
    source = Image.open(args.image).convert("RGB")
    augmentation = DataAugmentationGeoTopoDINO(
        (0.5, 1.0),
        (0.2, 0.5),
        8,
        global_crops_size=518,
        local_crops_size=196,
        patch_size=14,
    )
    output = augmentation(source)
    local_index = args.local_index
    anchor = _to_pil(output["global_crops"][0])
    local = _to_pil(output["local_crops"][local_index])
    overlay = anchor.copy()
    draw = ImageDraw.Draw(overlay)

    anchor_transform = output["anchor_geometry"]["transform"]
    local_transform = output["local_geometry"][local_index]["transform"]
    ys = (torch.arange(14, dtype=torch.float32) + 0.5) * 14
    xs = (torch.arange(14, dtype=torch.float32) + 0.5) * 14
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    local_points = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)
    source_points = transform_points(torch.linalg.inv(local_transform), local_points)
    anchor_points = transform_points(anchor_transform, source_points)
    for x, y in anchor_points.tolist():
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0))

    left, top, right, bottom = output["local_geometry"][local_index]["box_xyxy"].tolist()
    corners = torch.tensor([[left, top], [right, top], [right, bottom], [left, bottom]])
    projected_corners = transform_points(anchor_transform, corners).tolist()
    draw.line(projected_corners + [projected_corners[0]], fill=(0, 255, 0), width=3)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source.save(output_dir / "source.png")
    anchor.save(output_dir / "anchor.png")
    local.save(output_dir / f"local_{local_index}.png")
    overlay.save(output_dir / f"anchor_correspondence_{local_index}.png")
    print(f"Saved correspondence visualization to {output_dir}")


if __name__ == "__main__":
    main()

