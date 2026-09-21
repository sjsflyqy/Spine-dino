"""Render aligned targets and decoder predictions; also accepts a tensor dump.

python -m methods.geotopo_dino.tools.visualize_reconstruction \
    --input batch.pt --output reconstruction.png --patch-size 14

The dump must contain images (normalized BCHW), prediction (BNP), and masks (BN).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from dinov2.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ..losses.pixel_reconstruction_loss import patchify, unpatchify


@torch.no_grad()
def save_reconstruction_grid(
    images, prediction, masks, *, patch_size, output_path, norm_pix_loss=False, max_views=2
):
    selected = masks.any(dim=1).nonzero().flatten()[:max_views]
    if not selected.numel():
        return
    images = images[selected].float().cpu()
    prediction = prediction[selected].float().cpu()
    masks = masks[selected].bool().cpu()
    height, width = images.shape[-2:]
    grid_size = (height // patch_size, width // patch_size)
    target = patchify(images, patch_size)
    loss_target = target
    if norm_pix_loss:
        mean = target.mean(-1, keepdim=True)
        std = (target.var(-1, keepdim=True, unbiased=target.shape[-1] > 1) + 1e-6).sqrt()
        loss_target = (target - mean) / std
        display_prediction = prediction * std + mean
    else:
        display_prediction = prediction
    error = (prediction - loss_target).square().mean(-1)
    reconstructed = unpatchify(display_prediction, patch_size, grid_size)
    pixel_mask = masks.reshape(-1, *grid_size).repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2)
    mean = images.new_tensor(IMAGENET_DEFAULT_MEAN)[None, :, None, None]
    std = images.new_tensor(IMAGENET_DEFAULT_STD)[None, :, None, None]
    if images.shape[1] != 3:
        raise ValueError("visualization expects the current RGB-normalized training input")
    target_rgb = (images * std + mean).clamp(0, 1)
    prediction_rgb = (reconstructed * std + mean).clamp(0, 1)
    masked_rgb = target_rgb.masked_fill(pixel_mask[:, None], 0.5)
    composite = torch.where(pixel_mask[:, None], prediction_rgb, target_rgb)
    titles = ["Target", "Mask (gray)", "Decoder prediction", "Visible GT + masked prediction", "Masked MSE (relative)"]
    if norm_pix_loss:
        titles[2] = "Prediction + GT patch statistics"
    panel_width = max(width, 245)
    header = 32
    canvas = Image.new("RGB", (panel_width * 5, (height + header) * images.shape[0]), "white")
    draw = ImageDraw.Draw(canvas)
    for row in range(images.shape[0]):
        patch_error = error[row].masked_fill(~masks[row], 0)
        patch_error = patch_error / patch_error.max().clamp_min(1e-8)
        heat = patch_error.reshape(*grid_size).repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
        heat_rgb = torch.stack((heat, torch.zeros_like(heat), torch.zeros_like(heat)))
        for column, tensor in enumerate((target_rgb[row], masked_rgb[row], prediction_rgb[row], composite[row], heat_rgb)):
            tile = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
            x, y = column * panel_width, row * (height + header)
            draw.text((x + 4, y + 8), titles[column], fill="black")
            canvas.paste(tile, (x, y + header))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--norm-pix-loss", action="store_true")
    args = parser.parse_args()
    data = torch.load(args.input, map_location="cpu", weights_only=True)
    save_reconstruction_grid(
        data["images"], data["prediction"], data["masks"], patch_size=args.patch_size,
        output_path=args.output, norm_pix_loss=args.norm_pix_loss,
    )


if __name__ == "__main__":
    main()
