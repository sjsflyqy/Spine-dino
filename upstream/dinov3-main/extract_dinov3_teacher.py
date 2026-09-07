"""Extract a raw DINOv3 backbone from an SSL teacher evaluation checkpoint."""

import argparse
from pathlib import Path

import torch

from dinov3.hub.backbones import dinov3_vitb16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("teacher", checkpoint)
    prefix = "backbone."
    backbone = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not backbone:
        # Also accept an already extracted raw backbone.
        backbone = state

    model = dinov3_vitb16(pretrained=False)
    model.load_state_dict(backbone, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone, args.output)
    print(f"Strict validation succeeded: {len(backbone)} tensors")
    print(f"Saved downstream teacher backbone: {args.output}")


if __name__ == "__main__":
    main()
