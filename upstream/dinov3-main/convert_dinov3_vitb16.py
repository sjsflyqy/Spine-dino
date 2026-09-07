"""Validate and wrap an official DINOv3 ViT-B/16 backbone for SSL initialization."""

import argparse
from pathlib import Path

import torch

from dinov3.hub.backbones import dinov3_vitb16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.weights_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a state_dict, got {type(checkpoint).__name__}")
    if "teacher" in checkpoint:
        state_dict = checkpoint["teacher"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model = dinov3_vitb16(pretrained=False)
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {result}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "teacher": state_dict,
            "meta": {
                "architecture": "dinov3_vitb16",
                "patch_size": 16,
                "n_storage_tokens": 4,
                "source": str(args.weights_path.resolve()),
            },
        },
        args.output_path,
    )
    print(f"Strict validation succeeded: {len(state_dict)} tensors")
    print(f"Saved training initialization: {args.output_path}")


if __name__ == "__main__":
    main()
