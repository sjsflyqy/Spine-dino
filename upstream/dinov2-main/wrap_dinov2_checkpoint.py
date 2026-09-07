"""Wrap an official DINOv2 backbone state dict for this repository's SSL loader."""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dict checkpoint, got {type(checkpoint).__name__}")

    # The official public DINOv2 backbone is a bare state_dict. Keep an already
    # wrapped checkpoint unchanged so running this helper twice is harmless.
    wrapped = checkpoint if "model" in checkpoint else {"model": checkpoint}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(wrapped, args.output)
    print(f"Saved: {args.output}")
    print(f"Model tensors: {len(wrapped['model'])}")


if __name__ == "__main__":
    main()
