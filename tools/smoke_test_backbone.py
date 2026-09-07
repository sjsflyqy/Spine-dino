from __future__ import annotations

import argparse

import torch

from backbone import build_backbone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/rad-dino")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    name = (
        "rad_dino_maira2"
        if "maira" in args.model.lower()
        else "rad_dino"
    )
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = build_backbone(
        name,
        args.model,
        freeze=True,
        local_files_only=args.local_files_only,
    ).to(device)
    image = torch.randn(1, 3, args.size, args.size, device=device)
    with torch.inference_mode():
        output = model(image)
    print(
        {
            "cls": tuple(output.cls_token.shape),
            "patches": tuple(output.patch_tokens.shape),
            "map": tuple(output.feature_map.shape),
            "grid": output.grid_size,
        }
    )


if __name__ == "__main__":
    main()
