"""Run numerical and geometry checks for one visualization input."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT))
    from visualization.dino_spine_maps.map_extractor import extract_layer_maps
    from visualization.dino_spine_maps.model_loader import load_dino_model
    from visualization.dino_spine_maps.preprocessing import prepare_image
else:
    from .map_extractor import extract_layer_maps
    from .model_loader import load_dino_model
    from .preprocessing import prepare_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate DINO map extraction on one image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("auto", "dinov2", "dinov3"), default="auto"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--long-side", type=int, default=896)
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    loaded = load_dino_model(args.weights, architecture=args.architecture, device=device)
    prepared = prepare_image(
        args.image, patch_size=loaded.patch_size, long_side=args.long_side
    )
    batch = prepared.tensor.unsqueeze(0).to(device)
    result = extract_layer_maps(loaded, batch)

    assert len(result.layers) == loaded.num_layers
    assert result.grid_size == (
        prepared.geometry.grid_height,
        prepared.geometry.grid_width,
    )
    for layer in result.layers:
        assert layer.attention_heads.shape == (
            loaded.num_heads,
            *result.grid_size,
        )
        assert np.isfinite(layer.attention_heads).all()
        assert np.isfinite(layer.cosine).all()
        assert float(layer.cosine.min()) >= -1.0001
        assert float(layer.cosine.max()) <= 1.0001
        assert (layer.attention_patch_mass >= 0).all()
        assert (layer.attention_patch_mass <= 1.0001).all()
        restored = prepared.geometry.grid_to_original(layer.cosine)
        assert restored.shape == (
            prepared.geometry.original_height,
            prepared.geometry.original_width,
        )

    standard = loaded.model.forward_features(batch)
    standard_cls = standard["x_norm_clstoken"].detach().cpu().float()
    standard_patches = standard["x_norm_patchtokens"].detach().cpu().float()
    extracted_cls = result.final_cls_token.float()
    extracted_patches = result.final_patch_tokens.float()
    cls_error = float((standard_cls - extracted_cls).abs().max())
    patch_error = float((standard_patches - extracted_patches).abs().max())
    tolerance = 2e-4
    if cls_error > tolerance or patch_error > tolerance:
        raise AssertionError(
            f"Manual block traversal differs from forward_features: "
            f"cls max error={cls_error:.6g}, patch max error={patch_error:.6g}"
        )

    print(
        "Validation passed:\n"
        f"  architecture: {loaded.architecture}\n"
        f"  layers/heads: {loaded.num_layers}/{loaded.num_heads}\n"
        f"  patch grid: {result.grid_size}\n"
        f"  valid patches: {prepared.valid_patch_mask.sum()}/{prepared.valid_patch_mask.size}\n"
        f"  final CLS max error: {cls_error:.3g}\n"
        f"  final patch max error: {patch_error:.3g}"
    )


if __name__ == "__main__":
    main()
