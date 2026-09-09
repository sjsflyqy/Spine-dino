"""Extract normalized spatial maps from several ViT blocks without changing the old API."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _tokens_to_map(tokens: torch.Tensor, images: torch.Tensor, patch_size: int) -> torch.Tensor:
    grid_h = images.shape[-2] // patch_size
    grid_w = images.shape[-1] // patch_size
    expected = grid_h * grid_w
    # CLS and optional register/storage tokens precede the patch tokens.
    tokens = tokens[:, -expected:]
    if tokens.shape[1] != expected:
        raise RuntimeError(
            f"Cannot reshape {tokens.shape[1]} tokens to {grid_h}x{grid_w}"
        )
    return tokens.transpose(1, 2).reshape(
        images.shape[0], tokens.shape[-1], grid_h, grid_w
    )


def extract_intermediate_feature_maps(
    backbone,
    images: torch.Tensor,
    layers: Sequence[int] = (2, 5, 8, 11),
) -> list[torch.Tensor]:
    """Return zero-based block outputs as ``B,C,Htokens,Wtokens`` maps.

    Hugging Face DINO models expose hidden states, while the vendored native
    DINOv2/DINOv3 implementations expose ``get_intermediate_layers``.  Keeping
    this adapter outside ``backbone.registry`` leaves every existing linear
    probe unchanged.
    """
    indices = tuple(int(index) for index in layers)
    if not indices or tuple(sorted(set(indices))) != indices:
        raise ValueError(f"decoder layers must be unique and increasing: {indices}")

    model = backbone.model
    if hasattr(model, "get_intermediate_layers"):
        maps = model.get_intermediate_layers(images, n=indices, reshape=True)
        return list(maps)

    kwargs = {
        "pixel_values": images,
        "return_dict": True,
        "output_hidden_states": True,
    }
    try:
        output = model(interpolate_pos_encoding=True, **kwargs)
    except TypeError:
        output = model(**kwargs)
    hidden_states = output.hidden_states
    if hidden_states is None:
        raise RuntimeError("Backbone did not return intermediate hidden states")
    if max(indices) + 1 >= len(hidden_states):
        raise ValueError(
            f"Requested block {max(indices)}, but backbone returned "
            f"{len(hidden_states) - 1} blocks"
        )
    return [
        _tokens_to_map(hidden_states[index + 1], images, backbone.patch_size)
        for index in indices
    ]
