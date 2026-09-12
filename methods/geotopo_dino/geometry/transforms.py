"""Homogeneous transforms used by GeoTopo-DINO.

Coordinates use a continuous pixel-edge frame: an image occupies
``[0, width] x [0, height]`` and integer pixel ``(x, y)`` has center
``(x + 0.5, y + 0.5)``.  Every matrix maps source coordinates to view
coordinates.  This convention matches ``grid_sample(align_corners=False)``
without an implicit half-pixel correction.
"""

from __future__ import annotations

from typing import Tuple

import torch


def identity_transform(*, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.eye(3, dtype=dtype)


def crop_resize_transform(
    *,
    top: int,
    left: int,
    height: int,
    width: int,
    output_size: Tuple[int, int],
) -> torch.Tensor:
    """Return the source-to-view transform for crop followed by resize."""
    output_height, output_width = output_size
    scale_x = float(output_width) / float(width)
    scale_y = float(output_height) / float(height)
    return torch.tensor(
        [
            [scale_x, 0.0, -scale_x * float(left)],
            [0.0, scale_y, -scale_y * float(top)],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def horizontal_flip_transform(width: int) -> torch.Tensor:
    """Return a transform that flips a view around its vertical centreline."""
    return torch.tensor(
        [[-1.0, 0.0, float(width)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def letterbox_transform(
    *,
    source_size: Tuple[int, int],
    resized_size: Tuple[int, int],
    padding_left: int,
    padding_top: int,
) -> torch.Tensor:
    """Return the source-to-canvas transform for resize followed by padding.

    Sizes are ``(height, width)``.  Separate x/y scales are intentional:
    integer rounding of the resized dimensions can make them differ slightly.
    """
    source_height, source_width = source_size
    resized_height, resized_width = resized_size
    scale_x = float(resized_width) / float(source_width)
    scale_y = float(resized_height) / float(source_height)
    return torch.tensor(
        [
            [scale_x, 0.0, float(padding_left)],
            [0.0, scale_y, float(padding_top)],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def transform_points(transform: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    """Apply one homogeneous transform to an arbitrary set of xy points."""
    ones = torch.ones_like(points_xy[..., :1])
    homogeneous = torch.cat((points_xy, ones), dim=-1)
    projected = homogeneous @ transform.transpose(-1, -2)
    return projected[..., :2] / projected[..., 2:].clamp_min(1e-12)

