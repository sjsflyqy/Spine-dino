"""Center/offset targets and parameter-free decoding for tasks 07 and 08."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _convex_polygon_mask(
    corners: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    # Input order is TL, TR, BL, BR; polygon order must be TL, TR, BR, BL.
    polygon = corners[[0, 1, 3, 2]]
    ys = torch.arange(height, device=corners.device, dtype=corners.dtype)
    xs = torch.arange(width, device=corners.device, dtype=corners.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    signs = []
    for index in range(4):
        start = polygon[index]
        end = polygon[(index + 1) % 4]
        signs.append(
            (end[0] - start[0]) * (grid_y - start[1])
            - (end[1] - start[1]) * (grid_x - start[0])
        )
    stacked = torch.stack(signs)
    return ((stacked >= 0).all(dim=0) | (stacked <= 0).all(dim=0)).to(corners.dtype)


@torch.no_grad()
def build_structured_targets(
    points: torch.Tensor,
    output_size: tuple[int, int],
    *,
    task: str,
    heatmap_sigma: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Build one-center-map targets from normalized semantic landmark points."""
    batch, landmark_count, _ = points.shape
    height, width = output_size
    if task == "task07":
        object_count, full_objects = 17, 17
        if landmark_count != 68:
            raise ValueError(f"Task 07 expects 68 landmarks, got {landmark_count}")
    elif task == "task08":
        object_count, full_objects = 6, 5
        if landmark_count != 22:
            raise ValueError(f"Task 08 expects 22 landmarks, got {landmark_count}")
    else:
        raise ValueError(f"Unsupported structured landmark task: {task}")

    dtype, device = points.dtype, points.device
    heatmap = torch.zeros(batch, 1, height, width, dtype=dtype, device=device)
    indices = torch.zeros(batch, object_count, dtype=torch.long, device=device)
    center_offsets = torch.zeros(batch, object_count, 2, dtype=dtype, device=device)
    corner_offsets = torch.zeros(batch, object_count, 8, dtype=dtype, device=device)
    corner_mask = torch.zeros_like(corner_offsets)
    object_mask = torch.ones(batch, object_count, dtype=torch.bool, device=device)
    segmentation = torch.zeros(batch, 1, height, width, dtype=dtype, device=device)
    y_grid = torch.arange(height, device=device, dtype=dtype).view(height, 1)
    x_grid = torch.arange(width, device=device, dtype=dtype).view(1, width)

    for batch_index in range(batch):
        for object_index in range(object_count):
            if object_index < full_objects:
                object_points = points[
                    batch_index, 4 * object_index : 4 * object_index + 4
                ]
            else:
                object_points = points[batch_index, 20:22]
            scaled = object_points * points.new_tensor((width - 1, height - 1))
            center = scaled.mean(dim=0)
            center_int = center.floor().long()
            center_int[0].clamp_(0, width - 1)
            center_int[1].clamp_(0, height - 1)
            indices[batch_index, object_index] = center_int[1] * width + center_int[0]
            center_offsets[batch_index, object_index] = center - center_int.to(dtype)
            gaussian = torch.exp(
                -((x_grid - center_int[0]) ** 2 + (y_grid - center_int[1]) ** 2)
                / (2.0 * heatmap_sigma * heatmap_sigma)
            )
            heatmap[batch_index, 0] = torch.maximum(
                heatmap[batch_index, 0], gaussian
            )
            flattened = (center.unsqueeze(0) - scaled).flatten()
            corner_offsets[batch_index, object_index, : flattened.numel()] = flattened
            corner_mask[batch_index, object_index, : flattened.numel()] = 1
            if object_index < full_objects:
                segmentation[batch_index, 0] = torch.maximum(
                    segmentation[batch_index, 0],
                    _convex_polygon_mask(scaled, height, width),
                )

    return {
        "hm": heatmap,
        "indices": indices,
        "object_mask": object_mask,
        "reg": center_offsets,
        "corner_offsets": corner_offsets,
        "corner_mask": corner_mask,
        "segmentation": segmentation,
    }


def _gather_map(feature: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    flattened = feature.flatten(2).transpose(1, 2)
    return flattened.gather(
        1, indices.unsqueeze(-1).expand(-1, -1, flattened.shape[-1])
    )


def decode_structured_landmarks(
    output: dict[str, torch.Tensor], *, object_count: int, task: str
) -> torch.Tensor:
    """Decode and sort centers top-to-bottom, returning normalized landmarks."""
    heatmap = output["hm"].sigmoid()
    pooled = F.max_pool2d(heatmap, 3, stride=1, padding=1)
    heatmap = heatmap * pooled.eq(heatmap)
    batch, _, height, width = heatmap.shape
    _, indices = torch.topk(heatmap.flatten(1), object_count, dim=1)
    ys = (indices // width).to(heatmap.dtype)
    xs = (indices % width).to(heatmap.dtype)
    reg = _gather_map(output["reg"], indices)
    centers = torch.stack((xs, ys), dim=-1) + reg
    offsets = _gather_map(output["corner_offsets"], indices).reshape(
        batch, object_count, 4, 2
    )
    corners = centers.unsqueeze(2) - offsets
    order = centers[..., 1].argsort(dim=1)
    corners = corners.gather(
        1, order[:, :, None, None].expand(-1, -1, 4, 2)
    )
    corners[..., 0] /= max(width - 1, 1)
    corners[..., 1] /= max(height - 1, 1)
    corners = corners.clamp(0.0, 1.0)
    if task == "task07":
        return corners.reshape(batch, 68, 2)
    if task == "task08":
        return torch.cat((corners[:, :5].reshape(batch, 20, 2), corners[:, 5, :2]), dim=1)
    raise ValueError(f"Unsupported structured landmark task: {task}")
