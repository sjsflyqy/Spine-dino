"""Collation for geometry-conditioned DINO training."""

from __future__ import annotations

import random

import torch

from dinov2.data.masking import MaskingGenerator


def _block_mask_within_valid(valid: torch.Tensor, target_count: int) -> torch.Tensor:
    """Generate DINO-style block masks inside a rectangular valid token area."""
    output = torch.zeros_like(valid, dtype=torch.bool)
    if target_count <= 0 or not valid.any():
        return output
    rows, cols = valid.nonzero(as_tuple=True)
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    height, width = bottom - top, right - left
    valid_count = int(valid.sum())
    target_count = min(int(target_count), valid_count)
    generator = MaskingGenerator(
        input_size=(height, width),
        max_num_patches=max(1, int(0.5 * height * width)),
    )
    submask = torch.as_tensor(generator(target_count), dtype=torch.bool)
    submask &= valid[top:bottom, left:right]
    output[top:bottom, left:right] = submask

    missing = target_count - int(output.sum())
    if missing > 0:
        available = (valid & ~output).flatten().nonzero().flatten().tolist()
        for index in random.sample(available, k=min(missing, len(available))):
            output.flatten()[index] = True
    return output


def _make_valid_aware_masks(
    *,
    batch_size: int,
    anchor_valid_masks: torch.Tensor,
    mask_ratio_tuple,
    mask_probability: float,
    n_tokens: int,
    mask_generator,
):
    n_global_views = 2
    total_views = n_global_views * batch_size
    n_samples_masked = int(total_views * mask_probability)
    ratio_edges = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1).tolist()
    selected_rows = random.sample(range(total_views), k=n_samples_masked)
    random.shuffle(selected_rows)
    masks = torch.zeros((total_views, n_tokens), dtype=torch.bool)
    upperbound = 0
    target_counts = torch.zeros(total_views, dtype=torch.long)

    grid_height, grid_width = anchor_valid_masks.shape[-2:]
    for interval_index, row_index in enumerate(selected_rows):
        ratio_min = ratio_edges[interval_index]
        ratio_max = ratio_edges[interval_index + 1]
        ratio = random.uniform(ratio_min, ratio_max)
        upperbound += int(n_tokens * ratio_max)
        if row_index < batch_size:
            valid = anchor_valid_masks[row_index]
            target = int(int(valid.sum()) * ratio)
            mask = _block_mask_within_valid(valid, target)
        else:
            target = int(n_tokens * ratio)
            mask = torch.as_tensor(mask_generator(target), dtype=torch.bool)
        masks[row_index] = mask.flatten()
        target_counts[row_index] = int(mask.sum())
    return masks, target_counts, upperbound


def collate_data_and_cast_gcvd(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
):
    """Collate images and geometry while retaining DINOv2 view-major order."""
    batch_size = len(samples_list)
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])
    if n_global_crops != 2:
        raise ValueError("GeoTopo-DINO requires [anchor, random_global]")

    global_crops = torch.stack(
        [sample[0]["global_crops"][view] for view in range(n_global_crops) for sample in samples_list]
    )
    local_crops = torch.stack(
        [sample[0]["local_crops"][view] for view in range(n_local_crops) for sample in samples_list]
    )
    anchor_transforms = torch.stack(
        [sample[0]["anchor_geometry"]["transform"] for sample in samples_list]
    )
    anchor_valid_masks = torch.stack(
        [sample[0]["anchor_geometry"]["valid_mask"] for sample in samples_list]
    )
    random_global_transforms = torch.stack(
        [sample[0]["random_global_geometry"]["transform"] for sample in samples_list]
    )
    source_sizes = torch.stack(
        [sample[0]["anchor_geometry"]["source_size"] for sample in samples_list]
    )
    local_transforms = torch.stack(
        [
            sample[0]["local_geometry"][view]["transform"]
            for view in range(n_local_crops)
            for sample in samples_list
        ]
    )
    local_boxes = torch.stack(
        [
            sample[0]["local_geometry"][view]["box_xyxy"]
            for view in range(n_local_crops)
            for sample in samples_list
        ]
    )
    local_sample_ids = torch.tensor(
        [sample_id for _view in range(n_local_crops) for sample_id in range(batch_size)],
        dtype=torch.long,
    )
    local_view_ids = torch.tensor(
        [view for view in range(n_local_crops) for _sample_id in range(batch_size)],
        dtype=torch.long,
    )
    local_flip_flags = torch.tensor(
        [
            sample[0]["local_geometry"][view]["flipped"]
            for view in range(n_local_crops)
            for sample in samples_list
        ],
        dtype=torch.bool,
    )
    global_sample_ids = torch.tensor(
        [sample_id for _view in range(n_global_crops) for sample_id in range(batch_size)],
        dtype=torch.long,
    )
    global_view_ids = torch.tensor(
        [view for view in range(n_global_crops) for _sample_id in range(batch_size)],
        dtype=torch.long,
    )
    if local_crops.shape[0] != local_transforms.shape[0] or local_crops.shape[0] != local_sample_ids.shape[0]:
        raise AssertionError("local image and geometry ordering diverged")

    masks, mask_target_counts, upperbound = _make_valid_aware_masks(
        batch_size=batch_size,
        anchor_valid_masks=anchor_valid_masks,
        mask_ratio_tuple=mask_ratio_tuple,
        mask_probability=mask_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
    )
    return {
        "collated_global_crops": global_crops.to(dtype),
        "collated_local_crops": local_crops.to(dtype),
        "collated_masks": masks,
        "mask_target_counts": mask_target_counts,
        "upperbound": upperbound,
        "anchor_transforms": anchor_transforms,
        "anchor_valid_masks": anchor_valid_masks,
        "source_sizes": source_sizes,
        "random_global_transforms": random_global_transforms,
        "global_sample_ids": global_sample_ids,
        "global_view_ids": global_view_ids,
        "local_transforms": local_transforms,
        "local_boxes_source": local_boxes,
        "local_sample_ids": local_sample_ids,
        "local_view_ids": local_view_ids,
        "local_flip_flags": local_flip_flags,
    }
