"""Offline-tested spine-span proposals; deliberately not wired into training.

The caller supplies a baseline block mask. Every returned comparison has the
same token budget. No backbone, dataset, global RNG, or training state is used.
All tensors stay on the input device; localization is an uncalibrated heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpanConfig:
    span_fraction: float = 0.5
    band_mask_ratio: float = 0.5
    band_half_width_fraction: float = 0.12
    smooth_kernel: int = 3
    max_step: int = 2
    continuity_penalty: float = 0.15
    min_coverage: float = 0.25
    min_contrast: float = 1.4
    max_border_mass: float = 0.55
    row_threshold_fraction: float = 0.25
    context_rows: int = 2
    min_span_rows: int = 3

    def __post_init__(self):
        for name in ("span_fraction", "band_mask_ratio", "band_half_width_fraction", "min_coverage",
                     "max_border_mass", "row_threshold_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.smooth_kernel < 1 or self.smooth_kernel % 2 != 1:
            raise ValueError("smooth_kernel must be positive and odd")
        if self.max_step < 0 or self.context_rows < 1:
            raise ValueError("max_step must be nonnegative and context_rows >= 1")
        if self.min_span_rows < 1:
            raise ValueError("min_span_rows must be positive")
        if not math.isfinite(self.min_contrast) or self.min_contrast <= 1:
            raise ValueError("min_contrast must be finite and > 1")
        if not math.isfinite(self.continuity_penalty) or self.continuity_penalty < 0:
            raise ValueError("continuity_penalty must be finite and nonnegative")


@dataclass
class SpanComparison:
    response: torch.Tensor
    centerline: torch.Tensor  # [H], column index; -1 outside supported extent
    band: torch.Tensor
    span: torch.Tensor
    background: torch.Tensor
    supplement: torch.Tensor
    context: torch.Tensor
    masks: dict[str, torch.Tensor]
    diagnostics: dict


@torch.no_grad()
def fuse_attention(heads: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Equal-weight fusion after per-head normalization over valid image keys."""
    if heads.ndim != 3 or heads.shape[1:] != valid.shape or heads.shape[0] == 0:
        raise ValueError("Expected nonempty heads [K,H,W] and valid [H,W]")
    if heads.device != valid.device:
        raise ValueError("heads and valid must be on the same device")
    if not torch.isfinite(heads).all() or (heads < 0).any():
        raise ValueError("Attention must be finite and nonnegative")
    values = heads.detach().float() * valid
    mass = values.sum(dim=(-2, -1), keepdim=True)
    usable = mass.flatten() > 1e-12
    if not usable.any():
        return values.new_zeros(valid.shape)
    return (values[usable] / mass[usable]).mean(0)


def _sample(pool: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    indices = pool.flatten().nonzero().flatten()
    if count < 0 or count > indices.numel():
        raise ValueError("Requested sample exceeds eligible mask area")
    chosen = indices[torch.randperm(indices.numel(), device=pool.device, generator=generator)[:count]]
    output = torch.zeros_like(pool)
    output.flatten()[chosen] = True
    return output


def _supplement(eligible, preferred, count, generator):
    # Retain available baseline blocks first, and fill only the remaining budget.
    preferred = eligible & preferred
    result = _sample(preferred, min(count, int(preferred.sum())), generator)
    return result | _sample(eligible & ~result, count - int(result.sum()), generator)


def _runs(active: torch.Tensor) -> list[tuple[int, int]]:
    result = []
    start = None
    for index, value in enumerate(active.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index))
            start = None
    return result


def _localize(response, valid, config):
    height, width = valid.shape
    empty = torch.zeros_like(valid)
    centerline = torch.full((height,), -1, dtype=torch.long, device=valid.device)
    metrics = {"coverage": 0.0, "contrast": 0.0, "border_mass": 0.0,
               "confidence_proxy": 0.0, "band_half_width": 0, "segments": []}
    if not valid.any():
        return response.new_zeros(response.shape), centerline, empty, metrics, ["no_valid_patches"]
    if not torch.isfinite(response[valid]).all():
        return response.new_zeros(response.shape), centerline, empty, metrics, ["nonfinite_response"]
    response = torch.where(valid, response.float().clamp(min=0), 0)
    if response.sum() <= 1e-12:
        return response, centerline, empty, metrics, ["empty_response"]
    response = response / response.sum()
    ys, xs = valid.nonzero(as_tuple=True)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    border = valid.clone()
    border[top + 1:bottom - 1, left + 1:right - 1] = False
    metrics["border_mass"] = float(response[border].sum())
    kernel = config.smooth_kernel
    weight = F.avg_pool2d(valid.float()[None, None], kernel, 1, kernel // 2)[0, 0]
    smoothed = F.avg_pool2d(response[None, None], kernel, 1, kernel // 2)[0, 0]
    smoothed = smoothed / weight.clamp(min=1e-8) * valid
    score = smoothed / smoothed[valid].mean().clamp(min=1e-8)

    # Local-transition DP: no absolute horizontal center prior. Invalid rows
    # cannot be bridged. The full path is only a proposal; trim unsupported ends.
    columns = torch.arange(width, device=valid.device)
    offsets = torch.arange(-config.max_step, config.max_step + 1, device=valid.device)
    previous_cols = columns[:, None] + offsets[None, :]
    possible = (previous_cols >= 0) & (previous_cols < width)
    previous_cols = previous_cols.clamp(0, width - 1)
    previous = score[top].masked_fill(~valid[top], -torch.inf)
    parents = []
    for row in range(top + 1, bottom):
        candidates = previous[previous_cols] - config.continuity_penalty * offsets.abs()
        candidates = candidates.masked_fill(~possible, -torch.inf)
        best, choice = candidates.max(-1)
        parents.append(previous_cols.gather(1, choice[:, None]).squeeze(1))
        previous = (score[row] + best).masked_fill(~valid[row], -torch.inf)
    if not torch.isfinite(previous).any():
        return smoothed, centerline, empty, metrics, ["no_continuous_valid_path"]
    col = int(previous.argmax())
    path = [col]
    for parent in reversed(parents):
        col = int(parent[col])
        path.append(col)
    path = torch.tensor(list(reversed(path)), device=valid.device)
    row_scores = score[torch.arange(top, bottom, device=valid.device), path]
    threshold = max(1.05, config.row_threshold_fraction * float(row_scores.max()))
    active = row_scores >= threshold
    # Close a single unsupported row, while leaving unsupported endpoints out.
    if active.numel() >= 3:
        active[1:-1] |= active[:-2].clone() & active[2:].clone()
    runs = _runs(active)
    if not runs:
        return smoothed, centerline, empty, metrics, ["no_supported_extent"]
    # Keep separate supported segments; never create a span across an unsupported gap.
    runs = [(begin, end) for begin, end in runs
            if end - begin >= 2 * config.context_rows + config.min_span_rows]
    if not runs:
        return smoothed, centerline, empty, metrics, ["no_segment_with_context"]
    for begin, end in runs:
        centerline[top + begin:top + end] = path[begin:end]
    metrics["segments"] = [[top + begin, top + end] for begin, end in runs]
    half_width = max(1, round((right - left) * config.band_half_width_fraction))
    band = ((columns[None, :] - centerline[:, None]).abs() <= half_width)
    band &= valid & (centerline[:, None] >= 0)
    metrics["band_half_width"] = half_width
    metrics["coverage"] = sum(end - begin for begin, end in runs) / (bottom - top)
    outside = valid & ~band
    contrast = (float(smoothed[band].mean() / smoothed[outside].mean().clamp(min=1e-8))
                if outside.any() else 0.0)
    metrics["contrast"] = contrast
    metrics["confidence_proxy"] = (
        metrics["coverage"] * min(1.0, max(0.0, (contrast - 1) / config.min_contrast))
        * (1 - metrics["border_mass"])
    )
    reasons = []
    if metrics["coverage"] < config.min_coverage:
        reasons.append("short_supported_extent")
    if contrast < config.min_contrast:
        reasons.append("low_band_contrast")
    if metrics["border_mass"] > config.max_border_mass:
        reasons.append("border_dominated_response")
    return smoothed, centerline, band, metrics, reasons


def _choose_span(band, budget, fraction, context_rows, generator, outside_capacity):
    rows = band.any(-1).nonzero().flatten()
    if not rows.numel():
        return None
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    maximum = min(max(1, round((bottom - top) * fraction)), bottom - top - 2 * context_rows)
    # Shrink by complete rows only. Never puncture the selected span to meet budget.
    for length in range(maximum, 0, -1):
        choices = []
        for start in range(top + context_rows, bottom - context_rows - length + 1):
            count = int(band[start:start + length].sum())
            if 0 < count <= budget and budget - count <= outside_capacity:
                choices.append(start)
        if choices:
            start = choices[int(torch.randint(len(choices), (), device=band.device, generator=generator))]
            span = torch.zeros_like(band)
            span[start:start + length] = band[start:start + length]
            return span
    return None


def _choose_topology(band, valid, budget, config, generator):
    desired = round(int(band.sum()) * config.band_mask_ratio)
    feasible_segments = []
    for top, bottom in _runs(band.any(-1)):
        maximum = min(max(config.min_span_rows, round((bottom - top) * config.span_fraction)),
                      bottom - top - 2 * config.context_rows)
        for length in range(maximum, config.min_span_rows - 1, -1):
            choices = []
            for start in range(top + config.context_rows, bottom - config.context_rows - length + 1):
                stop = start + length
                count = int(band[start:stop].sum())
                if not 0 < count <= min(budget, desired):
                    continue
                context_count = int(band[start - config.context_rows:start].sum()
                                    + band[stop:stop + config.context_rows].sum())
                lower = max(count, budget - int((valid & ~band).sum()))
                upper = min(budget, int(band.sum()) - context_count)
                if lower <= upper:
                    quota = min(max(desired, lower), upper)
                    choices.append((start, stop, quota))
            if choices:
                feasible_segments.append((top, bottom, choices))
                break
    if not feasible_segments:
        return None
    # Uniform over feasible segments, then over positions; long segments do not
    # silently crowd out shorter supported segments.
    pick = lambda n: int(torch.randint(n, (), device=band.device, generator=generator))
    top, bottom, choices = feasible_segments[pick(len(feasible_segments))]
    start, stop, quota = choices[pick(len(choices))]
    span, context = torch.zeros_like(band), torch.zeros_like(band)
    span[start:stop] = band[start:stop]
    context[start - config.context_rows:start] = band[start - config.context_rows:start]
    context[stop:stop + config.context_rows] = band[stop:stop + config.context_rows]
    return span, context, quota, {
        "selected_segment": [top, bottom], "selected_span": [start, stop],
        "span_selected_segment_fraction": (stop - start) / (bottom - top),
        "band_target_count": desired, "band_allocated_count": quota,
        "band_budget_adjusted": desired != quota,
    }


@torch.no_grad()
def generate_mask_comparison(
    response: torch.Tensor,
    valid_mask: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    config: SpanConfig | None = None,
    seed: int = 0,
) -> SpanComparison:
    """Return four equal-budget masks and localization diagnostics for one image.

    ``block_mask`` is the exact baseline/fallback supplied by the caller.
    ``band_random`` and ``topology_span`` share identical outside-band masks
    and identical inside-band counts. Both protect the same immediate context
    above/below the span. Other band patches can receive supplementary masks.
    Infeasible proposals fall back to the supplied baseline.
    """
    config = config or SpanConfig()
    if response.ndim != 2 or response.shape != valid_mask.shape or response.shape != block_mask.shape:
        raise ValueError("response, valid_mask and block_mask must share shape [H,W]")
    if len({response.device, valid_mask.device, block_mask.device}) != 1:
        raise ValueError("All inputs must be on the same device")
    valid, block = valid_mask.bool(), block_mask.bool()
    if (block & ~valid).any():
        raise ValueError("Baseline mask includes padding")
    generator = torch.Generator(device=response.device).manual_seed(seed)
    budget = int(block.sum())
    smooth, centerline, band, diagnostics, reasons = _localize(response.detach(), valid, config)
    masks = {name: block.clone() for name in ("block", "vertical_span", "band_random", "topology_span")}
    span, background, supplement, context = [torch.zeros_like(valid) for _ in range(4)]
    vertical_reason = None
    if budget:
        # Keep the original non-teacher control independent of topology tuning.
        vertical = _choose_span(valid, budget, 0.25, 1,
                                generator, int(valid.sum()))
        if vertical is None:
            vertical_reason = "no_full_row_span_with_context_within_budget"
        else:
            masks["vertical_span"] = vertical | _supplement(
                valid & ~vertical, block, budget - int(vertical.sum()), generator)
        if not reasons:
            proposal = _choose_topology(band, valid, budget, config, generator)
            if proposal is None:
                reasons.append("no_span_with_context_and_band_budget")
            else:
                span, context, band_quota, allocation = proposal
                diagnostics.update(allocation)
                inside = _supplement(band & ~context & ~span, block, band_quota - int(span.sum()), generator)
                background = _supplement(valid & ~band, block, budget - band_quota, generator)
                supplement = inside | background
                masks["topology_span"] = span | supplement
                masks["band_random"] = _sample(band & ~context, band_quota, generator) | background
                if any((masks[name] & context).any() for name in ("topology_span", "band_random")):
                    raise AssertionError("Reserved local context was masked")
    else:
        reasons.append("zero_mask_budget")
    diagnostics.update({
        "used_fallback": bool(reasons), "fallback_reasons": reasons,
        "vertical_fallback_reason": vertical_reason,
        "valid_count": int(valid.sum()), "target_count": budget,
        "band_count": int(band.sum()), "span_count": int(span.sum()),
        "span_rows": int(span.any(-1).sum()),
        "span_band_fraction": int(span.sum()) / max(int(band.sum()), 1),
        "span_total_mask_fraction": int(span.sum()) / max(budget, 1),
        "band_mask_ratio_requested": config.band_mask_ratio,
        "band_mask_ratio_actual": int((masks["topology_span"] & band).sum()) / max(int(band.sum()), 1),
        "context_count": int(context.sum()),
        "supplement_count": int(supplement.sum()),
        "masks": {},
    })
    for name, mask in masks.items():
        if int(mask.sum()) != budget or (mask & ~valid).any():
            raise AssertionError("Mask budget or padding invariant violated")
        diagnostics["masks"][name] = {
            "masked_count": int(mask.sum()),
            "valid_mask_ratio": int(mask.sum()) / max(int(valid.sum()), 1),
            "band_masked_count": int((mask & band).sum()),
        }
    return SpanComparison(smooth, centerline, band, span, background, supplement, context, masks, diagnostics)
