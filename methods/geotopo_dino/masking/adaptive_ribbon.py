"""Width adaptation on an existing path, without anatomical labels.

The centerline and longitudinal span are held fixed to isolate width effects.
Widths describe the transverse support of S, not vertebral boundaries or count.
The calling training policy repairs supplements to preserve the total budget.
This function performs no candidate reranking or core-area repair.
"""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AdaptiveRibbon:
    core: torch.Tensor
    context: torch.Tensor
    widths: torch.Tensor
    raw_widths: torch.Tensor
    reliable: torch.Tensor
    prominence: torch.Tensor
    metrics: dict


def _nearest(value, choices):
    return min(choices, key=lambda width: (abs(width - value), width))


def _smooth_widths(raw, allowed, choices, penalty):
    """Choose widths jointly: fit evidence, penalize changes, step at most two."""
    values = torch.tensor(choices, dtype=torch.float32)
    transitions = penalty * (values[:, None] - values[None, :]).abs()
    transitions[(values[:, None] - values[None, :]).abs() > 2] = torch.inf
    costs = (raw[:, None].float() - values[None, :]).abs()
    costs = costs.masked_fill(~allowed, torch.inf)
    previous = costs[0]
    parents = []
    for row in range(1, len(raw)):
        best, parent = (previous[:, None] + transitions).min(0)
        previous = best + costs[row]
        parents.append(parent)
    if not torch.isfinite(previous).any():
        raise ValueError("no feasible smooth width profile")
    state = int(previous.argmin())
    states = [state]
    for parent in reversed(parents):
        state = int(parent[state])
        states.append(state)
    return values[torch.tensor(list(reversed(states)))].long()


@torch.no_grad()
def adapt_ribbon_width(score, valid, path, *, path_begin, span_start, span_stop,
                       axis="vertical", mode="local", min_width=3, max_width=9,
                       fallback_width=5, relative_threshold=0.5,
                       min_prominence=0.08, smooth_penalty=0.75, context_rows=2):
    """Estimate transverse response width, then draw a ribbon on a fixed path.

    For each row, local background is the median of valid flank samples outside
    the maximum band. Use a row's median if flanks are unavailable. The support
    containing the path center must exceed background + t * center prominence.
    Its contiguous extent determines an odd width. Weak/flat rows use a bounded
    fallback. Image mode uses the median reliable width in the masked span;
    local mode smooths widths with DP, allowing changes of at most two patches.
    Both modes fit inside valid tokens; unused full-path rows do not constrain
    an image-level estimate. Widths are estimated for the active span and ends.
    """
    if score.ndim != 2 or score.shape != valid.shape or score.device != valid.device:
        raise ValueError("expected matching [H,W] score and valid on the same device")
    if axis not in {"vertical", "horizontal"} or mode not in {"image", "local"}:
        raise ValueError("invalid axis or width mode")
    if any(not isinstance(w, int) or w < 1 or w % 2 != 1 for w in (min_width, max_width, fallback_width)):
        raise ValueError("width bounds and fallback must be positive odd integers")
    if not min_width <= fallback_width <= max_width or context_rows < 1:
        raise ValueError("invalid width range or context rows")
    if not 0 < relative_threshold < 1 or not math.isfinite(min_prominence) or min_prominence <= 0:
        raise ValueError("invalid response thresholds")
    if not math.isfinite(smooth_penalty) or smooth_penalty < 0:
        raise ValueError("invalid width smoothing penalty")
    device = score.device
    score = score.detach().float().cpu().contiguous()
    valid = valid.detach().bool().cpu().contiguous()
    path = torch.as_tensor(path, dtype=torch.long).detach().cpu().flatten()
    if not torch.isfinite(score[valid]).all() or (score[valid] < 0).any() or (score[valid] > 1 + 1e-6).any():
        raise ValueError("valid scores must be finite and in [0,1]")
    score = torch.where(valid, score, 0)
    if axis == "horizontal":
        score, valid = score.T.contiguous(), valid.T.contiguous()
    start, stop = span_start - context_rows, span_stop + context_rows
    if not (0 <= path_begin <= start < span_start < span_stop < stop <= path_begin + len(path) <= score.shape[0]):
        raise ValueError("span and visible ends must lie inside the cached path")
    active_path = path[start - path_begin:stop - path_begin]
    if (active_path < 0).any() or (active_path >= score.shape[1]).any():
        raise ValueError("path exceeds the transverse grid")
    if len(active_path) > 1 and (active_path[1:] - active_path[:-1]).abs().max() > 1:
        raise ValueError("expected a continuous path with at most one-patch steps")
    choices = list(range(min_width, max_width + 1, 2))
    allowed = torch.zeros(len(active_path), len(choices), dtype=torch.bool)
    raw, reliable, prominence = [], [], []
    half = max_width // 2
    for index, center in enumerate(active_path.tolist()):
        row = start + index
        for state, width in enumerate(choices):
            left, right = center - width // 2, center + width // 2 + 1
            allowed[index, state] = left >= 0 and right <= score.shape[1] and bool(valid[row, left:right].all())
        feasible = [width for width, ok in zip(choices, allowed[index].tolist()) if ok]
        if not feasible:
            raise ValueError("path cannot support the minimum width inside valid tokens")
        left_flank = list(range(max(0, center - half - 3), max(0, center - half)))
        right_flank = list(range(min(score.shape[1], center + half + 1), min(score.shape[1], center + half + 4)))
        flank = [column for column in left_flank + right_flank if valid[row, column]]
        background = float(score[row, flank].median()) if flank else float(score[row, valid[row]].median())
        center_score = float(score[row, center])
        contrast = center_score - background
        good = contrast >= min_prominence
        width = _nearest(fallback_width, feasible)
        if good:
            threshold = background + relative_threshold * contrast
            left = right = center
            while left > max(0, center - half) and valid[row, left - 1] and score[row, left - 1] >= threshold:
                left -= 1
            while right < min(score.shape[1] - 1, center + half) and valid[row, right + 1] and score[row, right + 1] >= threshold:
                right += 1
            width = _nearest(right - left + 1, feasible)
        raw.append(width)
        reliable.append(good)
        prominence.append(contrast)
    raw = torch.tensor(raw, dtype=torch.long)
    reliable = torch.tensor(reliable, dtype=torch.bool)
    prominence = torch.tensor(prominence, dtype=torch.float32)
    masked_slice = slice(context_rows, len(raw) - context_rows)
    if mode == "image":
        evidence = raw[masked_slice][reliable[masked_slice]]
        typical = float(evidence.float().median()) if evidence.numel() else fallback_width
        common = [width for width, ok in zip(choices, allowed.all(0).tolist()) if ok]
        if not common:
            raise ValueError("no common image-level width fits the active path")
        widths = torch.full_like(raw, _nearest(typical, common))
    else:
        local_allowed = allowed.clone()
        # Weak evidence must not spread a wide neighboring response into this row.
        for index in (~reliable).nonzero().flatten().tolist():
            local_allowed[index] &= torch.tensor(choices) == raw[index]
        widths = _smooth_widths(raw, local_allowed, choices, smooth_penalty)
    core, context = torch.zeros_like(valid), torch.zeros_like(valid)
    for row, center, width in zip(range(start, stop), active_path.tolist(), widths.tolist()):
        target = core if span_start <= row < span_stop else context
        target[row, center - width // 2:center + width // 2 + 1] = True
    core_widths = widths[masked_slice]
    metrics = {
        "mode": mode, "active_start_stop": [start, stop], "masked_start_stop": [span_start, span_stop],
        "width_min": int(core_widths.min()), "width_max": int(core_widths.max()),
        "width_mean": float(core_widths.float().mean()), "width_values": sorted(set(core_widths.tolist())),
        "reliable_rows": int(reliable[masked_slice].sum()), "masked_rows": span_stop - span_start,
        "weak_response_rows": int((~reliable[masked_slice]).sum()),
        "max_width_bound_rows": int((core_widths == max_width).sum()),
        "core_count": int(core.sum()), "context_count": int(context.sum()),
        "core_structure_mean": float(score[core].mean()),
        "effective_mask_ratio": int(core.sum()) / max(int(valid.sum()), 1),
        "width_is_anatomy_estimate": False,
    }
    if axis == "horizontal":
        core, context = core.T.contiguous(), context.T.contiguous()
    return AdaptiveRibbon(core.to(device), context.to(device), widths.to(device), raw.to(device),
                          reliable.to(device), prominence.to(device), metrics)
