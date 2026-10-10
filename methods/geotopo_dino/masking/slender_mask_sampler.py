"""Width-limited curved strips, separate budget supplements, and visible ends.

Shared proposal generator. Guided paths read teacher structure scores; random
straight/curved paths are controls. Long-FOV axis is an explicit geometry prior,
not an anatomical label. The thin core never expands to absorb the total budget.
"""

import math
import random

import torch
import torch.nn.functional as F

from .candidate_mask_sampler import CandidateComparison, _components, _geometry, _proposal


def _runs(rows):
    runs, start = [], None
    for index, value in enumerate(rows.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    return runs


def _path(valid_centers, response, begin, end, kind, rng, max_step):
    width = valid_centers.shape[1]
    if kind == "random_straight":
        available = valid_centers[begin:end].all(0).nonzero().flatten().tolist()
        if not available:
            return None
        return torch.full((end - begin,), rng.choice(available), dtype=torch.long)
    if kind == "random_curve":
        current = rng.choice(valid_centers[begin].nonzero().flatten().tolist())
        result = [current]
        # A persistent gentle direction gives bends rather than pixel zigzags.
        direction = rng.choice((-1, 0, 1)) if max_step else 0
        for row in range(begin + 1, end):
            if rng.random() < 0.18:
                direction = rng.choice((-1, 0, 1)) if max_step else 0
            proposed = current + direction if rng.random() < 0.6 else current
            choices = [x for x in range(max(0, current - max_step), min(width, current + max_step + 1))
                       if valid_centers[row, x]]
            if not choices:
                return None
            current = min(choices, key=lambda x: abs(x - proposed))
            result.append(current)
        return torch.tensor(result, dtype=torch.long)
    columns = torch.arange(width)
    offsets = torch.arange(-max_step, max_step + 1)
    previous_columns = columns[:, None] + offsets[None, :]
    possible = (previous_columns >= 0) & (previous_columns < width)
    previous_columns = previous_columns.clamp(0, width - 1)
    previous = response[begin].masked_fill(~valid_centers[begin], -torch.inf)
    parents = []
    for row in range(begin + 1, end):
        options = previous[previous_columns] - 0.05 * offsets.abs()
        options = options.masked_fill(~possible, -torch.inf)
        best, choice = options.max(-1)
        parents.append(previous_columns.gather(1, choice[:, None]).squeeze(1))
        previous = (response[row] + best).masked_fill(~valid_centers[row], -torch.inf)
    if not torch.isfinite(previous).any():
        return None
    col = int(previous.argmax())
    result = [col]
    for parent in reversed(parents):
        col = int(parent[col])
        result.append(col)
    return torch.tensor(list(reversed(result)), dtype=torch.long)


def _ribbon(path, begin, start, stop, shape, half_width):
    result = torch.zeros(shape, dtype=torch.bool)
    for row in range(start, stop):
        column = int(path[row - begin])
        result[row, column - half_width:column + half_width + 1] = True
    return result


def _supplement(eligible, baseline, count, rng):
    """Prefer baseline blocks, then grow compact patches in remaining space."""
    output = torch.zeros_like(eligible)
    for preferred in (eligible & baseline, eligible):
        components = _components(preferred & ~output)
        components.sort(key=len, reverse=True)
        for component in components:
            take = min(count - int(output.sum()), len(component))
            if take:
                output |= _proposal(eligible, component, take, "compact", rng)
            if int(output.sum()) == count:
                return output
    if int(output.sum()) != count:
        raise RuntimeError("insufficient supplement capacity")
    return output


@torch.no_grad()
def compare_slender_masks(score, valid, baseline, *, seed=42, num_candidates=8,
                          temperature=0.1, near_constant=False, strip_width=5,
                          span_fraction=0.55, context_rows=2, axis="auto", selection_mode="best"):
    if valid.ndim != 2 or score.shape != valid.shape or baseline.shape != valid.shape:
        raise ValueError("expected matching [H,W] score, valid and baseline")
    if len({score.device, valid.device, baseline.device}) != 1:
        raise ValueError("inputs must share a device")
    if num_candidates < 2 or strip_width < 1 or strip_width % 2 != 1 or context_rows < 1:
        raise ValueError("positive odd strip-width, positive context-rows and at least two candidates required")
    if axis not in {"auto", "vertical", "horizontal"} or not 0 < span_fraction < 1:
        raise ValueError("invalid axis or span_fraction")
    if selection_mode not in {"best", "softmax"}:
        raise ValueError("selection mode must be best or softmax")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    device = score.device
    valid = valid.detach().bool().cpu().contiguous()
    baseline = baseline.detach().bool().cpu().contiguous()
    score = score.detach().float().cpu().contiguous()
    if (baseline & ~valid).any() or not torch.isfinite(score[valid]).all():
        raise ValueError("invalid baseline padding or nonfinite valid scores")
    if (score[valid] < 0).any() or (score[valid] > 1 + 1e-6).any():
        raise ValueError("expected scores in [0,1]")
    score = torch.where(valid, score, 0)
    occupied = valid.nonzero()
    if occupied.numel():
        height = int(occupied[:, 0].max() - occupied[:, 0].min() + 1)
        width = int(occupied[:, 1].max() - occupied[:, 1].min() + 1)
    else:
        height = width = 0
    selected_axis = ("vertical" if height >= width else "horizontal") if axis == "auto" else axis
    transpose = selected_axis == "horizontal"
    work_valid = valid.T.contiguous() if transpose else valid
    work_score = score.T.contiguous() if transpose else score
    half = strip_width // 2
    kernel = torch.ones(1, 1, 1, strip_width)
    centers = F.conv2d(work_valid.float()[None, None], kernel, padding=(0, half))[0, 0] >= strip_width - 1e-6
    response = F.conv2d(work_score[None, None], kernel / strip_width, padding=(0, half))[0, 0]
    segments = _runs(centers.any(-1))
    begin, end = max(segments, key=lambda r: r[1] - r[0]) if segments else (0, 0)
    budget = int(baseline.sum())
    core_rows = min(round((end - begin) * span_fraction), end - begin - 2 * context_rows,
                    budget // strip_width)
    core_count = max(core_rows, 0) * strip_width
    rng = random.Random(seed)
    blank = torch.zeros_like(valid)
    masks, cores, supplements, contexts = [baseline.clone()], [blank.clone()], [baseline.clone()], [blank.clone()]
    kinds = ["baseline_block"]
    generation = [{"generation_fallback": None, "path": None, "span_start_stop": None}]
    suppressed = response.clone()
    for index in range(1, num_candidates):
        rng = random.Random(seed + 1009 * index)
        kind = ("random_straight", "random_curve", "guided_curve")[(index - 1) % 3]
        if index == num_candidates - 1:
            kind = "guided_curve"
        reason = None
        core, supplement, context = blank.clone(), baseline.clone(), blank.clone()
        path = None
        start = stop = None
        if core_rows < 3:
            reason = "insufficient_length_or_budget"
        else:
            path = _path(centers, suppressed if kind == "guided_curve" else response,
                         begin, end, kind, rng, 1 if strip_width >= 3 else 0)
            if path is None:
                reason = "no_supported_path"
            else:
                starts = list(range(begin + context_rows, end - context_rows - core_rows + 1))
                if kind == "guided_curve":
                    start = max(starts, key=lambda s: float(suppressed[torch.arange(s, s + core_rows),
                                                                             path[s - begin:s - begin + core_rows]].mean()))
                else:
                    start = rng.choice(starts)
                stop = start + core_rows
                working_core = _ribbon(path, begin, start, stop, work_valid.shape, half)
                working_context = _ribbon(path, begin, start - context_rows, start, work_valid.shape, half)
                working_context |= _ribbon(path, begin, stop, stop + context_rows, work_valid.shape, half)
                core = working_core.T.contiguous() if transpose else working_core
                context = working_context.T.contiguous() if transpose else working_context
                # Keep supplementary blocks one patch away from the thin core.
                guard = F.max_pool2d(core.float()[None, None], 3, 1, 1)[0, 0].bool() | context
                eligible = valid & ~guard
                remaining = budget - core_count
                if int(eligible.sum()) < remaining:
                    reason = "insufficient_supplement_capacity_with_context"
                    core, supplement, context = blank.clone(), baseline.clone(), blank.clone()
                else:
                    supplement = _supplement(eligible, baseline, remaining, rng)
                    if kind == "guided_curve":
                        whole_band = _ribbon(path, begin, begin, end, work_valid.shape, half)
                        suppressed = suppressed.masked_fill(whole_band, 0)
        masks.append(core | supplement)
        cores.append(core)
        supplements.append(supplement)
        contexts.append(context)
        kinds.append(kind)
        generation.append({"generation_fallback": reason, "path": path.tolist() if path is not None else None,
                           "span_start_stop": [start, stop] if start is not None else None})
    masks, cores, supplements, contexts = map(torch.stack, (masks, cores, supplements, contexts))
    scores = (cores * score).sum((-2, -1)) / max(core_count, 1)
    full_scores = (masks * score).sum((-2, -1)) / max(budget, 1)
    scores[0] = full_scores[0]  # Display-only baseline reference, not an eligible core.
    eligible_indices = [i for i in range(1, num_candidates) if generation[i]["generation_fallback"] is None]
    reasons = []
    if near_constant:
        reasons.append("near_constant_structure_map")
    if not eligible_indices:
        reasons.append("no_feasible_slender_candidate")
    elif float(scores[eligible_indices].max() - scores[eligible_indices].min()) <= 1e-4:
        reasons.append("indistinguishable_core_scores")
    probabilities = torch.zeros(num_candidates)
    generator = torch.Generator().manual_seed((seed + 104729) % (2**63 - 1))
    random_index = eligible_indices[int(torch.randint(len(eligible_indices), (), generator=generator))] if eligible_indices else 0
    best_index = max(eligible_indices, key=lambda i: float(scores[i])) if eligible_indices else 0
    if reasons:
        adaptive_index = 0
        probabilities[0] = 1
    elif selection_mode == "best":
        adaptive_index = best_index
        probabilities[best_index] = 1
    else:
        probabilities[eligible_indices] = torch.softmax(scores[eligible_indices] / temperature, dim=0)
        adaptive_index = int(torch.multinomial(probabilities, 1, generator=generator))
    valid_mean = float(score[valid].mean()) if valid.any() else 0.0
    mass = float(score.sum())
    geometry = []
    for index, mask in enumerate(masks):
        geometry.append({**_geometry(mask, valid), **generation[index],
            "core_count": int(cores[index].sum()), "supplement_count": int(supplements[index].sum()),
            "core_components": len(_components(cores[index])), "context_count": int(contexts[index].sum()),
            "core_structure_mean": float(scores[index]) if index and cores[index].any() else None,
            "structure_mean": float(full_scores[index]),
            "score_enrichment": float(scores[index]) / valid_mean if valid_mean > 1e-8 else None,
            "visible_structure_mass_fraction": float(score[~mask & valid].sum()) / mass if mass > 1e-8 else None})
    diagnostics = {"target_count": budget, "valid_count": int(valid.sum()), "core_target_count": core_count,
        "selection_mode": selection_mode,
        "core_rows": max(core_rows, 0), "strip_width": strip_width, "path_axis": selected_axis,
        "supported_segment": [begin, end], "context_rows": context_rows,
        "eligible_candidate_indices": eligible_indices, "candidate_geometry": geometry,
        "used_fallback": bool(reasons), "fallback_reasons": reasons,
        "score_scope": "thin core only; baseline score is full-mask display reference",
        "uniform_expected_score": float(scores[eligible_indices].mean()) if eligible_indices else float(scores[0]),
        "adaptive_expected_score": float((scores * probabilities).sum()),
        "unique_candidate_count": len({m.numpy().tobytes() for m in masks})}
    return CandidateComparison(masks.to(device), tuple(kinds), scores.to(device), probabilities.to(device),
        random_index, adaptive_index, best_index, diagnostics, cores.to(device), supplements.to(device), contexts.to(device))
