"""Offline equal-budget spatial proposals and structure-only selection.

Candidates do not read teacher scores. Connected growth follows randomly
oriented rectangle/adjacent-region distance fields and never adds isolated
patches to meet a budget. This CPU proposal implementation is not a training
mask policy; context recoverability is deliberately not scored here.
"""

from dataclasses import dataclass
import heapq
import math
import random

import torch


@dataclass(frozen=True)
class CandidateComparison:
    masks: torch.Tensor  # [K,H,W], candidate 0 is the supplied baseline
    kinds: tuple[str, ...]
    scores: torch.Tensor
    probabilities: torch.Tensor
    random_index: int
    adaptive_index: int
    best_index: int
    diagnostics: dict
    core_masks: torch.Tensor | None = None
    supplement_masks: torch.Tensor | None = None
    context_masks: torch.Tensor | None = None


def _neighbors(index, height, width):
    y, x = divmod(index, width)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yy, xx = y + dy, x + dx
        if 0 <= yy < height and 0 <= xx < width:
            yield yy * width + xx


def _components(valid):
    h, w = valid.shape
    eligible = valid.flatten().tolist()
    seen = set()
    groups = []
    for index, active in enumerate(eligible):
        if not active or index in seen:
            continue
        stack, component = [index], []
        seen.add(index)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in _neighbors(current, h, w):
                if eligible[neighbor] and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        groups.append(sorted(component))
    return groups


def _proposal(valid, component, budget, kind, rng):
    h, w = valid.shape
    cy, cx = divmod(rng.choice(component), w)
    angle = rng.uniform(0, math.pi)
    aspect = rng.uniform(1.0, 1.7) if kind == "compact" else rng.uniform(2.0, 4.0)
    a, b = math.sqrt(budget * aspect) / 2, math.sqrt(budget / aspect) / 2
    cosine, sine = math.cos(angle), math.sin(angle)
    # Adjacent, overlapping small regions share one rotated coordinate frame.
    part_count = rng.randint(2, 4) if kind == "multi_region" else 1
    offsets = [(i - (part_count - 1) / 2) * a / max(part_count - 1, 1) for i in range(part_count)]
    jitters = [rng.uniform(-0.2, 0.2) * b for _ in offsets]
    potential = {}
    for index in component:
        y, x = divmod(index, w)
        u = (x - cx) * cosine + (y - cy) * sine
        v = -(x - cx) * sine + (y - cy) * cosine
        if kind == "multi_region":
            value = min(max(abs(u - offset) / max(a / part_count, 0.5),
                            abs(v - jitter) / max(b, 0.5)) for offset, jitter in zip(offsets, jitters))
        else:
            value = max(abs(u) / max(a, 0.5), abs(v) / max(b, 0.5))
        potential[index] = value + rng.random() * 1e-6
    start = min(component, key=lambda index: potential[index])
    frontier, queued, selected = [(potential[start], start)], {start}, []
    while frontier and len(selected) < budget:
        _, current = heapq.heappop(frontier)
        selected.append(current)
        for neighbor in _neighbors(current, h, w):
            if neighbor in potential and neighbor not in queued:
                queued.add(neighbor)
                heapq.heappush(frontier, (potential[neighbor], neighbor))
    if len(selected) != budget:
        raise RuntimeError("connected proposal failed to fill its eligible component")
    # flatten() must be a view for indexed assignment, including transposed FOVs.
    mask = torch.zeros(valid.shape, dtype=torch.bool, device=valid.device)
    mask.flatten()[selected] = True
    return mask


def _geometry(mask, valid):
    h, w = mask.shape
    perimeter = 0
    border = 0
    for index in mask.flatten().nonzero().flatten().tolist():
        neighbors = list(_neighbors(index, h, w))
        perimeter += 4 - len(neighbors)
        perimeter += sum(not bool(mask.flatten()[j]) for j in neighbors)
        border += len(neighbors) < 4 or any(not bool(valid.flatten()[j]) for j in neighbors)
    count = int(mask.sum())
    return {"masked_count": count, "component_count": len(_components(mask)),
            "perimeter_edges": perimeter, "perimeter_per_patch": perimeter / max(count, 1),
            "valid_border_fraction": border / max(count, 1)}


@torch.no_grad()
def sample_candidate_masks(valid, baseline, *, seed=42, num_candidates=8):
    """Keep the exact baseline budget and include it unchanged as candidate 0.

    Returns (masks, kinds, generation diagnostics). CPU growth uses a private
    RNG. For fragmented validity with no large enough component, proposals
    explicitly fall back to baseline rather than silently disconnecting.
    """
    if valid.ndim != 2 or baseline.shape != valid.shape or min(valid.shape) < 1:
        raise ValueError("expected nonempty matching [H,W] validity and baseline")
    if valid.device != baseline.device or num_candidates < 2:
        raise ValueError("same device and at least two candidates required")
    device = valid.device
    valid, baseline = valid.detach().bool().cpu(), baseline.detach().bool().cpu()
    if (baseline & ~valid).any():
        raise ValueError("baseline masks invalid padding")
    budget = int(baseline.sum())
    components = [part for part in _components(valid) if len(part) >= budget]
    masks, kinds, info = [baseline.clone()], ["baseline_block"], [{"duplicate": False, "generation_fallback": None}]
    rng = random.Random(seed)
    seen = {baseline.numpy().tobytes()}
    for index in range(1, num_candidates):
        kind = ("compact", "span", "multi_region")[(index - 1) % 3]
        reason = "zero_budget" if budget == 0 else "fragmented_validity" if not components else None
        duplicate = False
        for _ in range(8):
            mask = baseline.clone() if reason else _proposal(valid, rng.choice(components), budget, kind, rng)
            key = mask.numpy().tobytes()
            duplicate = key in seen
            if not duplicate or reason:
                break
        seen.add(key)
        masks.append(mask)
        kinds.append(kind)
        info.append({"duplicate": duplicate, "generation_fallback": reason})
    return torch.stack(masks).to(device), tuple(kinds), info


@torch.no_grad()
def compare_candidate_masks(score, valid, baseline, *, seed=42, num_candidates=8,
                            temperature=0.1, near_constant=False, min_score_spread=1e-4,
                            selection_mode="best"):
    """Select maximum-A by default; optionally reproduce historical softmax.

    A is only mean normalized structure score inside the mask. No P/H,
    reconstruction prediction or anatomical label is used.
    """
    if score.shape != valid.shape or score.device != valid.device:
        raise ValueError("score and validity must share shape and device")
    if selection_mode not in {"best", "softmax"}:
        raise ValueError("selection mode must be best or softmax")
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(min_score_spread) or min_score_spread < 0:
        raise ValueError("invalid temperature or score-spread threshold")
    if not torch.isfinite(score[valid.bool()]).all():
        raise ValueError("valid structure scores must be finite")
    if (score[valid.bool()] < 0).any() or (score[valid.bool()] > 1 + 1e-6).any():
        raise ValueError("expected normalized structure scores in [0,1]")
    masks, kinds, info = sample_candidate_masks(valid, baseline, seed=seed, num_candidates=num_candidates)
    score = torch.where(valid.bool(), score.detach().float(), 0)
    count = int(baseline.sum())
    scores = (masks * score).sum((-2, -1)) / max(count, 1)
    reasons = []
    if count == 0:
        reasons.append("zero_budget")
    if near_constant:
        reasons.append("near_constant_structure_map")
    if float(scores.max() - scores.min()) <= min_score_spread:
        reasons.append("indistinguishable_candidate_scores")
    probabilities = torch.softmax((scores - scores.max()) / temperature, dim=0)
    generator = torch.Generator(device="cpu").manual_seed((seed + 104729) % (2**63 - 1))
    random_index = int(torch.randint(num_candidates, (), generator=generator))
    best_index = int(scores.argmax())
    if reasons:
        adaptive_index = 0
        probabilities = torch.zeros_like(scores)
        probabilities[0] = 1
    elif selection_mode == "best":
        adaptive_index = best_index
        probabilities = torch.zeros_like(scores)
        probabilities[best_index] = 1
    else:
        adaptive_index = int(torch.multinomial(probabilities.cpu(), 1, generator=generator))
    cpu_valid = valid.bool().cpu()
    score_mass = float(score.sum())
    valid_mean = float(score[valid.bool()].mean()) if valid.any() else 0.0
    geometry = []
    for index, mask in enumerate(masks):
        geometry.append({**_geometry(mask.cpu(), cpu_valid), **info[index],
                         "structure_mean": float(scores[index]),
                         "score_enrichment": float(scores[index]) / valid_mean if valid_mean > 1e-8 else None,
                         "visible_structure_mass_fraction": float(score[~mask & valid.bool()].sum()) / score_mass if score_mass > 1e-8 else None})
    diagnostics = {"target_count": count, "valid_count": int(valid.sum()),
                   "selection_mode": selection_mode,
                   "unique_candidate_count": len({mask.cpu().numpy().tobytes() for mask in masks}),
                   "fallback_reasons": reasons, "used_fallback": bool(reasons),
                   "candidate_score_spread": float(scores.max() - scores.min()),
                   "uniform_expected_score": float(scores.mean()),
                   "adaptive_expected_score": float((scores * probabilities).sum()),
                   "selection_entropy": float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()),
                   "candidate_geometry": geometry}
    return CandidateComparison(masks, kinds, scores, probabilities, random_index,
                               adaptive_index, best_index, diagnostics)
