"""Budget-preserving highest-A ribbons on online EMA Teacher anchor features.

The policy has no trainable or mutable sampling state. Choices depend on seed,
global iteration, rank and anchor index, so resume does not restart the curriculum
or consume the model/data RNG. Other global views and unmasked anchors are kept.
"""

from dataclasses import asdict, dataclass
import hashlib
import math
from numbers import Integral
import random

import torch
import torch.distributed as dist
from torch.nn import functional as F

from .adaptive_ribbon import adapt_ribbon_width
from .base import MaskPolicy
from .feature_structure_score import compute_structure_score
from .slender_mask_sampler import compare_slender_masks, _supplement


@dataclass(frozen=True)
class StructureMaskSettings:
    warmup_iterations: int = 6000
    ramp_iterations: int = 6000
    start_iteration: int = 0
    max_probability: float = 0.3
    num_candidates: int = 8
    strip_width: int = 5
    span_fraction: float = 0.55
    context_rows: int = 2
    path_axis: str = "auto"
    width_mode: str = "local"
    min_width: int = 3
    max_width: int = 9
    relative_threshold: float = 0.5
    min_prominence: float = 0.08
    smooth_penalty: float = 0.75
    guard_radius: int = 1
    score_radius: int = 1
    smooth_kernel: int = 3
    min_spread: float = 1e-5
    context_log_period: int = 100
    context_score_radius: int = 5
    context_threshold: float = 0.7
    visualization_period: int = 500

    def __post_init__(self):
        nonnegative = ("warmup_iterations", "ramp_iterations", "start_iteration", "context_log_period", "visualization_period")
        positive = ("num_candidates", "strip_width", "context_rows", "min_width", "max_width", "guard_radius",
                    "score_radius", "smooth_kernel", "context_score_radius")
        for name in nonnegative + positive:
            value = getattr(self, name)
            minimum = 0 if name in nonnegative else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"masking.structure.{name} must be an integer >= {minimum}")
        for name in ("strip_width", "min_width", "max_width", "smooth_kernel"):
            if getattr(self, name) % 2 != 1:
                raise ValueError(f"masking.structure.{name} must be odd")
        if self.num_candidates < 2 or not self.min_width <= self.strip_width <= self.max_width:
            raise ValueError("need at least two candidates and min_width <= strip_width <= max_width")
        for name in ("max_probability", "span_fraction", "relative_threshold", "min_prominence", "smooth_penalty", "min_spread", "context_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"masking.structure.{name} must be finite")
        if not 0 <= self.max_probability <= 1 or not 0 < self.span_fraction < 1 or not 0 < self.relative_threshold < 1:
            raise ValueError("invalid masking probabilities or fractions")
        if min(self.min_prominence, self.min_spread) <= 0 or self.smooth_penalty < 0 or not -1 <= self.context_threshold <= 1:
            raise ValueError("invalid structure thresholds or smoothing penalty")
        if self.path_axis not in ("auto", "vertical", "horizontal") or self.width_mode not in ("fixed", "image", "local"):
            raise ValueError("invalid structure path_axis or width_mode")


def structure_probability(iteration, settings):
    if isinstance(iteration, bool) or not isinstance(iteration, Integral) or iteration < 0:
        raise ValueError("structure masking requires a nonnegative global iteration")
    elapsed = int(iteration) - settings.start_iteration
    if elapsed < settings.warmup_iterations:
        return 0.0
    if settings.ramp_iterations == 0:
        return float(settings.max_probability)
    scale = min(max((elapsed - settings.warmup_iterations) / settings.ramp_iterations, 0.0), 1.0)
    return float(settings.max_probability * scale)


def policy_seed(seed, iteration, rank, index, purpose):
    payload = f"structure-v1:{seed}:{iteration}:{rank}:{index}:{purpose}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63 - 1)


@dataclass
class RibbonMask:
    mask: torch.Tensor
    core: torch.Tensor
    supplement: torch.Tensor
    context: torch.Tensor
    diagnostics: dict


@torch.no_grad()
def build_maxa_ribbon_mask(score, valid, baseline, settings, *, seed, near_constant=False):
    """Select at fixed width, adapt that path, then rebuild budget supplements.

    A failed final-width/budget check falls back to the original block. It never
    widens, clips the final core, reranks paths, or silently drops masked patches.
    """
    if score.shape != valid.shape or baseline.shape != valid.shape or valid.ndim != 2:
        raise ValueError("expected matching 2D score/valid/baseline")
    score, valid, baseline = score.detach().float().cpu(), valid.detach().bool().cpu(), baseline.detach().bool().cpu()
    if (baseline & ~valid).any():
        raise ValueError("anchor baseline masks padding")
    budget = int(baseline.sum())
    blank = torch.zeros_like(valid)
    def fallback(reason, **extra):
        return RibbonMask(baseline, blank.clone(), baseline.clone(), blank.clone(),
                          {"applied": False, "reason": reason, "target_count": budget, **extra})
    if budget == 0:
        return fallback("unmasked_anchor")
    if near_constant:
        return fallback("near_constant_structure_map")
    comparison = compare_slender_masks(score, valid, baseline, seed=seed, num_candidates=settings.num_candidates,
        near_constant=near_constant, strip_width=settings.strip_width, span_fraction=settings.span_fraction,
        context_rows=settings.context_rows, axis=settings.path_axis, selection_mode="best")
    if comparison.diagnostics["used_fallback"]:
        return fallback(";".join(comparison.diagnostics["fallback_reasons"]))
    best = comparison.best_index
    geometry = comparison.diagnostics["candidate_geometry"][best]
    core, context = comparison.core_masks[best], comparison.context_masks[best]
    width_mean = float(settings.strip_width)
    width_min = width_max = settings.strip_width
    weak_rows = 0
    if settings.width_mode != "fixed":
        try:
            ribbon = adapt_ribbon_width(score, valid, geometry["path"],
                path_begin=comparison.diagnostics["supported_segment"][0],
                span_start=geometry["span_start_stop"][0], span_stop=geometry["span_start_stop"][1],
                axis=comparison.diagnostics["path_axis"], mode=settings.width_mode,
                min_width=settings.min_width, max_width=settings.max_width, fallback_width=settings.strip_width,
                relative_threshold=settings.relative_threshold, min_prominence=settings.min_prominence,
                smooth_penalty=settings.smooth_penalty, context_rows=settings.context_rows)
        except ValueError as exc:
            return fallback("width_adaptation_unavailable", best_index=best, detail=str(exc))
        core, context = ribbon.core, ribbon.context
        width_mean, width_min, width_max = (ribbon.metrics[name] for name in ("width_mean", "width_min", "width_max"))
        weak_rows = ribbon.metrics["weak_response_rows"]
    core_count = int(core.sum())
    if core_count > budget:
        return fallback("adaptive_core_exceeds_budget", best_index=best, core_count=core_count)
    guard = F.max_pool2d(core.float()[None, None], 2 * settings.guard_radius + 1, 1, settings.guard_radius)[0, 0].bool() | context
    eligible = valid & ~guard
    remaining = budget - core_count
    if int(eligible.sum()) < remaining:
        return fallback("insufficient_supplement_capacity_with_context", best_index=best)
    supplement = _supplement(eligible, baseline, remaining, random.Random(seed + 49999))
    final = core | supplement
    if int(final.sum()) != budget or (final & ~valid).any() or (final & context).any() or (supplement & guard).any():
        raise RuntimeError("structure policy violated budget, validity, or context invariants")
    return RibbonMask(final, core, supplement, context, {
        "applied": True, "reason": "highest_A", "best_index": best, "target_count": budget,
        "core_count": core_count, "supplement_count": int(supplement.sum()),
        "context_count": int(context.sum()), "fixed_core_A": float(comparison.scores[best]),
        "final_core_A": float(score[core].mean()), "width_mean": width_mean, "width_min": width_min,
        "width_max": width_max, "weak_response_rows": weak_rows,
        "axis": comparison.diagnostics["path_axis"], "span_start_stop": geometry["span_start_stop"],
        "candidate_scores": comparison.scores.tolist(), "eligible_candidate_indices": comparison.diagnostics["eligible_candidate_indices"],
    })


@torch.no_grad()
def core_context_metrics(tokens, core, full_mask, valid, radius, threshold):
    """Full-image Teacher heuristic for the core under actual FULL visibility."""
    h, w = valid.shape
    tokens = F.normalize(tokens.detach().float().cpu().reshape(h * w, -1), dim=-1)
    positions = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), -1).reshape(-1, 2)
    ids = core.flatten().nonzero().flatten()
    visible = (valid & ~full_mask).flatten().nonzero().flatten()
    if not len(ids):
        raise ValueError("context diagnostic requires a nonempty core")
    if len(visible):
        nearby = (positions[ids, None] - positions[visible][None]).abs().amax(-1) <= radius
        values = (tokens[ids] @ tokens[visible].T).masked_fill(~nearby, -torch.inf).amax(-1).clamp(-1, 1)
        supported = nearby.any(-1)
    else:
        values = torch.zeros(len(ids))
        supported = torch.zeros(len(ids), dtype=torch.bool)
    return {"P": float(torch.where(supported, values, 0).mean()),
            "H": float(((~supported) | (values < threshold)).float().mean()),
            "missing": float((~supported).float().mean())}


class StructureMaxAMaskPolicy(MaskPolicy):
    def __init__(self, settings=None, *, seed=42):
        self.settings = settings or StructureMaskSettings()
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("structure policy seed must be a nonnegative integer")
        self.seed = int(seed)
        self.last_metrics = {}
        self.last_records = []
        self.last_examples = []

    @property
    def signature(self):
        fields = asdict(self.settings)
        for name in ("context_log_period", "context_score_radius", "context_threshold", "visualization_period"):
            fields.pop(name)
        return {"anchor_policy": "structure_maxa", "random_global_policy": "block", "algorithm_version": 1,
                "seed": self.seed, "settings": fields}

    @torch.no_grad()
    def select(self, candidate_masks, *, teacher_anchor_tokens, anchor_valid_mask, progress, iteration=None):
        del progress
        settings = self.settings
        probability = structure_probability(iteration, settings)
        if candidate_masks.dtype != torch.bool or candidate_masks.ndim != 2 or anchor_valid_mask.ndim != 3:
            raise ValueError("expected boolean [2B,N] masks and [B,H,W] anchor validity")
        batch, h, w = anchor_valid_mask.shape
        if candidate_masks.shape != (2 * batch, h * w) or teacher_anchor_tokens.shape[:2] != (batch, h * w):
            raise ValueError("expected view-major anchors first, with aligned Teacher features")
        metrics = {"mask_structure_probability": probability,
            "mask_structure_eligible_anchors": 0, "mask_structure_attempted_anchors": 0,
            "mask_structure_applied_anchors": 0, "mask_structure_fallback_anchors": 0,
            "mask_structure_core_patches": 0, "mask_structure_supplement_patches": 0,
            "mask_structure_core_A_sum": 0.0, "mask_structure_width_mean_sum": 0.0,
            "mask_structure_context_scored_cores": 0, "mask_structure_context_P_sum": 0.0,
            "mask_structure_context_H_sum": 0.0, "mask_structure_context_missing_sum": 0.0}
        self.last_metrics, self.last_records, self.last_examples = metrics, [], []
        counts = candidate_masks[:batch].sum(-1).detach().cpu().tolist()
        eligible = [index for index, count in enumerate(counts) if count > 0]
        metrics["mask_structure_eligible_anchors"] = len(eligible)
        # The p=0 path does not compute scores, copy features, or draw any RNG.
        if probability == 0:
            return candidate_masks, None
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        attempts = [index for index in eligible if random.Random(policy_seed(self.seed, iteration, rank, index, "gate")).random() < probability]
        metrics["mask_structure_attempted_anchors"] = len(attempts)
        if not attempts:
            return candidate_masks, None
        # Transfer only attempted anchor features, once. No extra Teacher forward.
        tokens = teacher_anchor_tokens[attempts].detach().float().cpu()
        valid = anchor_valid_mask[attempts].detach().bool().cpu()
        baselines = candidate_masks[attempts].detach().cpu().reshape(-1, h, w)
        final = candidate_masks.clone()
        capture = settings.visualization_period > 0 and (iteration + 1) % settings.visualization_period == 0
        log_context = settings.context_log_period > 0 and (iteration + 1) % settings.context_log_period == 0
        for local, index in enumerate(attempts):
            structure = compute_structure_score(tokens[local], valid[local], radius=settings.score_radius,
                smooth_kernel=settings.smooth_kernel, min_spread=settings.min_spread)
            seed = policy_seed(self.seed, iteration, rank, index, "proposals")
            ribbon = build_maxa_ribbon_mask(structure.score, valid[local], baselines[local], settings,
                                          seed=seed, near_constant=structure.metrics["near_constant"])
            self.last_records.append({"anchor_index": index, "seed": seed, **ribbon.diagnostics})
            if ribbon.diagnostics["applied"]:
                final[index] = ribbon.mask.flatten().to(final.device)
                metrics["mask_structure_applied_anchors"] += 1
                metrics["mask_structure_core_patches"] += ribbon.diagnostics["core_count"]
                metrics["mask_structure_supplement_patches"] += ribbon.diagnostics["supplement_count"]
                metrics["mask_structure_core_A_sum"] += ribbon.diagnostics["final_core_A"]
                metrics["mask_structure_width_mean_sum"] += ribbon.diagnostics["width_mean"]
                if log_context:
                    context = core_context_metrics(tokens[local], ribbon.core, ribbon.mask, valid[local],
                                                   settings.context_score_radius, settings.context_threshold)
                    metrics["mask_structure_context_scored_cores"] += 1
                    for name in ("P", "H", "missing"):
                        metrics[f"mask_structure_context_{name}_sum"] += context[name]
            else:
                metrics["mask_structure_fallback_anchors"] += 1
            if capture:
                self.last_examples.append({"anchor_index": index, "score": structure.score, "valid": valid[local], "baseline": baselines[local],
                                           "ribbon": ribbon})
        return final, None


def build_mask_policy(masking_cfg, *, seed=42):
    from .block_mask import BlockMaskPolicy
    anchor = str(masking_cfg.get("anchor_policy", "block"))
    other = str(masking_cfg.get("random_global_policy", "block"))
    if other != "block":
        raise NotImplementedError("random_global_policy currently supports only block")
    if anchor == "block":
        return BlockMaskPolicy()
    if anchor != "structure_maxa":
        raise NotImplementedError(f"unknown anchor mask policy: {anchor}")
    settings = StructureMaskSettings(**dict(masking_cfg.get("structure", {})))
    return StructureMaxAMaskPolicy(settings, seed=seed)
