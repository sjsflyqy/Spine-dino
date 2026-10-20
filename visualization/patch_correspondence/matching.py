"""Within-model cosine retrieval, spatial ambiguity, and explicit case ranking."""

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import grid_centres, sample_tokens, valid_tokens


@torch.inference_mode()
def extract_pair(loaded, a, b, device):
    # Separate forwards reduce memory. Only final backbone patch tokens are used.
    tokens = [loaded.model.forward_features(view.tensor[None].to(device))["x_norm_patchtokens"][0].float()
              for view in (a, b)]
    expected = (a.image.width // loaded.patch_size) ** 2
    if any(value.ndim != 2 or value.shape[0] != expected or not torch.isfinite(value).all() for value in tokens):
        raise ValueError("Unexpected/nonfinite backbone patch token output")
    return tokens


def prepared_features(tokens, queries, a, b, patch_size, border):
    query = sample_tokens(tokens[0], queries["xy_a"], a.image.width)
    query_valid = query.norm(dim=-1) > 1e-8
    b_valid = valid_tokens(b, patch_size, border).flatten().to(tokens[1].device)
    b_valid &= tokens[1].norm(dim=-1) > 1e-8
    if not b_valid.any():
        raise ValueError("No nonzero valid target patch features")
    return F.normalize(query, dim=-1), F.normalize(tokens[1], dim=-1), query_valid.cpu(), b_valid


@torch.inference_mode()
def measurements(query_features, target_features, b_valid, gt_b, *, size, patch_size,
                 near_radius, far_radius, peak_separation, peak_delta=0.05,
                 min_peak_similarity=0.3, temperature=0.07, chunk_size=128):
    """All distances are View B pixels; heatmaps are unmodified cosine values.

    Peaks use a common physical-radius NMS, capped at six. High-response counts
    use max(cosine_max - peak_delta, min_peak_similarity) for diagnostics ONLY.
    Softmax mass is a ranking diagnostic, never substituted for display cosine.
    """
    device = target_features.device
    centres = grid_centres(size, patch_size).to(device)
    all_metrics, heatmaps = [], []
    for start in range(0, len(query_features), chunk_size):
        raw = (query_features[start:start + chunk_size] @ target_features.T).clamp(-1, 1)
        similarity = raw.masked_fill(~b_valid[None], -torch.inf)
        maximum, index = similarity.max(dim=-1)
        predicted = centres[index]
        gt = gt_b[start:start + chunk_size].to(device)
        distance = torch.cdist(gt[None], centres[None])[0]
        near = (distance <= near_radius) & b_valid[None]
        far = (distance >= far_radius) & b_valid[None]
        high = (similarity >= torch.maximum(maximum[:, None] - peak_delta,
                                            maximum.new_tensor(min_peak_similarity))) & b_valid[None]
        high_count = high.sum(-1).clamp_min(1)
        mass = F.softmax(similarity / temperature, dim=-1)
        side = size // patch_size
        local_maximum = F.max_pool2d(similarity.reshape(-1, 1, side, side), 3, 1, 1).flatten(1)
        remaining = similarity.masked_fill(similarity < local_maximum, -torch.inf)
        peak_count = torch.zeros(len(raw), device=device, dtype=torch.long)
        for _ in range(6):
            value, peak = remaining.max(-1)
            accepted = (value >= maximum - peak_delta) & (value >= min_peak_similarity) & torch.isfinite(value)
            peak_count += accepted.long()
            separation = torch.cdist(centres[peak][None], centres[None])[0]
            remaining.masked_fill_(separation < peak_separation, -torch.inf)
        error = (predicted - gt).norm(dim=-1)
        metrics = {
            "pred_x": predicted[:, 0], "pred_y": predicted[:, 1], "error_px": error,
            "error_patch": error / patch_size, "peak_similarity": maximum,
            "separated_peaks": peak_count,
            "near_gt_mass": (mass * near).sum(-1),
            "near_gt_high_fraction": (high & near).sum(-1) / high_count,
            "far_high_fraction": (high & far).sum(-1) / high_count,
        }
        all_metrics.append({key: value.cpu().numpy() for key, value in metrics.items()})
        heatmaps.append(raw.cpu())
    metrics = {key: np.concatenate([part[key] for part in all_metrics]) for key in all_metrics[0]}
    side = size // patch_size
    return metrics, torch.cat(heatmaps).numpy().reshape(-1, side, side)


def ranking(baseline, ours, queries, texture, source_size, image_quality, *, size, reference_patch,
            baseline_min_patches=6.0, ours_max_patches=1.5, min_gain_patches=4.0,
            min_query_gradient=0.004, min_match_similarity=0.3, require_ambiguity=False):
    intensity, gradient, spread = texture
    gain = baseline["error_px"] - ours["error_px"]
    informative = (gradient >= min_query_gradient) & (spread >= 0.01) & (intensity >= 0.04) & (intensity <= 0.96)
    error_ok = ((baseline["error_px"] >= baseline_min_patches * reference_patch)
                & (ours["error_px"] <= ours_max_patches * reference_patch)
                & (gain >= min_gain_patches * reference_patch))
    ambiguity = ((baseline["separated_peaks"] >= 2)
                 & (ours["near_gt_mass"] >= baseline["near_gt_mass"] + 0.05)
                 & (ours["far_high_fraction"] < baseline["far_high_fraction"]))
    confident = ((baseline["peak_similarity"] >= min_match_similarity)
                 & (ours["peak_similarity"] >= min_match_similarity))
    eligible = informative & error_ok & confident & (ambiguity if require_ambiguity else True)
    normalized_xy = queries["source_xy"].numpy() / np.asarray(source_size)
    # Soft image-centre prior helps reduce border text; it is NOT an anatomical ROI.
    central = np.clip(1 - np.abs(normalized_xy[:, 0] - 0.5) / 0.5, 0, 1)
    quality = 0.5 * np.clip(gradient / 0.04, 0, 1) + 0.25 * np.clip(spread / 0.12, 0, 1) + 0.25 * central
    concentration_gain = ours["near_gt_mass"] - baseline["near_gt_mass"]
    score = gain / (size * 2 ** 0.5) + 0.3 * concentration_gain + 0.05 * quality + 0.05 * image_quality
    return {"eligible": eligible, "ambiguity_improved": ambiguity, "error_criteria_met": error_ok,
            "informative_texture": informative, "gain_px": gain, "query_quality": quality,
            "confident_matches": confident, "query_gradient": gradient, "query_std": spread, "score": score}


def rank_key(record):
    return (bool(record["eligible"]), bool(record["ambiguity_improved"]), float(record["score"]))


class Summary:
    """All-query aggregation, distinct from the deliberately selected illustrations."""
    def __init__(self, thresholds):
        self.thresholds = list(thresholds)
        self.count = 0
        self.totals = {name: {"error_sum": 0.0, "pck_hits": np.zeros(len(thresholds), dtype=np.int64)}
                       for name in ("baseline", "ours")}

    def add(self, baseline, ours):
        self.count += len(baseline["error_px"])
        for name, values in (("baseline", baseline), ("ours", ours)):
            errors = values["error_px"]
            self.totals[name]["error_sum"] += float(errors.astype(np.float64).sum())
            self.totals[name]["pck_hits"] += (errors[:, None] <= np.asarray(self.thresholds)[None]).sum(0)

    def report(self):
        return {"query_count": self.count, "models": {
            name: {"mean_error_px": value["error_sum"] / self.count if self.count else None,
                   "pck": {f"{threshold:g}px": int(hits) / self.count if self.count else None
                           for threshold, hits in zip(self.thresholds, value["pck_hits"])}}
            for name, value in self.totals.items()}}
