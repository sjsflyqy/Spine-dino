"""Step 3: measure recovery and test context dependence with frozen trained modules.

Consumes a step-2 core-only candidate run, keeping highest-A selection. This is
an offline diagnostic on development images, not a training or held-out result.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "upstream" / "dinov2-main"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from methods.geotopo_dino.masking.adaptive_ribbon import adapt_ribbon_width
from methods.geotopo_dino.losses.pixel_reconstruction_loss import patchify, unpatchify
from visualization.structure_maps.context_metrics import (
    context_score, context_interventions, patch_errors, prototype_errors, visible_baselines, spearman,
)
from visualization.structure_maps.reconstruction_model import load_bundle
from visualization.spine_masks.visualize import make_anchor, seeded_cpu_random


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_source(source, record, key):
    relative = Path(record["output"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("case paths must be relative to the candidate run")
    origin = (source / relative).resolve()
    if not origin.is_relative_to(source):
        raise ValueError("case escaped source run")
    metadata = json.loads((origin / "metadata.json").read_text())
    if key not in metadata["comparisons"]:
        raise ValueError(f"comparison {key} unavailable: {list(metadata['comparisons'])}")
    comparison = metadata["comparisons"][key]
    if comparison.get("budget_mode") != "core-only":
        raise ValueError("generate step 2 with --budget-mode core-only for budget-matched recovery tests")
    with np.load(origin / "candidate_masks.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    structure_case = Path(metadata["source_case"])
    structure_metadata = json.loads((structure_case / "metadata.json").read_text())
    with Image.open(record["image"]) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    tensor, geometry, display = make_anchor(original, size=518, mode=comparison["mode"], seed=structure_metadata["seed"])
    valid = torch.from_numpy(arrays[f"{comparison['mode']}_valid"]).bool()
    if not torch.equal(geometry["valid_mask"], valid):
        raise ValueError("reconstructed input geometry differs from step 2")
    with Image.open(structure_case / f"input_{comparison['mode']}.png") as opened:
        if not np.array_equal(np.asarray(display), np.asarray(opened.convert("RGB"))):
            raise ValueError("reconstructed input pixels differ from cached step 1")
    return relative, metadata, comparison, arrays, tensor, valid, display


def _mask_pool(comparison, arrays, valid, score, key, random_controls, width_mode="local"):
    from methods.geotopo_dino.data.collate import _block_mask_within_valid
    masks, labels = [], []
    def add(mask, label):
        mask = mask.bool().cpu()
        if not mask.any() or (mask & ~valid).any():
            raise ValueError("invalid evaluation mask")
        for index, previous in enumerate(masks):
            if torch.equal(mask, previous):
                labels[index].append(label)
                return index
        masks.append(mask)
        labels.append([label])
        return len(masks) - 1
    best = 0 if comparison["used_fallback"] else comparison["best_index"]
    raw = torch.from_numpy(arrays[f"{key}_masks"])
    for index, mask in enumerate(raw):
        add(mask, f"candidate_{index}_{comparison['kinds'][index]}" + ("_MAX_A_fixed" if index == best else ""))
    final = add(raw[best], "selected_final")
    width_metadata = {}
    if best and not comparison["used_fallback"]:
        stats = comparison["candidate_geometry"][best]
        for mode in ("image", "local"):
            ribbon = adapt_ribbon_width(score, valid, stats["path"],
                path_begin=comparison["supported_segment"][0], span_start=stats["span_start_stop"][0],
                span_stop=stats["span_start_stop"][1], axis=comparison["path_axis"], mode=mode,
                fallback_width=comparison["strip_width"], context_rows=comparison["context_rows"])
            selected = add(ribbon.core, f"MAX_A_{mode}")
            width_metadata[mode] = ribbon.metrics
            if mode == width_mode:
                labels[final].remove("selected_final")
                final = selected
                labels[final].append("selected_final")
    # Equal counts for each width variant, as well as the original fixed cores.
    for count in sorted({int(mask.sum()) for mask in masks}):
        for repeat in range(random_controls):
            with seeded_cpu_random((comparison["seed"] + count * 101 + repeat * 7919) % (2**32)):
                add(_block_mask_within_valid(valid, count), f"random_block_N{count}_r{repeat}")
    return masks, labels, final, width_metadata


def _rgb(patches):
    tensor = unpatchify(patches[None].cpu(), 14, (37, 37))[0]
    mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _render(path, title, display, ground_truth, entries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization.structure_maps.candidates import _draw_input, _draw_mask, _expand
    vmax = max(float(np.nanpercentile(entry["error_map"], 95)) for entry in entries)
    vmax = max(vmax, 1e-6)
    fig, axes = plt.subplots(len(entries), 4, figsize=(15, 3.8 * len(entries)), squeeze=False)
    original = np.asarray(display).astype(np.float32) / 255
    for row, entry in enumerate(entries):
        target, input_mask = entry["target"], entry["input_mask"]
        masked = original.copy()
        masked[_expand(input_mask.numpy(), display)] = 0.35
        _draw_input(axes[row, 0], Image.fromarray((masked * 255).round().astype(np.uint8)),
                    f"{entry['name']}\ninput hidden={int(input_mask.sum())}; evaluated={int(target.sum())}")
        _draw_mask(axes[row, 1], display, target, "Red=evaluated target; blue=extra hidden", input_mask & ~target)
        composite = ground_truth.clone()
        composite[target.flatten()] = entry["prediction"][target.flatten()]
        axes[row, 2].imshow(_rgb(composite))
        axes[row, 2].set_title(f"Composite: prediction only inside RED\nMSE={entry['metrics']['pixel_mse']:.4f}; Haar={entry['metrics']['haar_l1']:.4f}", fontsize=10)
        axes[row, 2].axis("off")
        _draw_input(axes[row, 3], display, f"Target pixel error (shared scale)\nP5={entry['metrics']['P_r5']:.3f}; KL={entry['metrics']['ibot_kl']:.3f}" if "P_r5" in entry["metrics"] else "Target pixel error (shared scale)")
        axes[row, 3].imshow(np.ma.masked_invalid(_expand(entry["error_map"], display)), cmap="inferno", vmin=0, vmax=vmax, alpha=0.9)
    fig.suptitle(title + "\nUnmasked image outside the red target is copied for display. Gray input indicates token replacement.", fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94, hspace=0.26, wspace=0.08)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _render_detail(path, display, ground_truth, entries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = entries[0]["target"]
    positions = target.nonzero()
    lo = (positions.amin(0) - 1).clamp_min(0) * 14
    hi = (positions.amax(0) + 2).clamp_max(37) * 14
    region = (slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1])))
    original = _rgb(ground_truth)
    fig, axes = plt.subplots(len(entries), 3, figsize=(10, 3.8 * len(entries)), squeeze=False)
    for row, entry in enumerate(entries):
        composite = ground_truth.clone()
        composite[target.flatten()] = entry["prediction"][target.flatten()]
        predicted = _rgb(composite)
        diff = np.abs(predicted - original).mean(-1)
        # Outside the target is copied and carries no model recovery evidence.
        expanded = target.numpy().repeat(14, 0).repeat(14, 1)
        diff[~expanded] = np.nan
        axes[row, 0].imshow(original[region])
        axes[row, 0].set_title("Original target region", fontsize=10)
        axes[row, 1].imshow(predicted[region])
        axes[row, 1].set_title(f"{entry['name']}\nMSE={entry['metrics']['pixel_mse']:.4f}; Haar={entry['metrics']['haar_l1']:.4f}", fontsize=9)
        axes[row, 2].imshow(original[region])
        axes[row, 2].imshow(np.ma.masked_invalid(diff[region]), cmap="inferno", vmin=0, vmax=0.2, alpha=0.85)
        axes[row, 2].set_title("RGB absolute error inside target\nshared scale: 0 to 0.2", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Target close-up: inspect vertebral borders and texture, not just overall brightness", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _correlations(rows):
    result = []
    groups = {row["case"] for row in rows}
    for case in sorted(groups):
        case_rows = [row for row in rows if row["case"] == case]
        # Same count as fixed highest-A core; varying widths are separately recorded.
        fixed = next(row for row in case_rows if "MAX_A_fixed" in row["labels"])
        matched = [row for row in case_rows if row["target_count"] == fixed["target_count"]]
        for subset_name, subset in (("all", case_rows), ("matched_fixed_count", matched)):
            for radius in (3, 5):
                for error in ("pixel_mse", "haar_l1", "ibot_kl"):
                    p = [row[f"P_r{radius}"] for row in subset]
                    y = [row[error] for row in subset]
                    controls = [[row["target_count"], row["target_variance"], row["A"]] for row in subset]
                    result.append({"case": case, "subset": subset_name, "n": len(subset), "radius": radius,
                                   "error": error, "spearman": spearman(p, y),
                                   "partial_rank_correlation": spearman(p, y, controls)})
    return result


def _summarize(rows, interventions, correlations):
    def average(values):
        values = [value for value in values if value is not None]
        return float(np.mean(values)) if values else None
    aggregate = {}
    for error in ("pixel_mse", "haar_l1", "ibot_kl"):
        corr = [row["spearman"] for row in correlations if row["subset"] == "matched_fixed_count" and row["radius"] == 5 and row["error"] == error]
        comparisons = []
        for case in sorted({row["case"] for row in interventions}):
            local = [row for row in interventions if row["case"] == case]
            base = next(row for row in local if row["condition"] == "normal")
            near = next(row for row in local if row["condition"] == "remove_near")
            far = [row for row in local if row["condition"].startswith("remove_far_")]
            comparisons.append({"case": case, "base": base[error], "near": near[error],
                "far_mean": average([row[error] for row in far]),
                "near_minus_base": near[error] - base[error],
                "far_minus_base": average([row[error] - base[error] for row in far]),
                "near_minus_far": near[error] - average([row[error] for row in far])})
        aggregate[error] = {"macro_matched_P5_spearman": average(corr),
                            "negative_correlation_images": sum(value < 0 for value in corr),
                            "correlation_images": len([value for value in corr if value is not None]),
                            "context_comparisons": comparisons,
                            "macro_near_minus_far": average([row["near_minus_far"] for row in comparisons]),
                            "near_worse_than_far_images": sum(row["near_minus_far"] > 0 for row in comparisons)}
    return aggregate


def _scatter(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = sorted({row["case"] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for case_index, case in enumerate(cases):
        local = [row for row in rows if row["case"] == case]
        fixed = next(row for row in local if "MAX_A_fixed" in row["labels"])
        local = [row for row in local if row["target_count"] == fixed["target_count"]]
        for ax, error in zip(axes, ("pixel_mse", "haar_l1", "ibot_kl")):
            ax.scatter([row["P_r5"] for row in local], [row[error] for row in local], label=case.split("/")[0], alpha=0.8)
            ax.set(xlabel="P5 (unmasked Teacher heuristic)", ylabel=error)
            ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=7)
    fig.suptitle("Same target count within each image; use per-image correlations in correlations.csv")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


@torch.inference_mode()
def verify_hidden_pixel_invariance(modules, tensor, mask, reference):
    """Changing hidden raw pixels must not change decoder output anywhere."""
    expanded = mask.repeat_interleave(14, 0).repeat_interleave(14, 1).to(tensor.device)
    modified = tensor.clone()
    modified[:, :, expanded] = 7.0
    tokens = modules["student"].forward_features(modified, masks=mask.flatten()[None].to(tensor.device))["x_norm_patchtokens"]
    prediction = modules["decoder"](tokens)[0].cpu()
    maximum = float((prediction - reference).abs().max())
    if not torch.equal(prediction, reference):
        raise ValueError(f"hidden-pixel invariance check failed: maximum decoder change {maximum}")
    return {"passed": True, "replaced_hidden_normalized_pixel_value": 7.0,
            "evaluated_output": "entire decoder patch grid", "max_absolute_decoder_change": maximum}


def render_overview(root, completed):
    """Summarize final masks using saved outputs without any model inference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization.structure_maps.candidates import _draw_input, _draw_mask
    for start in range(0, len(completed), 6):
        records = completed[start:start + 6]
        fig, axes = plt.subplots(len(records), 3, figsize=(12, 3.9 * len(records)), squeeze=False)
        for row, record in enumerate(records):
            case = root / record["output"]
            metadata = json.loads((case / "metadata.json").read_text())
            index = metadata["selected_mask_index"]
            prefix = f"mask_{index:02d}"
            with np.load(case / "raw_results.npz", allow_pickle=False) as arrays:
                target = torch.from_numpy(arrays[f"{prefix}_target"])
                pixels = torch.from_numpy(arrays["target_pixels"]).clone()
                pixels[target.flatten()] = torch.from_numpy(arrays[f"{prefix}_prediction_on_target"])
                output = _rgb(pixels)
            with Image.open(case / "input.png") as opened:
                display = opened.convert("RGB")
            metrics = metadata["metrics"][index]
            _draw_input(axes[row, 0], display, Path(record["image"]).name)
            _draw_mask(axes[row, 1], display, target, f"Highest-A path; selected width mode\nN={int(target.sum())}; P5={metrics['P_r5']:.3f}")
            axes[row, 2].imshow(output)
            axes[row, 2].set_title(f"Prediction only inside red target\nMSE={metrics['pixel_mse']:.4f}; Haar={metrics['haar_l1']:.4f}", fontsize=10)
            axes[row, 2].axis("off")
        fig.suptitle("Final mask recovery overview: outside the target is copied from the original", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(root / f"overview_reconstruction_{start // 6 + 1:03d}.jpg", dpi=130)
        plt.close(fig)


@torch.inference_mode()
def run_case(args, record, modules, center, bundle, verify_invariance=False):
    relative, metadata, comparison, arrays, tensor, valid, display = _read_source(args.from_run, record, args.comparison_key)
    key = args.comparison_key
    score = torch.from_numpy(arrays[f"{key}_structure"]).float()
    masks, labels, final, widths = _mask_pool(comparison, arrays, valid, score, key, args.random_controls, args.width_mode)
    tensor = tensor[None].to(args.device)
    target_pixels = patchify(tensor, 14)[0].cpu()
    teacher_tokens = modules["teacher"].forward_features(tensor)["x_norm_patchtokens"][0]
    teacher_probs = ((modules["teacher_head"](teacher_tokens).float() - center) / bundle["teacher_temp"]).softmax(-1)
    teacher_tokens = teacher_tokens.cpu()
    unmasked_features = modules["student"].forward_features(tensor)["x_norm_patchtokens"]
    unmasked_prediction = modules["decoder"](unmasked_features)[0].cpu()
    unmasked_logits = modules["student_head"](unmasked_features[0])
    destination = args.output_dir / relative
    destination.mkdir(parents=True, exist_ok=False)
    display.save(destination / "input.png")
    saved = {"valid": valid.numpy(), "structure": score.numpy(), "target_pixels": target_pixels.numpy(),
             "unmasked_prediction": unmasked_prediction.numpy()}
    rows, entries = [], []

    def evaluate(target, input_mask, name, prefix, prediction=None, logits=None, compute_context=True):
        ids = target.flatten().nonzero().flatten()
        if prediction is None:
            features = modules["student"].forward_features(tensor, masks=input_mask.flatten()[None].to(args.device))["x_norm_patchtokens"]
            prediction = modules["decoder"](features)[0].cpu()
            logits = modules["student_head"](features[0, ids.to(args.device)])
        errors = patch_errors(prediction[ids], target_pixels[ids])
        proto = prototype_errors(logits, teacher_probs[ids.to(args.device)], bundle["student_temp"])
        errors.update({key: value.cpu() for key, value in proto.items()})
        metrics = {key: float(value.mean()) for key, value in errors.items()}
        for radius in ((3, 5) if compute_context else ()):
            pmap, stats = context_score(teacher_tokens, target, input_mask, valid, radius, args.context_threshold)
            for field in ("P", "P_supported", "H", "missing_fraction"):
                metrics[f"{field}_r{radius}"] = stats[field]
            saved[f"{prefix}_P_r{radius}"] = pmap.numpy()
        error_map = np.full(valid.shape, np.nan, dtype=np.float32)
        for error, values in errors.items():
            grid = np.full(valid.shape, np.nan, dtype=np.float32)
            grid[target.numpy()] = values.numpy()
            saved[f"{prefix}_{error}"] = grid
            if error == "pixel_mse":
                error_map = grid
        saved[f"{prefix}_target"] = target.numpy()
        saved[f"{prefix}_input_mask"] = input_mask.numpy()
        saved[f"{prefix}_prediction_on_target"] = prediction[ids].numpy()
        return {"name": name, "target": target, "input_mask": input_mask, "prediction": prediction,
                "error_map": error_map, "metrics": metrics}

    for index, (mask, names) in enumerate(zip(masks, labels)):
        prefix = f"mask_{index:02d}"
        entry = evaluate(mask, mask, "; ".join(names), prefix)
        reference = patch_errors(unmasked_prediction[mask.flatten()], target_pixels[mask.flatten()])
        reference_proto = prototype_errors(unmasked_logits[mask.flatten().to(args.device)], teacher_probs[mask.flatten().to(args.device)], bundle["student_temp"])
        entry["metrics"].update({f"unmasked_{key}": float(value.mean()) for key, value in {**reference, **reference_proto}.items()})
        baseline = visible_baselines(target_pixels, mask, mask, valid)
        for name, prediction in baseline.items():
            entry["metrics"].update({f"{name}_{key}": float(value.mean()) for key, value in patch_errors(prediction, target_pixels[mask.flatten()]).items()})
        row = {"case": str(relative), "image": record["image"], "mask_index": index, "labels": ";".join(names),
               "target_count": int(mask.sum()), "input_mask_count": int(mask.sum()), "valid_count": int(valid.sum()),
               "A": float(score[mask].mean()), "target_variance": float(target_pixels[mask.flatten()].var(unbiased=False)),
               **entry["metrics"]}
        rows.append(row)
        entries.append(entry)
        print(f"  {Path(record['image']).name}: mask {index + 1}/{len(masks)} MSE={row['pixel_mse']:.4f}", flush=True)
    normal = entries[final]
    if verify_invariance:
        write_json(args.output_dir / "hidden_pixel_invariance.json",
                   verify_hidden_pixel_invariance(modules, tensor, masks[final], normal["prediction"]))
    intervention_rows = [{"case": str(relative), "condition": "normal", "extra_count": 0,
                          "target_count": int(masks[final].sum()), "input_mask_count": int(masks[final].sum()), **normal["metrics"]}]
    near, far, intervention_meta = context_interventions(masks[final], valid, args.near_radius, args.far_distance,
                                                        args.far_repeats, comparison["seed"])
    context_entries = [normal]
    for name, input_mask in [("remove_near", near)] + [(f"remove_far_{index}", mask) for index, mask in enumerate(far)]:
        entry = evaluate(masks[final], input_mask, name, name)
        context_entries.append(entry)
        intervention_rows.append({"case": str(relative), "condition": name, "extra_count": intervention_meta["extra_count"],
                                  "target_count": int(masks[final].sum()), "input_mask_count": int(input_mask.sum()), **entry["metrics"]})
    reference = evaluate(masks[final], masks[final], "Unmasked Student decoder reference (sees target pixels)", "unmasked_ref",
                         unmasked_prediction, unmasked_logits[masks[final].flatten().to(args.device)], compute_context=False)
    reference["input_mask"] = torch.zeros_like(valid)
    saved["unmasked_ref_input_mask"] = np.zeros(valid.shape, dtype=bool)
    context_entries.append(reference)
    chosen = [0]
    for index, names in enumerate(labels):
        if any("MAX_A" in name for name in names) and index not in chosen:
            chosen.append(index)
    _render(destination / "reconstruction.png", Path(record["image"]).name + " | highest-A path, width comparison", display, target_pixels, [entries[index] for index in chosen])
    _render(destination / "context_ablation.png", Path(record["image"]).name + " | same target under context intervention", display, target_pixels, context_entries)
    _render_detail(destination / "target_detail.png", display, target_pixels, [normal, context_entries[1], context_entries[2], reference])
    _render(destination / "all_candidates.png", Path(record["image"]).name + " | all evaluated masks", display, target_pixels, entries)
    np.savez_compressed(destination / "raw_results.npz", **saved)
    write_json(destination / "metadata.json", {"source_case": str(args.from_run / relative), "structure_case": metadata["source_case"],
        "image": record["image"], "weights": record["weights"], "comparison_key": key, "selection": "highest A; never P or error",
        "width_mode": args.width_mode, "selection_fallback": {"used": comparison["used_fallback"], "reasons": comparison["fallback_reasons"]},
        "source_best_index": comparison["best_index"], "selected_mask_index": final, "widths": widths,
        "context_intervention": intervention_meta, "labels": labels, "teacher_temperature": bundle["teacher_temp"],
        "student_temperature": bundle["student_temp"], "metrics": rows, "interventions": intervention_rows})
    write_csv(destination / "metrics.csv", rows)
    write_csv(destination / "interventions.csv", intervention_rows)
    return rows, intervention_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-key", default="clean_layer_12_ratio_00")
    parser.add_argument("--width-mode", choices=("fixed", "image", "local"), default="local")
    parser.add_argument("--random-controls", type=int, default=2)
    parser.add_argument("--context-threshold", type=float, default=0.7)
    parser.add_argument("--near-radius", type=int, default=2)
    parser.add_argument("--far-distance", type=int, default=6)
    parser.add_argument("--far-repeats", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.random_controls < 1 or args.far_repeats < 1 or args.cpu_threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("counts must be positive")
    if args.near_radius < 1 or args.far_distance <= args.near_radius or not -1 <= args.context_threshold <= 1:
        parser.error("invalid context geometry or threshold")
    args.from_run, args.output_dir, args.model_bundle = [path.expanduser().resolve() for path in (args.from_run, args.output_dir, args.model_bundle)]
    if args.output_dir.exists():
        raise FileExistsError(f"choose a new output directory: {args.output_dir}")
    torch.set_num_threads(args.cpu_threads)
    args.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    torch.manual_seed(0)
    modules, center, bundle = load_bundle(args.model_bundle, args.device)
    verified_teacher = bundle["provenance"]["verified_teacher_backbone"]
    if not bundle["provenance"]["teacher_exact_match"] or not verified_teacher:
        raise ValueError("export bundle with --verify-teacher-backbone before comparing cached candidates")
    records = json.loads((args.from_run / "summary.json").read_text())["completed"]
    records = [record for record in records if Path(record["weights"]).resolve() == Path(verified_teacher).resolve()]
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError("no cached cases use the verified Teacher in this bundle")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "run_config.json", {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model_provenance": bundle["provenance"], "iteration": bundle["iteration"], "training_modified": False,
        "caveats": ["Unmasked Teacher features include global hidden-image context; P is a heuristic.",
                    "Unmasked decoder reference sees target pixels, and is not a context predictor.",
                    "F025 pixel-trained decoder; Haar is a diagnostic, not a trained Haar objective.",
                    "Near/far interventions match count, not geometry or semantic content.",
                    "Development images: no held-out generalization or downstream training claim."]})
    rows, interventions, completed = [], [], []
    for record in records:
        print(f"CASE {len(completed) + 1}/{len(records)}: {record['image']}", flush=True)
        local, context = run_case(args, record, modules, center, bundle, verify_invariance=not completed)
        rows.extend(local)
        interventions.extend(context)
        completed.append(record)
        write_csv(args.output_dir / "metrics.csv", rows)
        write_csv(args.output_dir / "interventions.csv", interventions)
        write_json(args.output_dir / "progress.json", {"completed": completed, "planned_count": len(records)})
    correlations = _correlations(rows)
    aggregate = _summarize(rows, interventions, correlations)
    write_csv(args.output_dir / "correlations.csv", correlations)
    write_json(args.output_dir / "summary.json", {"completed": completed, "image_count": len(completed),
        "unique_mask_count": len(rows), "intervention_count": len(interventions), "aggregate": aggregate})
    _scatter(args.output_dir / "P_vs_error.png", rows)
    render_overview(args.output_dir, completed)
    lines = ["# 第三步：上下文评分与实测恢复", "", "这是冻结 F025 的离线诊断，最高 A 的选择规则保持不变。", "",
             f"完成 {len(completed)} 张开发图、{len(rows)} 个去重候选。下列统计按图平均；评分越高误差越低应表现为负相关。", "",
             "|误差|同预算 P5–误差 Spearman 均值|负相关图片|移除附近–移除远处平均误差差值|附近移除更差图片|", "|---|---:|---:|---:|---:|"]
    for error, stats in aggregate.items():
        corr = stats['macro_matched_P5_spearman']
        lines.append(f"|{error}|{corr:.4f}" if corr is not None else f"|{error}|无法计算")
        lines[-1] += f"|{stats['negative_correlation_images']}/{stats['correlation_images']}|{stats['macro_near_minus_far']:.4f}|{stats['near_worse_than_far_images']}/{len(completed)}|"
    lines += ["", "先看每张图的 reconstruction.png 与 context_ablation.png；红色是固定评价目标，蓝色是额外隐藏上下文。", "",
              "拼接图只在红色目标内放入模型输出；目标外复制原图。未遮挡 decoder 参照能看到目标像素。", "",
              "P 使用完整图像 Teacher 的特征，存在全局上下文信息；相关性只检验该启发式的预测能力。H 阈值 0.7 没有校准。", "",
              "Haar 误差用于检查细节；本模型只接受过像素重建训练。开发图不能证明泛化或训练收益。", "",
              "详细数据：metrics.csv、interventions.csv、correlations.csv、各图 raw_results.npz。"]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output_dir), "images": len(completed), "masks": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
