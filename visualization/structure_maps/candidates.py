"""Step 2: compare equal-budget masks using saved step-1 structure maps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
upstream = str(REPO_ROOT / "upstream" / "dinov2-main")
if upstream not in sys.path:
    sys.path.insert(0, upstream)

from methods.geotopo_dino.data.collate import _block_mask_within_valid
from methods.geotopo_dino.masking.candidate_mask_sampler import compare_candidate_masks
from methods.geotopo_dino.masking.slender_mask_sampler import compare_slender_masks
from visualization.dino_spine_maps.visualize import _layers
from visualization.spine_masks.visualize import seeded_cpu_random


def _read_case(source, record):
    relative = Path(record["output"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("cached output must be a relative case path")
    origin = (source / relative).resolve()
    if not origin.is_relative_to(source):
        raise ValueError("cached case must remain inside source run")
    metadata = json.loads((origin / "metadata.json").read_text())
    with np.load(origin / "raw_maps.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return relative, origin, metadata, arrays


def _expand(grid, image):
    h, w = grid.shape
    if image.height % h or image.width % w:
        raise ValueError("cached input size does not match patch grid")
    return np.repeat(np.repeat(grid, image.height // h, 0), image.width // w, 1)


def _draw_input(ax, image, title):
    ax.imshow(image)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _draw_mask(ax, image, mask, title, supplement=None, context=None):
    _draw_input(ax, image, title)
    expanded = _expand(mask.numpy(), image)
    rgba = np.zeros((*expanded.shape, 4), dtype=np.float32)
    rgba[..., :3] = (1.0, 0.12, 0.02)
    rgba[..., 3] = expanded * 0.55
    ax.imshow(rgba, interpolation="nearest")
    for extra, color in ((supplement, (0.1, 0.45, 1.0)), (context, (0.15, 0.85, 0.25))):
        if extra is not None:
            layer = np.zeros_like(rgba)
            layer[..., :3] = color
            layer[..., 3] = _expand(extra.numpy(), image) * 0.5
            ax.imshow(layer, interpolation="nearest")


def _draw_score(ax, image, score, valid, title):
    _draw_input(ax, image, title)
    values = _expand(score.numpy(), image)
    visible = _expand(valid.numpy(), image)
    ax.imshow(np.ma.array(values, mask=~visible), vmin=0, vmax=1,
              cmap="magma", alpha=0.55, interpolation="nearest")


def _roles(result, index):
    return ",".join(name for name, selected in (("base", 0), ("random", result.random_index),
        ("selected", result.adaptive_index), ("best", result.best_index)) if selected == index)


def _render(destination, panels, image_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slender = panels[0]["result"].core_masks is not None
    full_column = slender and panels[0].get("budget_mode", "mixed") == "mixed"
    fig, axes = plt.subplots(len(panels), 5 if full_column else 4,
                            figsize=(19 if full_column else 16, 4.2 * len(panels)), squeeze=False)
    for row, panel in enumerate(panels):
        result, image = panel["result"], panel["image"]
        effective = result.diagnostics['target_count'] / max(result.diagnostics['valid_count'], 1)
        descriptor = f"{panel['mode']} L{panel['layer']} effective r={effective:.2f}"
        _draw_input(axes[row, 0], image, f"{descriptor}\n{result.diagnostics['target_count']} masked patches")
        _draw_score(axes[row, 1], image, panel["score"], panel["valid"], "Structure S overlay")
        rule = "MAX-A" if result.diagnostics.get("selection_mode") == "best" else "softmax"
        selected_title = f"Selected {rule} core" if slender else f"Selected {rule} mask"
        selections = [("Original block", 0), (selected_title, result.adaptive_index)]
        if full_column:
            selections.append((f"Selected {rule} FULL mask", result.adaptive_index))
        for col, (name, index) in enumerate(selections, start=2):
            stats = result.diagnostics["candidate_geometry"][index]
            enrichment = stats["score_enrichment"]
            extra = f"; x{enrichment:.2f}" if enrichment is not None else ""
            title = f"{name}: #{index} {result.kinds[index]}\nA={float(result.scores[index]):.3f}{extra}"
            if slender and index:
                title += f"; core={stats['core_count']}"
            if col > 2 and result.diagnostics["used_fallback"]:
                title += "\nfallback to original block"
            if slender and index:
                _draw_mask(axes[row, col], image, result.core_masks[index], title,
                           result.supplement_masks[index] if name.endswith("FULL mask") else None,
                           result.context_masks[index])
            else:
                _draw_mask(axes[row, col], image, result.masks[index], title)
        cols = 3
        candidate_fig, candidate_axes = plt.subplots((len(result.kinds) + cols) // cols, cols,
            figsize=(12, 4 * ((len(result.kinds) + cols) // cols)), squeeze=False)
        _draw_score(candidate_axes.flat[0], image, panel["score"], panel["valid"], descriptor + "\nStructure S")
        for index, ax in enumerate(list(candidate_axes.flat)[1:len(result.kinds) + 1]):
            stats = result.diagnostics["candidate_geometry"][index]
            title = f"#{index}: {result.kinds[index]} [{_roles(result, index)}]\nA={float(result.scores[index]):.3f}; p={float(result.probabilities[index]):.3f}; components={stats['component_count']}"
            if slender and index:
                _draw_mask(ax, image, result.core_masks[index], title,
                           result.supplement_masks[index], result.context_masks[index])
            else:
                _draw_mask(ax, image, result.masks[index], title)
        for ax in list(candidate_axes.flat)[len(result.kinds) + 1:]:
            ax.axis("off")
        legend = "Red=thin core; blue=supplement; green=visible ends. A scores core; p=selection probability." if slender else "Same budget for every candidate. Red=masked; A=mean S, not anatomy accuracy; p=selection probability."
        candidate_fig.suptitle(f"{image_name} | {descriptor}\n{legend}")
        candidate_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90,
                                     wspace=0.12, hspace=0.32)
        candidate_fig.savefig(destination / f"candidates_{panel['key']}.png", dpi=130, bbox_inches="tight")
        plt.close(candidate_fig)
    legend = "Red=thin core; blue=supplement; green=visible ends. Baseline A is full-mask reference." if slender else "Red=masked; selected mask uses the configured rule. A is not anatomy accuracy."
    fig.suptitle(f"{image_name} | {destination.name}\n{legend}")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.80 if len(panels) == 1 else 0.90,
                        wspace=0.08, hspace=0.32)
    fig.savefig(destination / "comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _overview(root, completed):
    for start in range(0, len(completed), 6):
        panels = []
        for record in completed[start:start + 6]:
            with Image.open(root / record["output"] / "comparison.png") as opened:
                image = opened.convert("RGB")
                image.thumbnail((1800, 900))
                panels.append(image.copy())
        sheet = Image.new("RGB", (max(p.width for p in panels), sum(p.height for p in panels)), "white")
        y = 0
        for panel in panels:
            sheet.paste(panel, (0, y))
            y += panel.height
        sheet.save(root / f"overview_{start // 6 + 1:03d}.jpg", quality=90)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path, required=True, help="Step-1 structure-map output directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--layers", default="12", help="Cached one-based layers: 12, 8,12, all")
    parser.add_argument("--anchor-mode", choices=("clean", "train", "both"), default="both")
    parser.add_argument("--mask-ratios", type=float, nargs="+", default=[0.4], help="Fractions of valid patches")
    parser.add_argument("--num-candidates", type=int, default=8, help="Includes original block as candidate 0")
    parser.add_argument("--selection-mode", choices=("best", "softmax"), default="best", help="Select highest mean S; softmax only reproduces historical selection")
    parser.add_argument("--temperature", type=float, default=0.1, help="Used only with --selection-mode softmax")
    parser.add_argument("--candidate-style", choices=("slender", "compact"), default="slender",
                        help="Width-limited cores with supplements, or the original v1 compact proposals")
    parser.add_argument("--budget-mode", choices=("core-only", "mixed"), default="mixed",
                        help="Core-only matches baseline to actual thin-core count; mixed keeps requested total budget")
    parser.add_argument("--strip-width", type=int, default=5, help="Odd core width in patches; never expanded to fill total budget")
    parser.add_argument("--span-fraction", type=float, default=0.55, help="Core length / supported long-FOV extent")
    parser.add_argument("--context-rows", type=int, default=2, help="Reserved visible path positions on both ends")
    parser.add_argument("--path-axis", choices=("auto", "vertical", "horizontal"), default="auto",
                        help="Auto uses valid-FOV long axis; explicit geometry prior, not anatomy detection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Maximum image/checkpoint cases; 0 means all")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.num_candidates < 2 or args.limit < 0 or not np.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("invalid num-candidates, limit or temperature")
    if not all(np.isfinite(r) and 0 <= r <= 1 for r in args.mask_ratios) or len(set(args.mask_ratios)) != len(args.mask_ratios):
        raise ValueError("mask-ratios must be unique finite values in [0,1]")
    if args.strip_width < 1 or args.strip_width % 2 != 1 or args.context_rows < 1 or not 0 < args.span_fraction < 1:
        raise ValueError("strip-width must be positive odd, context-rows positive, span-fraction in (0,1)")
    if args.candidate_style == "compact" and args.budget_mode != "mixed":
        raise ValueError("core-only budget requires slender candidates")
    source, root = args.from_run.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {root}")
    if root.is_relative_to(source):
        raise ValueError("output must be outside the cached step-1 run")
    records = json.loads((source / "summary.json").read_text())["completed"]
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError("no successful cached cases")
    torch.set_num_threads(4)
    root.mkdir(parents=True)
    config = {"algorithm_version": 3, "selection_mode": args.selection_mode, "source_run": str(source),
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "score": "A=mean(S in thin core); baseline reference is full-mask mean" if args.candidate_style == "slender" else "A(M)=mean(S_i for i in M)", "model_inference_performed": False,
              "training_integration": False, "context_scoring": False,
              "proposal": "random and feature-guided width-limited paths; separate supplements, protected ends" if args.candidate_style == "slender" else "score-independent connected growth with random orientation; no isolated budget supplement",
              "min_candidate_score_spread": 1e-4,
              "note": "Selection gains on A are by construction; they do not establish anatomical accuracy or downstream benefit."}
    (root / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    completed, failures, rows = [], [], []

    def save_summary():
        (root / "summary.json").write_text(json.dumps({"completed": completed, "failures": failures,
            "success_count": len(completed), "failure_count": len(failures),
            "comparison_count": sum(r["comparison_count"] for r in completed)}, indent=2) + "\n")
        if rows:
            with (root / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    for record in records:
        print(record["output"], flush=True)
        try:
            relative, origin, previous, arrays = _read_case(source, record)
            layers = previous["layers"] if args.layers == "all" else [i + 1 for i in _layers(args.layers, max(previous["layers"]))]
            if any(layer not in previous["layers"] for layer in layers):
                raise ValueError(f"requested layer absent from cache; available: {previous['layers']}")
            modes = [m for m in ("clean", "train") if f"{m}_valid" in arrays and args.anchor_mode in ("both", m)]
            if not modes:
                raise ValueError("requested input mode absent from cache")
            panels, case_rows, selections, output_arrays = [], [], {}, {}
            for mode in modes:
                with Image.open(origin / f"input_{mode}.png") as opened:
                    image = opened.convert("RGB")
                valid = torch.from_numpy(arrays[f"{mode}_valid"]).bool()
                output_arrays[f"{mode}_valid"] = valid.numpy()
                for ratio_index, ratio in enumerate(args.mask_ratios):
                    # Seed ignores checkpoint/layer, so proposals are matched across them.
                    seed = (previous["seed"] + args.seed + (10000 if mode == "train" else 0) + round(ratio * 1000000)) % (2**32)
                    with seeded_cpu_random(seed):
                        baseline = _block_mask_within_valid(valid, int(int(valid.sum()) * ratio))
                    for layer in layers:
                        current_baseline = baseline
                        prefix = f"{mode}_layer_{layer:02d}"
                        score = torch.from_numpy(arrays[f"{prefix}_score"])
                        options = dict(seed=seed, num_candidates=args.num_candidates, temperature=args.temperature,
                                       near_constant=previous["metrics"][prefix]["near_constant"], selection_mode=args.selection_mode)
                        if args.candidate_style == "slender":
                            comparison = compare_slender_masks(score, valid, baseline, **options,
                                strip_width=args.strip_width, span_fraction=args.span_fraction,
                                context_rows=args.context_rows, axis=args.path_axis)
                            if args.budget_mode == "core-only" and comparison.diagnostics["core_target_count"] > 0:
                                # A ribbon defines its own feasible area; match the control
                                # to that area rather than widening the ribbon to fill 40%.
                                with seeded_cpu_random(seed):
                                    current_baseline = _block_mask_within_valid(valid, comparison.diagnostics["core_target_count"])
                                comparison = compare_slender_masks(score, valid, current_baseline, **options,
                                    strip_width=args.strip_width, span_fraction=args.span_fraction,
                                    context_rows=args.context_rows, axis=args.path_axis)
                        else:
                            comparison = compare_candidate_masks(score, valid, baseline, **options)
                        key = f"{mode}_layer_{layer:02d}_ratio_{ratio_index:02d}"
                        panels.append({"mode": mode, "layer": layer, "ratio": ratio, "key": key,
                            "image": image, "valid": valid, "score": score, "result": comparison, "budget_mode": args.budget_mode})
                        for name, value in (("masks", comparison.masks), ("structure", score),
                                            ("scores", comparison.scores), ("probabilities", comparison.probabilities)):
                            output_arrays[f"{key}_{name}"] = value.cpu().numpy()
                        output_arrays[f"{key}_final_mask"] = comparison.masks[comparison.adaptive_index].cpu().numpy()
                        if comparison.core_masks is not None:
                            for name, value in (("cores", comparison.core_masks), ("supplements", comparison.supplement_masks),
                                                ("contexts", comparison.context_masks)):
                                output_arrays[f"{key}_{name}"] = value.cpu().numpy()
                        selections[key] = {"mode": mode, "layer": layer, "mask_ratio": ratio,
                            "effective_mask_ratio": comparison.diagnostics["target_count"] / max(int(valid.sum()), 1),
                            "budget_mode": args.budget_mode, "seed": seed,
                            "kinds": comparison.kinds, "scores": comparison.scores.tolist(),
                            "probabilities": comparison.probabilities.tolist(), "random_index": comparison.random_index,
                            "adaptive_index": comparison.adaptive_index, "best_index": comparison.best_index,
                            **comparison.diagnostics}
                        for index, stats in enumerate(comparison.diagnostics["candidate_geometry"]):
                            raw = arrays[f"{prefix}_difference"][comparison.masks[index].numpy()]
                            case_rows.append({"image": record["image"], "weights": record["weights"], "mode": mode,
                                "layer": layer, "mask_ratio": ratio,
                                "effective_mask_ratio": comparison.diagnostics["target_count"] / max(int(valid.sum()), 1),
                                "budget_mode": args.budget_mode, "candidate_index": index, "kind": comparison.kinds[index],
                                "selection_roles": _roles(comparison, index), "probability": float(comparison.probabilities[index]),
                                "raw_difference_mean": float(raw.mean()) if raw.size else 0.0,
                                **stats, "selection_fallback": ",".join(comparison.diagnostics["fallback_reasons"]),
                                "output": str(relative)})
            destination = root / relative
            destination.mkdir(parents=True)
            _render(destination, panels, Path(record["image"]).name)
            np.savez_compressed(destination / "candidate_masks.npz", **output_arrays)
            metadata = {"source_case": str(origin), "image": record["image"], "weights": record["weights"],
                        "geometry": previous["geometry"], "comparisons": selections}
            (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            completed.append({**record, "comparison_count": len(panels)})
            rows.extend(case_rows)
        except Exception as exc:
            failures.append({"output": record["output"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED: {exc}", file=sys.stderr, flush=True)
            save_summary()
            if args.fail_fast:
                raise
        save_summary()
    if completed:
        _overview(root, completed)
    print(f"Saved {len(completed)} cases, {sum(r['comparison_count'] for r in completed)} comparisons; failed {len(failures)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
