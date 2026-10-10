"""Compare ribbon widths on the highest-scoring cached path and export a mask."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from methods.geotopo_dino.masking.adaptive_ribbon import AdaptiveRibbon, adapt_ribbon_width
from visualization.structure_maps.candidates import _draw_input, _draw_mask, _draw_score, _overview


def _width_plot(ax, results, span, context_rows, axis):
    image_result, local_result = results["image"], results["local"]
    begin, end = local_result.metrics["active_start_stop"]
    positions = np.arange(begin, end)
    ax.plot(positions, local_result.raw_widths.numpy(), ":", color="0.55", label="Detected support")
    ax.plot(positions, results["fixed"].widths.numpy(), "--", color="tab:blue", label="Fixed")
    ax.plot(positions, image_result.widths.numpy(), color="tab:orange", label="Image adaptive")
    ax.plot(positions, local_result.widths.numpy(), color="tab:red", drawstyle="steps-mid", label="Local adaptive")
    weak = ~local_result.reliable.numpy()
    if weak.any():
        ax.scatter(positions[weak], local_result.widths.numpy()[weak], color="black", marker="x", label="Weak: fallback")
    ax.axvspan(span[0] - 0.5, span[1] - 0.5, color="red", alpha=0.06)
    ax.set_xlabel("Row along path" if axis == "vertical" else "Column along path")
    ax.set_ylabel("Width (patches)")
    ax.set_yticks(sorted(set(local_result.widths.tolist() + image_result.widths.tolist() + results["fixed"].widths.tolist())))
    ax.set_title("6. Width profile\nShaded = masked span", fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="best")


def _fallback(core, score, valid, reason):
    empty = torch.empty(0, dtype=torch.long)
    metrics = {"mode": "fallback", "active_start_stop": None, "masked_start_stop": None,
        "width_min": None, "width_max": None, "width_mean": None, "width_values": [],
        "reliable_rows": 0, "masked_rows": 0, "weak_response_rows": 0, "max_width_bound_rows": 0,
        "core_count": int(core.sum()), "context_count": 0,
        "core_structure_mean": float(score[core].mean()) if core.any() else 0.0,
        "effective_mask_ratio": int(core.sum()) / max(int(valid.sum()), 1),
        "width_is_anatomy_estimate": False, "fallback_reason": reason}
    return AdaptiveRibbon(core, torch.zeros_like(core), empty, empty, empty.bool(), empty.float(), metrics)


def _render(destination, panels, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(panels), 6, figsize=(23, 4.7 * len(panels)), squeeze=False)
    for row, panel in enumerate(panels):
        image, results = panel["image"], panel["results"]
        source = f"{panel['roles']} path #{panel['index']} | {panel['mode']} L{panel['layer']}"
        _draw_input(axes[row, 0], image, f"1. Input\n{source}")
        _draw_score(axes[row, 1], image, panel["score"], panel["valid"], "2. Structure S\nFeature variation, not anatomy probability")
        if panel.get("fallback_reason"):
            _draw_mask(axes[row, 2], image, results["fixed"].core, "3. Fallback: original block")
            for ax in axes[row, 3:]:
                ax.axis("off")
            axes[row, 4].text(0.5, 0.5, "Width adaptation skipped\n" + panel["fallback_reason"], ha="center", va="center", transform=axes[row, 4].transAxes)
            continue
        for col, mode, label in ((2, "fixed", "3. Fixed width"), (3, "image", "4. Image adaptive"), (4, "local", "5. Local adaptive")):
            result = results[mode]
            m = result.metrics
            width = str(m["width_min"]) if m["width_min"] == m["width_max"] else f"{m['width_min']}..{m['width_max']}"
            title = f"{label}: {width} patches\nN={m['core_count']}; r={m['effective_mask_ratio']:.3f}; A={m['core_structure_mean']:.3f}"
            _draw_mask(axes[row, col], image, result.core, title, context=result.context)
        _width_plot(axes[row, 5], results, panel["span"], panel["context_rows"], panel["axis"])
    fig.suptitle(f"{name} | {destination.name}\nHighest-A path; same centerline and span. RED=masked; GREEN=visible ends; no blue supplements.\nWidths have different areas; no re-ranking. Exported final width mode: {panels[0]['final_width_mode']}.", fontsize=11)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.11 if len(panels) == 1 else 0.065,
                        top=0.76 if len(panels) == 1 else 0.87, wspace=0.22, hspace=0.4)
    fig.savefig(destination / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.7 * len(panels)), squeeze=False)
    for row, panel in enumerate(panels):
        if panel.get("fallback_reason"):
            axes[row, 0].text(0.5, 0.5, "Fallback to original block; no width profile\n" + panel["fallback_reason"], ha="center", va="center")
            axes[row, 0].axis("off")
            continue
        _width_plot(axes[row, 0], panel["results"], panel["span"], panel["context_rows"], panel["axis"])
        axes[row, 0].set_title(f"{name} | {panel['roles']} #{panel['index']} | {panel['mode']} L{panel['layer']}")
    fig.tight_layout()
    fig.savefig(destination / "width_profiles.png", dpi=150)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path, required=True, help="Existing slender candidate run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-width", type=int, default=3)
    parser.add_argument("--max-width", type=int, default=9)
    parser.add_argument("--relative-threshold", type=float, default=0.5)
    parser.add_argument("--min-prominence", type=float, default=0.08)
    parser.add_argument("--smooth-penalty", type=float, default=0.75)
    parser.add_argument("--width-mode", choices=("fixed", "image", "local"), default="local", help="Width mode exported as final_mask; all three are visualized")
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    if args.min_width < 1 or args.max_width < args.min_width or args.min_width % 2 != 1 or args.max_width % 2 != 1:
        raise ValueError("width bounds must be positive odd integers")
    if not 0 < args.relative_threshold < 1 or not np.isfinite(args.min_prominence) or args.min_prominence <= 0:
        raise ValueError("invalid response thresholds")
    if not np.isfinite(args.smooth_penalty) or args.smooth_penalty < 0:
        raise ValueError("invalid smoothing penalty")
    source, root = args.from_run.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if root.exists() or root.is_relative_to(source):
        raise ValueError("output must be new and outside the source run")
    records = json.loads((source / "summary.json").read_text())["completed"]
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError("no cached candidates")
    torch.set_num_threads(4)
    root.mkdir(parents=True)
    manifest = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_run": str(source), "model_inference_performed": False, "training_integration": False,
        "selection_mode": "best", "final_width_mode": args.width_mode,
        "comparison": "highest-A original path only; hold centerline and span fixed; compare fixed, image and local widths",
        "width_evidence": "contiguous transverse S response above local background + relative_threshold * prominence",
        "background": "median valid samples 1..3 patches beyond maximum band on both sides; row median if unavailable",
        "weak_response": "center prominence < min_prominence; bounded fixed-width fallback; local mode enforces it",
        "smoothing": "minimize width evidence mismatch + smooth_penalty * adjacent width change; maximum step 2",
        "anatomy_probability": False, "equal_mask_budget": False, "supplementary_blocks": False,
        "candidate_reselection": False,
        "note": "Width adapts to S response, not vertebra count. A is descriptive and biased by changing mask size.",
    }
    (root / "run_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    completed, failures, rows = [], [], []
    for record in records:
        relative = Path(record["output"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe cached relative path")
        try:
            case = source / relative
            metadata = json.loads((case / "metadata.json").read_text())
            origin = Path(metadata["source_case"])
            arrays_out, panels, comparisons, case_rows = {}, [], {}, []
            with np.load(origin / "raw_maps.npz", allow_pickle=False) as maps, np.load(case / "candidate_masks.npz", allow_pickle=False) as originals:
                for key, selection in metadata["comparisons"].items():
                    mode, layer = selection["mode"], selection["layer"]
                    score = torch.from_numpy(maps[f"{mode}_layer_{layer:02d}_score"].copy())
                    valid = torch.from_numpy(maps[f"{mode}_valid"].copy()).bool()
                    with Image.open(origin / f"input_{mode}.png") as opened:
                        image = opened.convert("RGB")
                    indices = [0 if selection["used_fallback"] else selection["best_index"]]
                    for index in indices:
                        geometry = selection["candidate_geometry"][index]
                        if not selection["used_fallback"] and not geometry["path"]:
                            raise ValueError("cached candidate has no path")
                        roles = "Fallback" if selection["used_fallback"] else "Highest-A"
                        reason = ", ".join(selection["fallback_reasons"]) if selection["used_fallback"] else None
                        if reason:
                            start = stop = None
                            core = torch.from_numpy(originals[f"{key}_masks"][0].copy()).bool()
                            results = {name: _fallback(core, score, valid, reason) for name in ("fixed", "image", "local")}
                        else:
                            start, stop = geometry["span_start_stop"]
                            options = dict(path_begin=selection["supported_segment"][0], span_start=start, span_stop=stop,
                                           axis=selection["path_axis"], context_rows=selection["context_rows"],
                                           fallback_width=selection["strip_width"], relative_threshold=args.relative_threshold,
                                           min_prominence=args.min_prominence, smooth_penalty=args.smooth_penalty)
                            fixed_width = selection["strip_width"]
                            results = {
                                "fixed": adapt_ribbon_width(score, valid, geometry["path"], mode="image", min_width=fixed_width, max_width=fixed_width, **options),
                                "image": adapt_ribbon_width(score, valid, geometry["path"], mode="image", min_width=args.min_width, max_width=args.max_width, **options),
                                "local": adapt_ribbon_width(score, valid, geometry["path"], mode="local", min_width=args.min_width, max_width=args.max_width, **options),
                            }
                        if not reason and not np.array_equal(results["fixed"].core.numpy(), originals[f"{key}_cores"][index]):
                            raise ValueError("fixed-width control differs from cached core")
                        entry_key = f"{key}_candidate_{index:02d}"
                        comparisons[entry_key] = {"source_comparison": key, "candidate_index": index, "roles": roles,
                            "path": geometry["path"], "path_axis": selection["path_axis"], "original_score": selection["scores"][index],
                            "span": [start, stop], "selection_mode": "best", "final_width_mode": args.width_mode,
                            "used_fallback": bool(reason), "fallback_reason": reason,
                            "results": {name: result.metrics for name, result in results.items()}}
                        for name, result in results.items():
                            for field in ("core", "context", "widths", "raw_widths", "reliable", "prominence"):
                                arrays_out[f"{entry_key}_{name}_{field}"] = getattr(result, field).numpy()
                            case_rows.append({"image": record["image"], "weights": record["weights"], "source_comparison": key,
                                         "candidate_index": index, "roles": roles, "width_mode": name,
                                         "is_final": name == args.width_mode,
                                         **{k: v for k, v in result.metrics.items() if k != "fallback_reason"}, "fallback_reason": reason})
                        arrays_out[f"{entry_key}_score"] = score.numpy()
                        arrays_out[f"{entry_key}_valid"] = valid.numpy()
                        arrays_out[f"{key}_final_mask"] = results[args.width_mode].core.numpy()
                        arrays_out[f"{key}_final_context"] = results[args.width_mode].context.numpy()
                        arrays_out[f"{key}_final_widths"] = results[args.width_mode].widths.numpy()
                        panels.append(dict(image=image, score=score, valid=valid, results=results, index=index, roles=roles,
                                           mode=mode, layer=layer, span=[start, stop], axis=selection["path_axis"], context_rows=selection["context_rows"],
                                           final_width_mode=args.width_mode, fallback_reason=reason))
            destination = root / relative
            destination.mkdir(parents=True)
            _render(destination, panels, Path(record["image"]).name)
            np.savez_compressed(destination / "adaptive_masks.npz", **arrays_out)
            (destination / "metadata.json").write_text(json.dumps({"source_case": str(case), "image": record["image"],
                "weights": record["weights"], "comparisons": comparisons}, indent=2) + "\n")
            completed.append({**record, "comparison_count": len(panels)})
            rows.extend(case_rows)
        except Exception as exc:
            failures.append({"output": str(relative), "error": str(exc)})
    (root / "summary.json").write_text(json.dumps({"completed": completed, "failures": failures,
        "success_count": len(completed), "failure_count": len(failures)}, indent=2) + "\n")
    if rows:
        with (root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if completed:
        _overview(root, completed)
    print(json.dumps({"output": str(root), "completed": len(completed), "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
