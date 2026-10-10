"""Small PIL-only training snapshots of actual mixed masks and visible ends."""

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw


@torch.no_grad()
def save_structure_mask_grid(images, examples, *, output_path, iteration, probability, max_views=2):
    examples = examples[:max_views]
    if not examples:
        return
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    height, width = images.shape[-2:]
    panel_width, header = max(width, 330), 66
    canvas = Image.new("RGB", (panel_width * 5, (height + header) * len(examples)), "white")
    draw = ImageDraw.Draw(canvas)
    metadata = []
    for row, example in enumerate(examples):
        rgb = (images[example["anchor_index"]].detach().float().cpu() * std + mean).clamp(0, 1)
        h, w = example["valid"].shape
        if height % h or width % w:
            raise ValueError("visualization geometry does not match training input")
        def expand(grid):
            return grid.repeat_interleave(height // h, 0).repeat_interleave(width // w, 1)
        def overlay(base, mask, color, alpha=0.55):
            selected = expand(mask).bool()
            out = base.clone()
            out[:, selected] = out[:, selected] * (1 - alpha) + out.new_tensor(color)[:, None] * alpha
            return out
        ribbon = example["ribbon"]
        score = expand(example["score"])
        heat = torch.stack((score, score * 0.6, 1 - score))
        score_rgb = torch.where(expand(example["valid"])[None], rgb * 0.5 + heat * 0.5, rgb)
        baseline_rgb = overlay(rgb, example["baseline"], (1, 0.12, 0.02))
        core_rgb = overlay(rgb, ribbon.core, (1, 0.12, 0.02))
        core_rgb = overlay(core_rgb, ribbon.context, (0.15, 0.85, 0.25))
        full_rgb = overlay(core_rgb, ribbon.supplement, (0.1, 0.45, 1.0))
        diag = ribbon.diagnostics
        state = f"A={diag['final_core_A']:.3f}; width={diag['width_mean']:.2f}" if diag["applied"] else "FALLBACK: " + diag["reason"]
        titles = [f"Anchor #{example['anchor_index']} | step {iteration + 1}\np(structure)={probability:.3f}",
                  "Online Teacher structure S\nFeature variation; no anatomy labels",
                  f"Original random block\nN={int(example['baseline'].sum())}",
                  "Red=hidden core; green=visible ends\n" + state,
                  f"Actual FULL mask: red + blue\nN={int(ribbon.mask.sum())}; unchanged budget"]
        for column, (pixels, title) in enumerate(zip((rgb, score_rgb, baseline_rgb, core_rgb, full_rgb), titles)):
            x, y = column * panel_width, row * (height + header)
            draw.multiline_text((x + 5, y + 8), title, fill="black", spacing=4)
            tile = Image.fromarray((pixels.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
            canvas.paste(tile, (x, y + header))
        metadata.append({"anchor_index": example["anchor_index"], **diag})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    output_path.with_suffix(".json").write_text(json.dumps({"iteration": iteration, "probability": probability, "examples": metadata}, indent=2) + "\n")
