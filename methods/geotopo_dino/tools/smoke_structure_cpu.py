"""Real F025 ViT-B/14 + heads + decoder CPU check, without distributed training.

Loads the existing inference bundle, constructs a mixed mask and performs one
in-memory Student/head/decoder optimizer update on iBOT CE + pixel MSE. Model
weight files are read-only; no trained checkpoint is produced. For the full
CUDA/xFormers/FSDP path use tests.smoke_structure_fsdp instead.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, REPO_ROOT / "upstream" / "dinov2-main"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from methods.geotopo_dino.masking.structure_mask import StructureMaskSettings, StructureMaxAMaskPolicy
from methods.geotopo_dino.masking.packing import pack_masks
from methods.geotopo_dino.losses.pixel_reconstruction_loss import PixelReconstructionLoss
from methods.geotopo_dino.tools.visualize_structure_mask import save_structure_mask_grid
from visualization.structure_maps.reconstruction_model import load_bundle
from visualization.spine_masks.visualize import make_anchor, seeded_cpu_random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "SpinePretrain-v1/spine_dino_dataset/train/spine/buu2000_ap_00000002.jpg")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-mode", choices=("clean", "train"), default="train")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("choose a new CPU smoke output directory")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42)
    modules, center, bundle = load_bundle(args.model_bundle, "cpu")
    from methods.geotopo_dino.data.collate import _block_mask_within_valid
    with Image.open(args.image) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    tensor, geometry, display = make_anchor(image, size=518, mode=args.anchor_mode, seed=42)
    tensor = tensor[None]
    valid = geometry["valid_mask"]
    with seeded_cpu_random(42):
        baseline = _block_mask_within_valid(valid, int(int(valid.sum()) * .4))
    candidates = torch.stack((baseline.flatten(), baseline.flatten()))
    with torch.no_grad():
        teacher_tokens = modules["teacher"].forward_features(tensor)["x_norm_patchtokens"]
        q = ((modules["teacher_head"](teacher_tokens[0]).float() - center) / bundle["teacher_temp"]).softmax(-1)
    # Force application for a single image; proposal/width/budget rules are unchanged.
    settings = StructureMaskSettings(max_probability=1.0, visualization_period=1, context_log_period=1)
    policy = StructureMaxAMaskPolicy(settings)
    checks = []
    for iteration in (0, 9000, 12000):
        started = time.perf_counter()
        state, py_state = torch.random.get_rng_state().clone(), __import__("random").getstate()
        final, _ = policy.select(candidates, teacher_anchor_tokens=teacher_tokens,
            anchor_valid_mask=valid[None], progress=iteration / 29419, iteration=iteration)
        torch.testing.assert_close(final.sum(-1), candidates.sum(-1))
        torch.testing.assert_close(final[1], candidates[1])
        torch.testing.assert_close(state, torch.random.get_rng_state())
        assert py_state == __import__("random").getstate()
        if iteration == 0:
            torch.testing.assert_close(final, candidates)
        pack_masks(final, upperbound=int(candidates.sum()))
        checks.append({"iteration": iteration, "seconds": time.perf_counter() - started,
                       "metrics": policy.last_metrics.copy(), "records": policy.last_records.copy()})
    if policy.last_metrics["mask_structure_applied_anchors"] != 1:
        raise RuntimeError(f"real-image structure proposal fell back: {policy.last_records}")
    args.output_dir.mkdir(parents=True)
    save_structure_mask_grid(tensor, policy.last_examples, output_path=args.output_dir / "training_mask.png",
                             iteration=12000, probability=1.0)
    example = policy.last_examples[0]
    np.savez_compressed(args.output_dir / "mask_arrays.npz", baseline=baseline.numpy(), final=final[0].reshape(valid.shape).numpy(),
        core=example["ribbon"].core.numpy(), supplement=example["ribbon"].supplement.numpy(),
        context=example["ribbon"].context.numpy(), valid=valid.numpy(), structure=example["score"].numpy())
    print("PASS: real Teacher mixed mask, exact budget, protected ends, unchanged RNG and warmup", flush=True)
    for name in ("student", "student_head", "decoder"):
        modules[name].requires_grad_(True)
    optimizer = torch.optim.AdamW([parameter for name in ("student", "student_head", "decoder")
                                  for parameter in modules[name].parameters()], lr=1e-5)
    backbone_parameter = modules["student"].blocks[-1].mlp.fc2.weight
    decoder_parameter = modules["decoder"].pred.weight
    before = [backbone_parameter.detach().clone(), decoder_parameter.detach().clone()]
    started = time.perf_counter()
    features = modules["student"].forward_features(tensor, masks=final[:1])["x_norm_patchtokens"]
    ids = final[0].nonzero().flatten()
    pixel = PixelReconstructionLoss(14)(modules["decoder"](features), tensor, final[:1])
    student_logits = modules["student_head"](features[0, ids])
    ibot = -(q[ids] * F.log_softmax(student_logits.float() / bundle["student_temp"], -1)).sum(-1).mean()
    loss = pixel + ibot
    loss.backward()
    gradients = [parameter.grad for name in ("student", "student_head", "decoder") for parameter in modules[name].parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(parameter.grad is None for parameter in modules["teacher"].parameters())
    optimizer.step()
    assert not torch.equal(before[0], backbone_parameter)
    assert not torch.equal(before[1], decoder_parameter)
    result = {"passed": True, "scope": "native CPU ViT-B/14 and trained heads/decoder; not CUDA/FSDP",
        "image": str(args.image.resolve()), "anchor_mode": args.anchor_mode, "smoke_probability_override": 1.0,
        "checks": checks, "pixel_mse": float(pixel.detach()), "ibot_ce": float(ibot.detach()),
        "optimization_loss": float(loss.detach()), "forward_backward_update_seconds": time.perf_counter() - started,
        "backbone_updated": True, "decoder_updated": True, "teacher_has_gradients": False,
        "weight_files_modified": False, "trained_checkpoint_written": False}
    (args.output_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "output": str(args.output_dir.resolve()), "pixel_mse": result["pixel_mse"], "ibot_ce": result["ibot_ce"]}), flush=True)


if __name__ == "__main__":
    main()
