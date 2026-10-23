"""Export the teacher (DAX/CheXFound) or MAE encoder without training heads."""
from __future__ import annotations

import argparse
from pathlib import Path

from .checkpointing import atomic_torch_save, load_tensor_checkpoint


def encoder_state(payload):
    method = payload["config"]["method"]
    if method == "medical_mae":
        state = {k: v for k, v in payload["student"].items() if not k.startswith("decoder_") and k != "mask_token"}
    else:
        state = {k.removeprefix("backbone."): v for k, v in payload["teacher"].items() if k.startswith("backbone.")}
    if not state or "patch_embed.proj.weight" not in state:
        raise ValueError("Checkpoint does not contain the expected trained encoder")
    return state


def export_checkpoint(checkpoint, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Backbone export already exists: {output}")
    payload = load_tensor_checkpoint(checkpoint)
    if payload.get("format_version") != 1:
        raise ValueError("Expected a baseline_pretrain training checkpoint")
    state = encoder_state(payload)
    metadata = {"method": payload["config"]["method"], "architecture": payload["config"]["model"],
                "normalization": {key: payload["config"]["crops"][key] for key in ("mean", "std")},
                "optimizer_updates": payload["optimizer_updates"], "images_seen": payload["images_seen"],
                "identity": payload["identity"], "initialization": payload["initialization"],
                "source": "teacher" if payload["teacher"] is not None else "mae_encoder"}
    atomic_torch_save({"model": state, "metadata": metadata}, output)
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
