"""Restore trained inference modules from LOCAL_STATE_DICT rank checkpoints.

Only existing, trusted local training checkpoints should use the exporter.
ShardedTensor wrappers are read as inert records, not distributed Tensor objects.
The original named parameter fragments are joined by saved global rank; padding
in _flat_param is never interpreted as a named parameter. No process group or
FSDP inference model is created. Inference bundles contain ordinary tensors only.
"""

from dataclasses import dataclass
from pathlib import Path
import pickle
import types

import torch
import yaml

from visualization.dino_spine_maps.model_loader import _build_dinov2, _torch_load, _extract_state_dict


@dataclass
class LocalShardRecord:
    local_shards: list
    metadata: object
    pg_state: object


def _rebuild_tensor_record(func, new_type, args, state):
    if new_type.__module__ == "torch.distributed._shard.sharded_tensor.api" and new_type.__name__ == "ShardedTensor":
        if not isinstance(state, tuple) or len(state) != 5:
            raise ValueError("unsupported ShardedTensor checkpoint state")
        return LocalShardRecord(state[0], state[1], state[2])
    return torch._tensor._rebuild_from_type_v2(func, new_type, args, state)


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("torch._tensor", "_rebuild_from_type_v2"):
            return _rebuild_tensor_record
        return super().find_class(module, name)


def load_local_checkpoint(path):
    """Read our trusted local rank checkpoint without constructing process groups."""
    reader = types.ModuleType("local_checkpoint_pickle")
    reader.Unpickler = _CheckpointUnpickler
    reader.load, reader.loads = pickle.load, pickle.loads
    return torch.load(path, map_location="cpu", pickle_module=reader, weights_only=False, mmap=True)


def merge_named_state(states, prefix, reference):
    """Strictly reconstruct original parameters, including split parameters."""
    extracted = [{k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix) and not k.endswith("_flat_param")} for state in states]
    expected = reference.state_dict()
    for state in extracted:
        if set(state) != set(expected):
            raise ValueError(f"{prefix}: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}")
    buffers = set(dict(reference.named_buffers()))
    output = {}
    for name, target in expected.items():
        fragments = [state[name] for state in extracted]
        if not all(isinstance(fragment, torch.Tensor) and fragment.dtype == target.dtype for fragment in fragments):
            raise ValueError(f"{prefix}{name}: expected ordinary fragments of dtype {target.dtype}")
        if name in buffers:
            if any(fragment.shape != target.shape or not torch.equal(fragment, fragments[0]) for fragment in fragments):
                raise ValueError(f"{prefix}{name}: replicated buffer mismatch")
            tensor = fragments[0].clone()
        else:
            if sum(fragment.numel() for fragment in fragments) != target.numel():
                raise ValueError(f"{prefix}{name}: incomplete or duplicated parameter fragments")
            tensor = torch.cat([fragment.reshape(-1) for fragment in fragments]).reshape(target.shape)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{prefix}{name}: nonfinite restored values")
        output[name] = tensor
    return output


def _modules(signature, cfg, device="cpu"):
    student = _build_dinov2(Path(__file__).resolve().parents[2], device=torch.device(device))
    teacher = _build_dinov2(Path(__file__).resolve().parents[2], device=torch.device(device))
    from dinov2.layers import DINOHead
    from methods.geotopo_dino.models.pixel_decoder import PixelDecoder
    if cfg["ibot"]["separate_head"]:
        head_cfg, head_name = cfg["ibot"], "ibot_head"
    else:
        head_cfg, head_name = cfg["dino"], "dino_head"
    kwargs = dict(in_dim=signature["embed_dim"], out_dim=head_cfg["head_n_prototypes"],
                  hidden_dim=head_cfg["head_hidden_dim"], bottleneck_dim=head_cfg["head_bottleneck_dim"],
                  nlayers=head_cfg["head_nlayers"])
    student_head, teacher_head = DINOHead(**kwargs), DINOHead(**kwargs)
    decoder = PixelDecoder(**{key: signature[key] for key in (
        "embed_dim", "grid_size", "patch_size", "in_chans", "decoder_dim", "decoder_depth", "decoder_num_heads")})
    return dict(student=student, teacher=teacher, student_head=student_head, teacher_head=teacher_head, decoder=decoder), head_name


def export_bundle(rank_paths, config_path, output_path, verify_teacher=None):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    paths = [Path(path).resolve() for path in rank_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("provide distinct rank checkpoints")
    checkpoints = [load_local_checkpoint(path) for path in paths]
    ranks = []
    for checkpoint in checkpoints:
        records = [value for name, value in checkpoint["model"].items() if name.endswith("_flat_param") and isinstance(value, LocalShardRecord)]
        if not records:
            raise ValueError("expected LOCAL_STATE_DICT checkpoint with original named parameter fragments")
        pg = records[0].pg_state
        if any(record.pg_state != pg for record in records):
            raise ValueError("inconsistent saved ranks within checkpoint")
        ranks.append((pg.global_rank, pg.global_world_size))
    world = ranks[0][1]
    if any(size != world for _, size in ranks) or sorted(rank for rank, _ in ranks) != list(range(world)):
        raise ValueError("all ranks from the saved world size are required")
    ordered = sorted(zip(ranks, checkpoints, paths), key=lambda item: item[0][0])
    checkpoints, paths = [item[1] for item in ordered], [item[2] for item in ordered]
    iteration = checkpoints[0]["iteration"]
    signature = checkpoints[0]["pixel_reconstruction_signature"]
    if any(checkpoint["iteration"] != iteration or checkpoint["pixel_reconstruction_signature"] != signature for checkpoint in checkpoints):
        raise ValueError("rank checkpoints differ in iteration or decoder signature")
    if not signature["enabled"] or signature["norm_pix_loss"]:
        raise ValueError("this probe requires a trained raw-pixel decoder (norm_pix_loss=false)")
    cfg = yaml.safe_load(Path(config_path).read_text())
    if cfg["train"]["centering"] != "centering":
        raise ValueError("fixed offline prototype targets currently require saved teacher centering")
    if signature["patch_size"] != 14 or tuple(signature["grid_size"]) != (37, 37) or signature["embed_dim"] != 768:
        raise ValueError("probe supports the existing ViT-B/14, 518x518 model")
    modules, head_name = _modules(signature, cfg)
    states = [checkpoint["model"] for checkpoint in checkpoints]
    prefixes = dict(student="student.backbone.", teacher="teacher.backbone.",
                    student_head=f"student.{head_name}.", teacher_head=f"teacher.{head_name}.", decoder="student_aux.pixel_decoder.")
    restored = {name: merge_named_state(states, prefixes[name], module) for name, module in modules.items()}
    centers = [state["ibot_patch_loss.center"] for state in states]
    if any(not torch.equal(center, centers[0]) for center in centers):
        raise ValueError("teacher prototype centers differ across ranks")
    verified = False
    if verify_teacher:
        reference = _extract_state_dict(_torch_load(Path(verify_teacher)), Path(verify_teacher))
        if set(reference) != set(restored["teacher"]) or any(not torch.equal(reference[key], restored["teacher"][key]) for key in reference):
            raise ValueError("restored Teacher differs from the supplied exported backbone")
        verified = True
    for name, module in modules.items():
        module.load_state_dict(restored[name], strict=True)
    data = {"format_version": 1, "modules": restored, "teacher_center": centers[0].clone(),
        "pixel_signature": signature, "config": cfg, "iteration": int(iteration),
        "teacher_temp": float(cfg["teacher"]["teacher_temp"]), "student_temp": 0.1,
        "provenance": {"rank_paths": [str(path) for path in paths], "config_path": str(Path(config_path).resolve()),
                       "verified_teacher_backbone": str(Path(verify_teacher).resolve()) if verify_teacher else None,
                       "teacher_exact_match": verified, "restored_named_parameters": True}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    return {"output": str(output_path.resolve()), "iteration": int(iteration), "teacher_exact_match": verified,
            "module_keys": {name: len(state) for name, state in restored.items()}}


def load_bundle(path, device="cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if bundle["format_version"] != 1:
        raise ValueError("unsupported inference bundle")
    modules, _ = _modules(bundle["pixel_signature"], bundle["config"], device)
    for name, module in modules.items():
        module.load_state_dict(bundle["modules"][name], strict=True)
        module.requires_grad_(False).eval().to(device)
    center = bundle["teacher_center"].reshape(1, -1).to(device)
    return modules, center, bundle


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-rank", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verify-teacher-backbone", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    print(__import__("json").dumps(export_bundle(args.checkpoint_rank, args.config, args.output, args.verify_teacher_backbone)))
