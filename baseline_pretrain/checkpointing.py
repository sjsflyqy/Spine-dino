"""Strict public initialization, isolated atomic checkpoints and rank-local RNG state."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil

import numpy as np
import torch
import torch.distributed as dist

from .config import resolve


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy_scalar(*args):
    # Map the old NumPy pickle spelling without enabling arbitrary unpickling.
    from numpy.core.multiarray import scalar
    return scalar(*args)


_legacy_scalar.__name__ = "scalar"
_legacy_scalar.__module__ = "numpy.core.multiarray"


def load_tensor_checkpoint(path):
    allowed = [argparse.Namespace, set, _legacy_scalar, np.dtype]
    allowed += [type(np.dtype(name)) for name in ("float64", "float32", "int64", "int32")]
    # NumPy <2 uses the legacy spelling directly.
    from numpy.core.multiarray import scalar
    allowed.append(scalar)
    with torch.serialization.safe_globals(allowed):
        try:
            return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError as exc:
            if "mmap can only" not in str(exc):
                raise
            return torch.load(path, map_location="cpu", weights_only=True)


def public_state(checkpoint, method):
    state = checkpoint.get("model") if method == "medical_mae" else checkpoint.get("teacher")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{method}: expected {'model' if method == 'medical_mae' else 'teacher'} tensor dictionary")
    output = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Non-tensor public parameter: {key}")
        if key.startswith("module."):
            key = key[7:]
        if method == "dax" and key.startswith("head."):
            key = "dino_head." + key[5:]
        if method == "chexfound":
            key = re.sub(r"backbone\.blocks\.\d+\.(\d+)\.", r"backbone.blocks.\1.", key)
        key = key.replace("last_layer.parametrizations.weight.original0", "last_layer.weight_g")
        key = key.replace("last_layer.parametrizations.weight.original1", "last_layer.weight_v")
        if key in output:
            raise ValueError(f"Duplicate parameter after checkpoint conversion: {key}")
        output[key] = value
    return output


def initialize(model, config, *, compute_hash=True):
    path = resolve(config["initialization"])
    checkpoint = load_tensor_checkpoint(path)
    state = public_state(checkpoint, config["method"])
    expected = model.state_dict()
    missing, unexpected = sorted(expected.keys() - state.keys()), sorted(state.keys() - expected.keys())
    shapes = {k: (list(state[k].shape), list(expected[k].shape))
              for k in state.keys() & expected.keys() if state[k].shape != expected[k].shape}
    if missing or unexpected or shapes:
        raise ValueError(f"Public initialization mismatch: missing={missing[:12]}, unexpected={unexpected[:12]}, shapes={dict(list(shapes.items())[:8])}")
    # Construction on meta avoids initializing hundreds of millions of random parameters.
    model.load_state_dict(state, strict=True, assign=True)
    return {"path": str(path), "sha256": sha256(path) if compute_hash else None, "loaded_tensors": len(state),
            "loaded_elements": sum(x.numel() for x in state.values()),
            "public_epoch": checkpoint.get("epoch"), "all_parameters_loaded": True}


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def rng_state(device):
    np_state = np.random.get_state()
    result = {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
              "torch": torch.get_rng_state()}
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device)
    return result


def restore_rng(state, device):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def is_fsdp(model):
    from torch.distributed.fsdp import FullyShardedDataParallel
    return isinstance(model, FullyShardedDataParallel)


def unwrap(model):
    from torch.nn.parallel import DistributedDataParallel
    return model.module if isinstance(model, DistributedDataParallel) else model


@contextlib.contextmanager
def full_state(model, rank0_only=True):
    if not is_fsdp(model):
        yield
        return
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, StateDictType,
                                       FullStateDictConfig, FullOptimStateDictConfig)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only),
                             FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)):
        yield


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in unwrap(model).state_dict().items()}


def save_training(path, student, teacher, objective, optimizer, scaler, metadata, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    local_rng = rng_state(device)
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, local_rng)
    else:
        states[0] = local_rng
    with full_state(student):
        student_state = cpu_state(student)
        if is_fsdp(student):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            optimizer_state = FSDP.optim_state_dict(student, optimizer)
        else:
            optimizer_state = optimizer.state_dict() if rank == 0 else None
    if teacher is not None:
        with full_state(teacher):
            teacher_state = cpu_state(teacher)
    else:
        teacher_state = None
    # A new atomic checkpoint must fit alongside the previous valid file.
    # Check collectively so a rank does not wait indefinitely at the save barrier.
    disk_error = [None]
    if rank == 0:
        def tensor_bytes(value):
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(tensor_bytes(x) for x in value.values())
            if isinstance(value, (tuple, list)):
                return sum(tensor_bytes(x) for x in value)
            return 0
        required = tensor_bytes((student_state, teacher_state, optimizer_state)) + 64 * 1024 ** 2
        free = shutil.disk_usage(Path(path).parent).free
        if free < required:
            disk_error[0] = f"Insufficient checkpoint space: need about {required / 2**30:.2f} GiB free, have {free / 2**30:.2f} GiB"
    if world > 1:
        dist.broadcast_object_list(disk_error, src=0)
    if disk_error[0]:
        raise OSError(disk_error[0])
    if rank == 0:
        payload = {**metadata, "format_version": 1, "student": student_state, "teacher": teacher_state,
                   "objective": cpu_state(objective) if objective is not None else None,
                   "optimizer": optimizer_state, "scaler": scaler.state_dict(), "rng": states}
        atomic_torch_save(payload, path)
    if world > 1:
        dist.barrier()


def restore_training(payload, student, teacher, objective, optimizer, scaler, device):
    # All ranks load a full CPU checkpoint. FSDP scatters parameters into their local shards.
    with full_state(student, rank0_only=False):
        unwrap(student).load_state_dict(payload["student"], strict=True)
        if is_fsdp(student):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            state = FSDP.optim_state_dict_to_load(student, optimizer, payload["optimizer"])
        else:
            state = payload["optimizer"]
        optimizer.load_state_dict(state)
    if teacher is not None:
        with full_state(teacher, rank0_only=False):
            unwrap(teacher).load_state_dict(payload["teacher"], strict=True)
    if objective is not None:
        objective.load_state_dict(payload["objective"], strict=True)
    scaler.load_state_dict(payload["scaler"])
    rank = dist.get_rank() if dist.is_initialized() else 0
    restore_rng(payload["rng"][rank], device)
