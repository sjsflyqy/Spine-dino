"""Isolated DAX/CheXFound/Medical-MAE continued pretraining.

All schedules count optimizer updates. Run --dry-run to inspect a launch
without making directories, initializing distributed state, or allocating CUDA.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import yaml

from .checkpointing import (atomic_json, initialize, is_fsdp, load_tensor_checkpoint,
                            restore_training, save_training, sha256, unwrap)
from .config import ROOT, compatibility_config, config_digest, load_config, resolve
from .data import IndexedSpineDataset, StreamBatchSampler, block_masks
from .models import Block, DINOHead, VisionTransformer, build_network
from .objectives import DistillationObjective, learning_rate, schedule


def cuda_memory_metrics(device, world):
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    values = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
                          device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return dict(zip(("peak_allocated_gib", "peak_reserved_gib"), (values / 2 ** 30).cpu().tolist()))


def output_path(path):
    path = resolve(path)
    allowed = ROOT / "outputs" / "baseline_pretrain"
    if path == ROOT or (ROOT in path.parents and allowed not in path.parents):
        raise ValueError("Inside this repository outputs must be below outputs/baseline_pretrain/<experiment>")
    return path


@contextlib.contextmanager
def experiment_directory(path, resume):
    path.mkdir(parents=True, exist_ok=True)
    if not resume and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}. Use a new directory or explicit --resume")
    lock = path / ".run.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Experiment lock exists: {lock}. Check the previous job before removing a stale lock") from exc
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"pid": os.getpid(), "hostname": os.uname().nodename}, stream)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def parameter_groups(model, config):
    cfg, depth = config["training"], config["model"]["depth"]
    groups = {}
    for name, param in model.named_parameters():
        name = name.replace("_fsdp_wrapped_module.", "")
        if not param.requires_grad:
            continue
        backbone = name.startswith("backbone.") or (config["method"] == "medical_mae" and not name.startswith(("decoder_", "mask_token")))
        mult = 1.0
        if backbone:
            short = name.removeprefix("backbone.")
            layer = int(short.split(".")[1]) + 1 if short.startswith("blocks.") else (0 if short.startswith(("patch_embed", "pos_embed", "cls_token", "register_tokens", "mask_token")) else depth + 1)
            mult = cfg["backbone_lr_multiplier"] * cfg["layerwise_decay"] ** (depth + 1 - layer)
            if "patch_embed" in name:
                mult *= cfg["patch_embed_lr_multiplier"]
        decay = param.ndim > 1 and not any(x in name for x in ("pos_embed", "cls_token", "mask_token", "register_tokens"))
        key = (mult, decay)
        groups.setdefault(key, {"params": [], "lr_multiplier": mult, "apply_decay": decay})["params"].append(param)
    return list(groups.values())


def parallelize(model, cfg, device, world):
    kind = cfg["training"]["parallel"]
    if kind == "fsdp":
        if device.type != "cuda" or world < 2:
            raise ValueError("The FSDP preset requires >=2 CUDA ranks")
        from functools import partial
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[cfg["training"]["precision"]]
        return FSDP(model, device_id=device, use_orig_params=True, sync_module_states=True,
                    sharding_strategy=getattr(ShardingStrategy, cfg["training"]["fsdp_sharding"]),
                    mixed_precision=MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float32, buffer_dtype=torch.float32),
                    auto_wrap_policy=ModuleWrapPolicy({Block, DINOHead}))
    if world > 1 and kind == "none":
        raise ValueError("parallel=none is only valid for one rank")
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(model, device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=False)
    return model


def initialize_fsdp_adam_state(optimizer):
    """Materialize zero moments for locally owned original-parameter shards.

    PyTorch 2.5 FULL_STATE_DICT conversion cannot gather a flat group if some
    trainable parameters have never received a gradient (e.g. frozen DINO last
    layers during warmup). Do not fake an optimizer step: keep step=0 and zero
    moments, so unfreezing has exactly the normal AdamW initialization semantics.
    Empty non-owned shards must remain absent so FSDP gathers scalars only from
    ranks owning that parameter.
    """
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.numel() and not optimizer.state[parameter]:
                optimizer.state[parameter].update(step=torch.tensor(0.0),
                    exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))


@torch.no_grad()
def update_teacher(student, teacher, momentum):
    source = dict(unwrap(student).named_parameters())
    target = dict(unwrap(teacher).named_parameters())
    if source.keys() != target.keys():
        raise RuntimeError("Student/teacher parameter layouts differ")
    for name, param in target.items():
        param.lerp_(source[name].detach().to(param.dtype), 1 - momentum)


def make_plan(c, dataset, world, output):
    t = c["training"]
    batch = world * t["batch_size_per_gpu"] * t["accumulation_steps"]
    return {"method": c["method"], "world_size": world, "effective_batch": batch,
            "microbatch_per_rank": t["batch_size_per_gpu"], "accumulation_steps": t["accumulation_steps"],
            "total_optimizer_updates": t["total_updates"], "total_microbatches": t["total_updates"] * t["accumulation_steps"],
            "images_seen_at_completion": batch * t["total_updates"], "dataset_images": len(dataset),
            "equivalent_passes": batch * t["total_updates"] / len(dataset),
            "peak_head_lr": t["learning_rate"],
            "peak_backbone_lr_before_layer_decay": t["learning_rate"] * t["backbone_lr_multiplier"],
            "initialization": str(resolve(c["initialization"])), "output_dir": str(output)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--resume", help="Explicit checkpoint from this experiment; never auto-resume")
    group.add_argument("--no-resume", action="store_true", help="Explicit fresh run (also the default)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check-initialization", action="store_true", help="With --dry-run, strictly validate every public tensor on CPU")
    p.add_argument("--max-updates", type=int, help="Stop early at this absolute update count, preserving the full schedule")
    p.add_argument("overrides", nargs="*", help="Existing dotted YAML keys in key=value form")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    c = load_config(args.config, args.overrides)
    t = c["training"]
    output = output_path(args.output_dir or c["output_dir"])
    if not resolve(c["initialization"]).is_file():
        raise FileNotFoundError(f"Public initialization weights missing: {resolve(c['initialization'])}")
    resume = resolve(args.resume) if args.resume else None
    if resume and (resume.parent != output or not resume.is_file()):
        raise ValueError("--resume must be an existing checkpoint directly inside --output-dir")
    if resume and (output / "latest.json").is_file():
        latest = json.loads((output / "latest.json").read_text())["checkpoint"]
        if resume.name != latest:
            raise ValueError("Resume the latest checkpoint in this directory to avoid overwriting later results")
    stop = args.max_updates if args.max_updates is not None else t["total_updates"]
    if stop < 1 or stop > t["total_updates"]:
        raise ValueError("--max-updates must lie between 1 and total_updates")
    dataset = IndexedSpineDataset(c)
    if args.dry_run:
        world = int(os.environ.get("WORLD_SIZE", t["expected_world_size"]))
        plan = make_plan(c, dataset, world, output)
        if args.check_initialization:
            model = build_network(c, device="meta")
            plan["initialization_audit"] = initialize(model, c)
        if int(os.environ.get("RANK", 0)) == 0:
            print(json.dumps(plan, indent=2), flush=True)
        return

    world, rank, local_rank = (int(os.environ.get(k, default)) for k, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if world != t["expected_world_size"]:
        raise ValueError(f"This preset expects {t['expected_world_size']} ranks, got {world}. Use torchrun or explicitly override training.expected_world_size")
    cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
        if t["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 is not supported on this GPU; explicitly select fp16")
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(minutes=15))
    try:
        lock = experiment_directory(output, resume) if rank == 0 else contextlib.nullcontext()
        with lock:
            if world > 1:
                dist.barrier()
            run(c, args, dataset, output, resume, device, world, rank, stop)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(c, args, dataset, output, resume, device, world, rank, stop):
    t = c["training"]
    random.seed(c["seed"])
    np.random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    model = build_network(c, device="meta")
    initialization = initialize(model, c, compute_hash=rank == 0)
    if world > 1:
        audit = [initialization if rank == 0 else None]
        dist.broadcast_object_list(audit, src=0)
        initialization = audit[0]
    # CPU checkpoints are MAP_PRIVATE; move to device before optimization.
    model = model.to(device)
    teacher = None
    if c["method"] != "medical_mae":
        teacher = copy.deepcopy(model)
        teacher.is_teacher = True
        teacher.requires_grad_(False).eval()
        for m in teacher.modules():
            if isinstance(m, Block):
                m.drop_path = 0
            if isinstance(m, VisionTransformer):
                m.gradient_checkpointing = False
    # Same initialization on every rank; independent reproducible masking/drop-path RNG.
    random.seed(c["seed"] + rank)
    np.random.seed(c["seed"] + rank)
    torch.manual_seed(c["seed"] + rank)
    # FSDP exposes original parameters as 1-D shard views outside forward.
    # Derive decay/LR groups from full original shapes BEFORE wrapping, but
    # instantiate the optimizer AFTER wrapping (use_orig_params preserves ids).
    groups = parameter_groups(model, c)
    student = parallelize(model, c, device, world)
    if {id(p) for g in groups for p in g["params"]} != {id(p) for p in student.parameters() if p.requires_grad}:
        raise RuntimeError("Parallel wrapper replaced original optimizer parameter objects")
    if teacher is not None and t["parallel"] == "fsdp":
        teacher = parallelize(teacher, c, device, world)
    objective = DistillationObjective(c["method"], c["objective"]).to(device) if teacher is not None else None
    if c["method"] in {"chexfound", "cxformer"} and c["objective"]["koleo_weight"]:
        neighbors = t["batch_size_per_gpu"] * (world if c["objective"]["koleo_scope"] == "global" else 1)
        if neighbors < 2:
            raise ValueError("Configured KoLeo has fewer than two images per view")
    optimizer = torch.optim.AdamW(groups, lr=t["learning_rate"], betas=tuple(t["betas"]))
    if is_fsdp(student):
        initialize_fsdp_adam_state(optimizer)
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler(enabled=device.type == "cuda" and t["precision"] == "fp16")
    else:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and t["precision"] == "fp16")
    plan = make_plan(c, dataset, world, output)
    identity = {"implementation": "baseline_pretrain_v1_original_shape_groups",
                "config_digest": config_digest(c), "index_sha256": sha256(dataset.index_path),
                "initialization_sha256": initialization["sha256"], "world_size": world}
    start = 0
    if resume:
        payload = load_tensor_checkpoint(resume)
        if payload["identity"] != identity:
            raise ValueError("Resume identity mismatch: model/training config, data, initialization or world size changed")
        restore_training(payload, student, teacher, objective, optimizer, scaler, device)
        start = payload["optimizer_updates"]
        expected = start * t["accumulation_steps"]
        if payload["microbatches"] != expected or payload["images_seen"] != expected * world * t["batch_size_per_gpu"]:
            raise ValueError("Checkpoint counters are not at a valid accumulation boundary")
        del payload
    elif rank == 0:
        (output / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        atomic_json({**plan, "initialization_audit": initialization, "identity": identity,
                     "torch": str(torch.__version__), "device": str(device),
                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}, output / "provenance.json")
    if rank == 0:
        print(json.dumps({**plan, "starting_update": start}, indent=2), flush=True)
    if start >= stop:
        return
    accumulation = t["accumulation_steps"]
    sampler = StreamBatchSampler(len(dataset), t["batch_size_per_gpu"], rank, world, c["seed"], start * accumulation, stop * accumulation)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=t["workers"],
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(c["seed"] + rank),
                        **({"prefetch_factor": t["prefetch_factor"]} if t["workers"] else {}))
    iterator = iter(loader)
    student.train()
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[t["precision"]]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    beginning = time.monotonic()
    last_memory = {}
    for update in range(start, stop):
        lr = learning_rate(t, update)
        wd = schedule(t["weight_decay"], t["weight_decay_end"], update, t["total_updates"])
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_multiplier"]
            group["weight_decay"] = wd if group["apply_decay"] else 0.0
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        for _ in range(accumulation):
            batch = next(iterator)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=t["precision"] != "fp32"):
                if teacher is None:
                    loss = student(batch.to(device, non_blocking=True), mask_ratio=c["objective"]["mask_ratio"], norm_pix_loss=c["objective"]["norm_pix_loss"])
                    parts = {"mae_loss": loss.detach()}
                else:
                    crops = [image.to(device, non_blocking=True) for image in batch]
                    masks = block_masks(2 * len(crops[0]), c["crops"]["global_size"] // c["model"]["patch_size"], c["objective"], device) if c["method"] in {"chexfound", "cxformer"} else None
                    with torch.no_grad():
                        targets = teacher(crops[:2], masks)
                    predictions = student(crops, masks)
                    loss, parts = objective(predictions, targets, masks, update)
            finite = torch.isfinite(loss.detach()).to(torch.int32)
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f"Non-finite loss at optimizer update {update}")
            scaler.scale(loss / accumulation).backward()
            for key, value in {"loss": loss.detach(), **parts}.items():
                metrics[key] = metrics.get(key, 0.0) + float(value) / accumulation
            # Release large prototype logits before the next forward.
            if teacher is not None:
                del targets, predictions
            del loss, parts, batch
        scaler.unscale_(optimizer)
        if update < t["freeze_last_layer_updates"]:
            for name, parameter in student.named_parameters():
                if ".last_layer." in name:
                    parameter.grad = None
        if t["clip_grad"]:
            norm = student.clip_grad_norm_(t["clip_grad"]) if is_fsdp(student) else torch.nn.utils.clip_grad_norm_(student.parameters(), t["clip_grad"])
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Non-finite gradient at optimizer update {update}")
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            raise FloatingPointError("AMP skipped an update; lower LR or use BF16 and resume the last valid checkpoint")
        if teacher is not None:
            momentum = schedule(c["objective"]["teacher_momentum"], c["objective"]["teacher_momentum_end"], update, t["total_updates"])
            update_teacher(student, teacher, momentum)
            objective.finish_update()
        finished = update + 1
        if finished % t["log_every"] == 0 or finished == stop or finished == start + 1:
            record = {"optimizer_updates": finished, "microbatches": finished * accumulation,
                      "images_seen": finished * plan["effective_batch"], "lr": lr, "weight_decay": wd, **metrics}
            values = torch.tensor(list(metrics.values()), device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(values)
                values /= world
            record.update(zip(metrics, values.cpu().tolist()))
            record.update(cuda_memory_metrics(device, world))
            record["elapsed_seconds"] = time.monotonic() - beginning
            if rank == 0:
                print(json.dumps(record), flush=True)
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
        if finished % t["checkpoint_every"] == 0 or finished == stop:
            metadata = {"config": c, "identity": identity, "initialization": initialization,
                        "optimizer_updates": finished, "microbatches": finished * accumulation,
                        "images_seen": finished * plan["effective_batch"]}
            path = output / f"checkpoint_u{finished:08d}.pt"
            save_training(path, student, teacher, objective, optimizer, scaler, metadata, device)
            last_memory = cuda_memory_metrics(device, world)
            if rank == 0:
                atomic_json({"checkpoint": path.name, "optimizer_updates": finished}, output / "latest.json")
                with (output / "checkpoint_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps({"optimizer_updates": finished, **last_memory,
                                              "elapsed_seconds": time.monotonic() - beginning}) + "\n")
                for old in sorted(output.glob("checkpoint_u????????.pt"))[:-t["keep_last"]]:
                    old.unlink()
            if world > 1:
                dist.barrier()
    if rank == 0:
        from .export_backbone import export_checkpoint
        export_checkpoint(path, output / f"backbone_u{stop:08d}.pth")
        atomic_json({"optimizer_updates": stop, "schedule_complete": stop == t["total_updates"],
                     "checkpoint": path.name, "backbone": f"backbone_u{stop:08d}.pth",
                     "max_rank_memory_including_checkpoint": last_memory}, output / "training_summary.json")


if __name__ == "__main__":
    main()
