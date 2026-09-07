"""Launch DINOv3 ViT-B/16 continued pretraining on the SpineDINO dataset."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "SpinePretrain-v1" / "spine_dino_dataset"


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--pretrained-weights",
        default="",
        help=(
            "Optional DINOv3 training-initialization checkpoint. "
            "Leave unset to train from random initialization."
        ),
    )
    parser.add_argument("--config", default="dinov3/configs/train/vitb16_spine.yaml")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "upstream" / "pretrain" / "dinov3_vitb16_init"),
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset-extra", default="")
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epoch-length", type=int, default=200)
    parser.add_argument(
        "--target-passes",
        type=float,
        default=0.0,
        help="Stop after this many equivalent dataset passes; overrides --epochs as the stop point",
    )
    parser.add_argument(
        "--schedule-passes",
        type=float,
        default=0.0,
        help="LR/WD/EMA schedule horizon in passes; set to the final planned budget for staged resume",
    )
    parser.add_argument("--checkpoint-period", type=int, default=1000)
    parser.add_argument("--checkpoint-max-to-keep", type=int, default=3)
    parser.add_argument(
        "--checkpoint-keep-every",
        type=int,
        default=5000,
        help="Permanent recovery checkpoint interval in steps; 0 disables permanent copies",
    )
    parser.add_argument(
        "--teacher-period",
        type=int,
        default=2000,
        help="Consolidated EMA teacher interval in steps; 0 saves only the final teacher",
    )
    parser.add_argument("--global-crop-size", type=int, default=512)
    parser.add_argument("--local-crop-size", type=int, default=192)
    parser.add_argument("--global-crop-scale", type=float, nargs=2, default=(0.50, 1.00))
    parser.add_argument("--local-crop-scale", type=float, nargs=2, default=(0.20, 0.50))
    parser.add_argument("--local-crops-number", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the launch command only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dataset_extra = (
        Path(args.dataset_extra).expanduser().resolve() if args.dataset_extra else dataset_root / "extra"
    )
    weights = (
        Path(args.pretrained_weights).expanduser().resolve()
        if args.pretrained_weights
        else None
    )
    required = [
        dataset_root / "train" / "spine",
        dataset_extra / "image_files-TRAIN.npy",
        dataset_extra / "class-ids-TRAIN.npy",
    ]
    if weights is not None:
        required.insert(0, weights)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n  " + "\n  ".join(missing))
    positive_values = {
        "ngpu": args.ngpu,
        "batch-size": args.batch_size,
        "num-workers": args.num_workers + 1,
        "prefetch-factor": args.prefetch_factor,
        "epoch-length": args.epoch_length,
        "checkpoint-period": args.checkpoint_period,
        "checkpoint-max-to-keep": args.checkpoint_max_to_keep,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError("These options must be positive: " + ", ".join(invalid))
    if args.checkpoint_keep_every < 0 or args.teacher_period < 0:
        raise ValueError("Checkpoint/teacher periods must be >= 0")
    for name, size in (("global", args.global_crop_size), ("local", args.local_crop_size)):
        if size % 16 != 0:
            raise ValueError(f"{name} crop size must be divisible by ViT-B/16 patch size 16: {size}")

    dataset = (
        f"SpineDINO:split=TRAIN:root={dataset_root.as_posix()}:extra={dataset_extra.as_posix()}"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_size = int(
        np.load(dataset_extra / "image_files-TRAIN.npy", mmap_mode="r").shape[0]
    )
    global_batch_size = args.ngpu * args.batch_size
    target_steps = None
    schedule_steps = args.epochs * args.epoch_length
    if args.target_passes > 0:
        target_steps = math.ceil(args.target_passes * dataset_size / global_batch_size)
        schedule_passes = args.schedule_passes or args.target_passes
        if schedule_passes < args.target_passes:
            raise ValueError("--schedule-passes must be >= --target-passes")
        schedule_steps = math.ceil(schedule_passes * dataset_size / global_batch_size)
        args.epochs = math.ceil(schedule_steps / args.epoch_length)
        schedule_steps = args.epochs * args.epoch_length
    elif args.schedule_passes > 0:
        raise ValueError("--schedule-passes requires --target-passes")

    plan = {
        "dataset_size": dataset_size,
        "global_batch_size": global_batch_size,
        "epoch_length": args.epoch_length,
        "schedule_steps": schedule_steps,
        "schedule_passes": schedule_steps * global_batch_size / dataset_size,
        "pretrained_weights": weights.as_posix() if weights is not None else "random_init",
    }
    plan_path = output_dir / "spine_run_plan.json"
    if args.target_passes > 0 and plan_path.is_file() and not args.no_resume:
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "dataset_size", "global_batch_size", "epoch_length",
            "schedule_steps", "pretrained_weights",
        )
        changed = {
            key: (existing_plan.get(key), plan.get(key))
            for key in comparable_keys
            if existing_plan.get(key) != plan.get(key)
        }
        if changed:
            raise RuntimeError(
                "Resume plan differs from the original schedule; this would make the "
                f"continuation inconsistent: {changed}"
            )
    if args.target_passes > 0:
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    opts = [
        f"train.dataset_path={dataset}",
        f"train.batch_size_per_gpu={args.batch_size}",
        f"train.num_workers={args.num_workers}",
        f"train.prefetch_factor={args.prefetch_factor}",
        f"train.OFFICIAL_EPOCH_LENGTH={args.epoch_length}",
        f"optim.epochs={args.epochs}",
        f"train.stop_after_iterations={target_steps or 0}",
        f"checkpointing.period={args.checkpoint_period}",
        f"checkpointing.max_to_keep={args.checkpoint_max_to_keep}",
        f"checkpointing.keep_every={args.checkpoint_keep_every}",
        f"evaluation.eval_period_iterations={args.teacher_period}",
        f"crops.global_crops_size={args.global_crop_size}",
        f"crops.local_crops_size={args.local_crop_size}",
        f"crops.global_crops_scale={list(args.global_crop_scale)}",
        f"crops.local_crops_scale={list(args.local_crop_scale)}",
        f"crops.local_crops_number={args.local_crops_number}",
    ]
    if weights is not None:
        opts.insert(0, f"student.pretrained_weights={weights.as_posix()}")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={args.ngpu}",
        "--nnodes=1",
        "--standalone",
        "dinov3/train/train.py",
        "--config-file",
        args.config,
        "--output-dir",
        str(output_dir),
    ]
    if args.no_resume:
        command.append("--no-resume")
    command.extend(opts)

    print("DINOv3 Spine pretraining")
    print(f"weights: {weights if weights is not None else '(random initialization)'}")
    print(f"dataset: {dataset_root}")
    print(f"output:  {output_dir}")
    print(f"dataset size:   {dataset_size}")
    print(f"global batch:   {global_batch_size}")
    print(f"stop steps:     {target_steps or schedule_steps}")
    print(f"schedule steps: {schedule_steps}")
    if target_steps:
        print(f"target passes:  {target_steps * global_batch_size / dataset_size:.6f}")
        print(f"schedule passes:{schedule_steps * global_batch_size / dataset_size:.6f}")
    print("command: " + " ".join(command), flush=True)
    if args.dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(command, cwd=Path(__file__).resolve().parent, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
