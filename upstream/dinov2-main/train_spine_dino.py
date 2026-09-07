"""
Spine-DINO 训练启动脚本（Windows 兼容）

使用流程：
  步骤 1  转换 RAD-DINO 权重
          python load_rad_dino.py \
            --weights-path <safetensors路径> \
            --output-path rad_dino_dinov2_format.pth

  步骤 2  启动训练
          python train_spine_dino.py \
            --pretrained-weights rad_dino_dinov2_format.pth

  完整参数示例：
          python train_spine_dino.py \
            --pretrained-weights rad_dino_dinov2_format.pth \
            --output-dir F:/AAAzpj/spine_dino_output \
            --ngpu 1 \
            --batch-size 16 \
            --epochs 50
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "SpinePretrain-v1" / "spine_dino_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "spine_dino"


def parse_args():
    parser = argparse.ArgumentParser(
        description="启动 Spine-DINO 自监督预训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default="",
        help="RAD-DINO 转换后的权重路径（rad_dino_dinov2_format.pth）；"
             "留空则从随机初始化开始训练",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="dinov2/configs/train/vitb14_spine.yaml",
        help="训练配置文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="检查点和日志保存目录（相对路径会基于运行目录解析）",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=1,
        help="使用的 GPU 数量",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="每张 GPU 的 batch size（0 表示使用配置文件中的值）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="训练总 epoch 数（0 表示使用配置文件中的值）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="每个 GPU 训练进程的数据加载 worker 数量",
    )
    parser.add_argument("--epoch-length", type=int, default=200)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="每个 DataLoader worker 预取的 batch 数；共享内存紧张时设为1",
    )
    parser.add_argument(
        "--target-passes",
        type=float,
        default=0.0,
        help="训练到多少次等效数据遍历后停止；大于0时覆盖 --epochs 停止点",
    )
    parser.add_argument(
        "--schedule-passes",
        type=float,
        default=0.0,
        help="LR/WD/EMA 的最终调度长度；分阶段续训时固定为最终计划 passes",
    )
    parser.add_argument("--checkpoint-period", type=int, default=600)
    parser.add_argument(
        "--checkpoint-max-to-keep",
        type=int,
        default=4,
        help="每个 rank 最多保留的完整恢复权重总数（包括 model_final）",
    )
    parser.add_argument(
        "--teacher-period",
        type=int,
        default=1000,
        help="合并 EMA teacher 的保存间隔（steps）；0 表示只在阶段末保存",
    )
    parser.add_argument(
        "--global-crop-size",
        type=int,
        default=0,
        help="全局裁剪边长（0 表示使用配置文件；RAD-DINO 为 518）",
    )
    parser.add_argument(
        "--local-crop-size",
        type=int,
        default=0,
        help="局部裁剪边长（0 表示使用配置文件；RAD-DINO 为 196）",
    )
    parser.add_argument(
        "--global-crop-scale",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="全局裁剪面积比例范围（RAD-DINO 为 0.50 1.00）",
    )
    parser.add_argument(
        "--local-crop-scale",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="局部裁剪面积比例范围（RAD-DINO 为 0.20 0.50）",
    )
    parser.add_argument(
        "--local-crops-number",
        type=int,
        default=0,
        help="每幅图像的局部裁剪数量（0 表示使用配置文件；RAD-DINO 为 8）",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不从已有检查点恢复，从头开始训练",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="脊柱数据集根目录（默认自动定位仓库中的 SpinePretrain-v1）",
    )
    parser.add_argument(
        "--dataset-extra",
        type=str,
        default="",
        help="数据集元数据目录（默认为 dataset-root/extra）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只验证并打印命令，不启动训练")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dataset_extra = (
        Path(args.dataset_extra).expanduser().resolve()
        if args.dataset_extra
        else dataset_root / "extra"
    )
    required = (
        dataset_root / "train" / "spine",
        dataset_extra / "image_files-TRAIN.npy",
        dataset_extra / "class-ids-TRAIN.npy",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("数据集不完整，缺少:\n  " + "\n  ".join(missing))
    if (
        args.ngpu <= 0
        or args.batch_size < 0
        or args.num_workers < 0
        or args.epoch_length <= 0
        or args.prefetch_factor <= 0
    ):
        raise ValueError("--ngpu 必须 > 0；--batch-size 和 --num-workers 必须 >= 0")
    if args.checkpoint_period <= 0 or args.checkpoint_max_to_keep < 2:
        raise ValueError(
            "--checkpoint-period 必须 > 0；--checkpoint-max-to-keep 至少为2，"
            "以同时保留当前恢复点和阶段 final"
        )
    if args.teacher_period < 0:
        raise ValueError("--teacher-period 必须 >= 0")

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    configured_batch_size = int(OmegaConf.load(config_path).train.batch_size_per_gpu)
    effective_batch_size = args.batch_size or configured_batch_size
    dataset_size = int(
        np.load(dataset_extra / "image_files-TRAIN.npy", mmap_mode="r").shape[0]
    )
    global_batch_size = args.ngpu * effective_batch_size
    epoch_length = args.epoch_length
    target_steps = None
    schedule_steps = args.epochs * epoch_length if args.epochs > 0 else None
    if args.target_passes > 0:
        target_steps = math.ceil(args.target_passes * dataset_size / global_batch_size)
        schedule_passes = args.schedule_passes or args.target_passes
        if schedule_passes < args.target_passes:
            raise ValueError("--schedule-passes 必须 >= --target-passes")
        raw_schedule_steps = math.ceil(schedule_passes * dataset_size / global_batch_size)
        args.epochs = math.ceil(raw_schedule_steps / epoch_length)
        schedule_steps = args.epochs * epoch_length
    elif args.schedule_passes > 0:
        raise ValueError("--schedule-passes 需要同时设置 --target-passes")

    weights_plan_value = (
        str(Path(args.pretrained_weights).expanduser().resolve())
        if args.pretrained_weights
        else "random_init"
    )
    if schedule_steps is not None:
        plan = {
            "dataset_size": dataset_size,
            "global_batch_size": global_batch_size,
            "world_size": args.ngpu,
            "batch_size_per_gpu": effective_batch_size,
            "epoch_length": epoch_length,
            "schedule_steps": schedule_steps,
            "schedule_passes": schedule_steps * global_batch_size / dataset_size,
            "pretrained_weights": weights_plan_value,
        }
    else:
        plan = None
    plan_path = Path(args.output_dir).expanduser().resolve() / "spine_run_plan.json"
    if args.target_passes > 0 and plan_path.is_file() and not args.no_resume:
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        keys = (
            "dataset_size", "global_batch_size", "world_size",
            "batch_size_per_gpu", "epoch_length",
            "schedule_steps", "pretrained_weights",
        )
        changed = {
            key: (existing_plan.get(key), plan.get(key))
            for key in keys
            if existing_plan.get(key) != plan.get(key)
        }
        if changed:
            raise RuntimeError(
                "恢复参数与原始训练计划不一致，会破坏连续调度：" + str(changed)
            )
    if args.target_passes > 0:
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    # 构造 dataset_path（使用修复后的路径解析，支持 Windows 驱动器盘符）
    dataset_path = (
        f"SpineDINO:split=TRAIN"
        f":root={dataset_root.as_posix()}"
        f":extra={dataset_extra.as_posix()}"
    )

    # 基础训练命令：使用 torchrun 启动分布式训练
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={args.ngpu}",
        "--nnodes=1",
        "--standalone",
        "dinov2/train/train.py",
        "--config-file", args.config,
        "--output-dir", args.output_dir,
    ]

    if args.no_resume:
        cmd.append("--no-resume")

    # 通过命令行 opts 覆盖配置文件中的参数
    # OmegaConf.from_cli 要求每个 opt 为 "key=value" 单字符串格式
    opts = []

    if args.pretrained_weights:
        if not os.path.exists(args.pretrained_weights):
            print(f"[警告] 预训练权重文件不存在: {args.pretrained_weights}")
            print("       将从随机初始化开始训练。")
        else:
            opts += [f"student.pretrained_weights={args.pretrained_weights}"]

    opts += [f"train.dataset_path={dataset_path}"]
    opts += [f"train.num_workers={args.num_workers}"]
    opts += [f"train.prefetch_factor={args.prefetch_factor}"]
    opts += [f"train.OFFICIAL_EPOCH_LENGTH={args.epoch_length}"]
    opts += [f"train.checkpoint_period={args.checkpoint_period}"]
    opts += [f"train.checkpoint_max_to_keep={args.checkpoint_max_to_keep}"]
    opts += [f"train.stop_after_iterations={target_steps or 0}"]
    opts += [f"evaluation.eval_period_iterations={args.teacher_period}"]

    if args.batch_size > 0:
        opts += [f"train.batch_size_per_gpu={args.batch_size}"]

    if args.epochs > 0:
        opts += [f"optim.epochs={args.epochs}"]

    if args.global_crop_size > 0:
        opts += [f"crops.global_crops_size={args.global_crop_size}"]
    if args.local_crop_size > 0:
        opts += [f"crops.local_crops_size={args.local_crop_size}"]
    if args.global_crop_scale is not None:
        opts += [f"crops.global_crops_scale={list(args.global_crop_scale)}"]
    if args.local_crop_scale is not None:
        opts += [f"crops.local_crops_scale={list(args.local_crop_scale)}"]
    if args.local_crops_number > 0:
        opts += [f"crops.local_crops_number={args.local_crops_number}"]

    if opts:
        cmd.extend(opts)

    # 打印并执行命令
    print("=" * 60)
    print("Spine-DINO 训练启动")
    print("=" * 60)
    print(f"配置文件    : {args.config}")
    print(f"预训练权重  : {args.pretrained_weights or '（随机初始化）'}")
    print(f"数据集路径  : {dataset_path}")
    print(f"输出目录    : {args.output_dir}")
    print(f"GPU 数量    : {args.ngpu}")
    print(f"数据集大小  : {dataset_size}")
    print(f"全局 batch  : {global_batch_size}")
    if schedule_steps is not None:
        print(f"停止 steps  : {target_steps or schedule_steps}")
        print(f"调度 steps  : {schedule_steps}")
    if target_steps:
        print(f"目标 passes : {target_steps * global_batch_size / dataset_size:.6f}")
        print(f"调度 passes : {schedule_steps * global_batch_size / dataset_size:.6f}")
    print("=" * 60)
    print("\n执行命令:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print()

    if args.dry_run:
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = script_dir + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, cwd=script_dir, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
