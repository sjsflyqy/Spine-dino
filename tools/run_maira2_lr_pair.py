"""Launch paired V025/F025 runs; --dry-run only prints the commands."""

import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("V025", "F025", "both"))
    parser.add_argument("--steps", type=int, help="Defaults to 29420 for two GPUs or 14710 for four GPUs")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "4,5,6,7"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/geotopo_dino"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",")]
    gpu_count = len(gpu_ids)
    if gpu_count not in (2, 4) or any(not item for item in gpu_ids) or len(set(gpu_ids)) != gpu_count:
        parser.error("Provide two or four distinct GPU IDs")
    args.gpus = ",".join(gpu_ids)
    global_batch = gpu_count * 8
    if args.steps is None:
        args.steps = 470720 // global_batch
    if args.steps <= 6000:
        parser.error("--steps must exceed the fixed 6000-step LR warmup")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "upstream/dinov2-main"), str(REPO_ROOT)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    variants = ("V025", "F025") if args.variant == "both" else (args.variant,)
    jobs = []
    for variant in variants:
        output = args.output_root / f"maira2_{variant}_{gpu_count}gpu_{args.steps}steps"
        if not output.is_absolute():
            output = REPO_ROOT / output
        if output.exists() and not args.dry_run:
            parser.error(f"Output already exists: {output}. Use a new --output-root; this launcher starts fresh.")
        command = [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nnodes=1", f"--nproc-per-node={gpu_count}",
            "--module", "methods.geotopo_dino.train.train",
            "--config-file", f"methods/geotopo_dino/configs/maira2_{variant.lower()}_{gpu_count}gpu.yaml",
            "--output-dir", str(output), "--no-resume",
            f"optim.total_iterations={args.steps}",
            f"optim.epochs={math.ceil(args.steps / 200)}",
            f"train.stop_after_iterations={args.steps}",
        ]
        jobs.append(command)

    if not args.dry_run:
        required = [
            REPO_ROOT / "weights/initialization/rad_dino_maira2_dinov2_format.pth",
            REPO_ROOT / "SpinePretrain-v1/spine_dino_no_known_cervical_with_downstream_v2/extra/image_files-TRAIN.npy",
        ]
        for path in required:
            if not path.is_file():
                parser.error(f"Required input is missing: {path}")
        # Isolate CUDA visibility in a child rather than initializing CUDA here.
        subprocess.run(
            [sys.executable, "-c", "import torch; n=torch.cuda.device_count(); "
             f"assert torch.cuda.is_available() and n == {gpu_count}, "
             f"f'Expected {gpu_count} visible CUDA GPUs; found {{n}}'"],
            env=env, cwd=REPO_ROOT, check=True,
        )

    print(f"Steps={args.steps}; global batch={global_batch}; sample presentations={args.steps * global_batch}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={args.gpus}", flush=True)
    print(f"PYTHONPATH={env['PYTHONPATH']}", flush=True)
    for command in jobs:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, env=env, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
