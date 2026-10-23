"""Explicit two-rank integration test; synthetic tiny checkpoints, never public long training.

python -m baseline_pretrain.tests.distributed_smoke --root /tmp/baseline-smoke --device cpu
CUDA_VISIBLE_DEVICES=0,1 python -m baseline_pretrain.tests.distributed_smoke --root /tmp/baseline-smoke-cuda --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from baseline_pretrain.checkpointing import load_tensor_checkpoint
from baseline_pretrain.tests.test_pretrain import write_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--methods", nargs="+", default=["dax", "medical_mae", "chexfound"])
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Smoke directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    reports = []
    for method in args.methods:
        c, config = write_fixture(root, method)
        c["training"].update(expected_world_size=2, parallel="fsdp" if method == "chexfound" and args.device == "cuda" else "ddp")
        if args.device == "cuda":
            c["training"]["precision"] = "bf16"
        config.write_text(yaml.safe_dump(c))
        full, resumed = root / f"{method}_full", root / f"{method}_resume"
        base = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                "--nproc-per-node=2", "--module", "baseline_pretrain.train", "--config", str(config), "--device", args.device]
        commands = [base + ["--output-dir", str(full)],
                    base + ["--output-dir", str(resumed), "--max-updates", "2"],
                    base + ["--output-dir", str(resumed), "--resume", str(resumed / "checkpoint_u00000002.pt")]]
        for index, command in enumerate(commands):
            log = root / f"{method}_{index}.log"
            print(f"Testing {method} phase {index + 1}/3; log={log}", flush=True)
            with log.open("w") as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                print(log.read_text()[-12000:], flush=True)
                raise RuntimeError(f"Distributed smoke failed: {method} phase {index + 1}")
        a = load_tensor_checkpoint(full / "checkpoint_u00000004.pt")
        b = load_tensor_checkpoint(resumed / "checkpoint_u00000004.pt")
        assert any(g["apply_decay"] for g in b["optimizer"]["param_groups"]), "Matrix weight decay was lost after sharding"
        assert any(not g["apply_decay"] for g in b["optimizer"]["param_groups"]), "Bias/norm exclusion was lost after sharding"
        assert len({g["lr_multiplier"] for g in b["optimizer"]["param_groups"]}) > 1, "Backbone/head LR grouping was lost"
        max_difference = 0.0
        for part in ("student", "teacher", "objective"):
            if a[part] is not None:
                for key, value in a[part].items():
                    difference = float((value.float() - b[part][key].float()).abs().max())
                    max_difference = max(max_difference, difference)
                    torch.testing.assert_close(value, b[part][key], atol=2e-6 if args.device == "cuda" else 0,
                                               rtol=2e-5 if args.device == "cuda" else 0)
        report = {"method": method, "parallel": c["training"]["parallel"], "device": args.device,
                  "resume_max_difference": max_difference, "updates": b["optimizer_updates"]}
        reports.append(report)
        print(json.dumps(report), flush=True)
    (root / "report.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
