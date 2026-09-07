"""Unified entry point for the repository's DINOv2/DINOv3 teacher extractors."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a downstream backbone from a DINOv2/DINOv3 SSL checkpoint"
    )
    parser.add_argument("--architecture", required=True, choices=("dinov2", "dinov3"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source",
        default="teacher",
        choices=("teacher", "student"),
        help="DINOv2 source (teacher is recommended); DINOv3 supports teacher only",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")

    if args.architecture == "dinov2":
        script = REPO_ROOT / "upstream/dinov2-main/extract_backbone.py"
        command = [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--source",
            args.source,
        ]
        workdir = REPO_ROOT
    else:
        if args.source != "teacher":
            parser.error("DINOv3 extraction currently supports --source teacher only")
        script = REPO_ROOT / "upstream/dinov3-main/extract_dinov3_teacher.py"
        command = [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ]
        workdir = script.parent

    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=workdir, check=True)


if __name__ == "__main__":
    main()
