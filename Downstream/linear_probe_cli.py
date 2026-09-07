"""Shared command-line configuration overrides for downstream linear probes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_linear_probe_config(description: str) -> dict[str, Any]:
    """Load a YAML config and apply explicit experiment-specific overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path, help="Base YAML config")
    parser.add_argument(
        "--weights",
        type=Path,
        help="Extracted backbone .pth; overrides model.weights in the YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Experiment output directory; overrides output_dir in the YAML",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError(f"Invalid linear-probe config: {config_path}")

    if args.weights is not None:
        config["model"]["weights"] = str(args.weights.expanduser())
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir.expanduser())

    if not config["model"].get("weights"):
        raise ValueError("model.weights is required (in YAML or via --weights)")
    if not config.get("output_dir"):
        raise ValueError("output_dir is required (in YAML or via --output-dir)")
    return config
