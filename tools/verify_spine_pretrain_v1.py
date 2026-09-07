from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "SpinePretrain-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "SpinePretrain-v1 directory. Defaults to the copy inside the "
            "repository, so the script works on both Windows and Linux."
        ),
    )
    parser.add_argument("--full-image-hash", action="store_true")
    parser.add_argument("--full-shard-hash", action="store_true")
    parser.add_argument(
        "--check-shards",
        action="store_true",
        help="Require and structurally validate every shard listed in metadata/shards.csv.",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    rows = read_csv(root / "metadata" / "manifest.csv")
    excluded_path = root / "metadata" / "excluded.csv"
    excluded = read_csv(excluded_path) if excluded_path.stat().st_size else []
    image_files = np.load(
        root / "spine_dino_dataset" / "extra" / "image_files-TRAIN.npy",
        allow_pickle=False,
    )
    class_ids = np.load(
        root / "spine_dino_dataset" / "extra" / "class-ids-TRAIN.npy",
        allow_pickle=False,
    )
    if len(rows) != len(image_files) or len(rows) != len(class_ids):
        raise RuntimeError("Manifest and DINOv2 extra array lengths differ")
    if np.any(class_ids != 0):
        raise RuntimeError("SpineDINO pseudo class IDs must all be zero")

    for index, row in enumerate(rows):
        image = root / row["image_relative_path"]
        if not image.exists():
            raise FileNotFoundError(image)
        if image.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"Size mismatch: {image}")
        if image_files[index] != row["dinov2_relative_path"]:
            raise RuntimeError(f"Extra path mismatch at index {index}")
        if args.full_image_hash and sha256(image) != row["sha256"]:
            raise RuntimeError(f"Hash mismatch: {image}")

    shard_rows = read_csv(root / "metadata" / "shards.csv")
    check_shards = args.check_shards or args.full_shard_hash
    if check_shards:
        shard_samples = 0
        for shard_row in shard_rows:
            shard = root / "shards" / shard_row["shard"]
            if not shard.exists():
                raise FileNotFoundError(shard)
            if args.full_shard_hash and sha256(shard) != shard_row["sha256"]:
                raise RuntimeError(f"Shard hash mismatch: {shard}")
            with tarfile.open(shard, "r") as archive:
                members = [member for member in archive.getmembers() if member.isfile()]
            expected_members = int(shard_row["samples"]) * 2
            if len(members) != expected_members:
                raise RuntimeError(
                    f"{shard.name}: {len(members)} members != {expected_members}"
                )
            shard_samples += int(shard_row["samples"])
        if shard_samples != len(rows):
            raise RuntimeError(f"Shard samples {shard_samples} != images {len(rows)}")

    report = {
        "images": len(rows),
        "excluded": len(excluded),
        "sources": dict(Counter(row["source"] for row in rows)),
        "expected_shards": len(shard_rows),
        "shards_checked": check_shards,
        "full_image_hash": args.full_image_hash,
        "full_shard_hash": args.full_shard_hash,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
