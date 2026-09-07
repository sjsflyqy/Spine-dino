"""Build image-directory and WebDataset representations of SpinePretrain-v1.

The builder preserves image bytes, creates stable source-prefixed keys, writes
DINOv2-compatible ``extra`` arrays, and emits uncompressed tar shards. It is
safe to rerun: completed image copies and completed shards are reused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

SOURCES = {
    "nhanes2": Path(r"H:\OpenSource_Data\01NHANES2_img"),
    "buu2000_ap": Path(r"H:\OpenSource_Data\02BUU2000\AP"),
    "buu2000_lat": Path(r"H:\OpenSource_Data\02BUU2000\LA"),
    "vindr_train": Path(r"H:\OpenSource_Data\03vindr_Spinexr\train_png"),
    "csxa": Path(r"H:\OpenSource_Data\04CSXA\datasets-PNG"),
    "nanning": Path(r"H:\Hospital_data\nanning3\png_data"),
    "ningbo": Path(
        r"H:\组里的数据集\脊柱数据集(脊柱侧弯)--无目录结构--MD5去重版"
        r"\private_data_use_for_pretrain"
    ),
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
EXPECTED_COUNTS = {
    "nhanes2": 16531,
    "buu2000_ap": 2000,
    "buu2000_lat": 2000,
    "vindr_train": 8389,
    "csxa": 4963,  # Local audited count; original planning note said 4,693.
    "nanning": 20558,  # Local audited count; original planning note said 20,588.
    "ningbo": 14493,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def guard_space(path: Path, minimum_free_gib: float) -> None:
    available = free_gib(path)
    if available < minimum_free_gib:
        raise RuntimeError(
            f"Free space safety stop: {available:.1f} GiB < "
            f"{minimum_free_gib:.1f} GiB at {path}"
        )


def collect_sources() -> dict[str, list[Path]]:
    collected = {}
    for source, root in SOURCES.items():
        if not root.exists():
            raise FileNotFoundError(root)
        images = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) != EXPECTED_COUNTS[source]:
            raise RuntimeError(
                f"{source}: found {len(images)}, expected {EXPECTED_COUNTS[source]}"
            )
        collected[source] = images
    return collected


def infer_view(source: str, path: Path) -> str:
    if source == "buu2000_ap":
        return "AP"
    if source == "buu2000_lat":
        return "LAT"
    upper = path.stem.upper()
    if "_AP_" in upper or upper.endswith("_AP"):
        return "AP"
    if "_LAT_" in upper or "_LA_" in upper or upper.endswith(("_LAT", "_LA")):
        return "LAT"
    return "unknown"


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Existing destination has wrong size: {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    # The audited sources and build target normally live on the same NTFS
    # volume. A hard link is a regular file entry (not a shortcut/symlink), is
    # read normally by training and upload tools, and avoids a redundant local
    # 75 GiB copy. Fall back to a byte copy when hard links are unavailable.
    try:
        os.link(source, destination)
        return
    except OSError:
        pass
    with source.open("rb") as src, partial.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(partial, destination)


def build_images(
    output: Path,
    collected: dict[str, list[Path]],
    minimum_free_gib: float,
) -> tuple[list[dict], list[dict]]:
    image_root = output / "spine_dino_dataset" / "train" / "spine"
    metadata_root = output / "metadata"
    image_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    excluded: list[dict] = []
    seen_hashes: dict[str, str] = {}
    processed = 0
    started = time.time()

    for source, paths in collected.items():
        root = SOURCES[source]
        for source_index, source_path in enumerate(paths, start=1):
            processed += 1
            if processed % 250 == 0:
                guard_space(output, minimum_free_gib)
                elapsed = max(time.time() - started, 0.001)
                print(
                    f"[images] {processed}/{sum(map(len, collected.values()))} "
                    f"({processed / elapsed:.1f} files/s), free={free_gib(output):.1f} GiB",
                    flush=True,
                )

            digest = sha256(source_path)
            relative_source = source_path.relative_to(root).as_posix()
            if digest in seen_hashes:
                excluded.append(
                    {
                        "source": source,
                        "original_relative_path": relative_source,
                        "sha256": digest,
                        "reason": "exact_duplicate",
                        "duplicate_of": seen_hashes[digest],
                    }
                )
                continue

            key = f"{source}_{source_index:08d}"
            extension = source_path.suffix.lower()
            destination = image_root / f"{key}{extension}"
            copy_atomic(source_path, destination)
            seen_hashes[digest] = key
            rows.append(
                {
                    "key": key,
                    "source": source,
                    "original_relative_path": relative_source,
                    "image_relative_path": destination.relative_to(output).as_posix(),
                    "dinov2_relative_path": f"spine/{destination.name}",
                    "extension": extension,
                    "size_bytes": source_path.stat().st_size,
                    "sha256": digest,
                    "view": infer_view(source, source_path),
                    "modality": "xray",
                }
            )

    write_csv(metadata_root / "manifest.csv", rows)
    write_csv(metadata_root / "excluded.csv", excluded)
    return rows, excluded


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_dinov2_extra(output: Path, rows: list[dict]) -> None:
    extra = output / "spine_dino_dataset" / "extra"
    extra.mkdir(parents=True, exist_ok=True)
    image_files = np.asarray([row["dinov2_relative_path"] for row in rows])
    class_ids = np.zeros(len(rows), dtype=np.int64)
    class_names = np.asarray(["spine"])
    np.save(extra / "image_files-TRAIN.npy", image_files)
    np.save(extra / "class-ids-TRAIN.npy", class_ids)
    np.save(extra / "class-names-TRAIN.npy", class_names)
    with (extra / "train_list.txt").open("w", encoding="utf-8") as stream:
        for relative_path in image_files:
            stream.write(f"{relative_path}\t0\n")


def tar_add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def build_shards(
    output: Path,
    rows: list[dict],
    *,
    max_samples: int,
    max_bytes: int,
    minimum_free_gib: float,
) -> list[dict]:
    shards_root = output / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    shard_rows = []
    cursor = 0
    shard_index = 0

    while cursor < len(rows):
        guard_space(output, minimum_free_gib)
        members = []
        accumulated = 0
        while cursor + len(members) < len(rows) and len(members) < max_samples:
            candidate = rows[cursor + len(members)]
            candidate_bytes = int(candidate["size_bytes"])
            if members and accumulated + candidate_bytes > max_bytes:
                break
            members.append(candidate)
            accumulated += candidate_bytes

        filename = f"train-{shard_index:06d}.tar"
        destination = shards_root / filename
        if destination.exists():
            # A completed shard is immutable. The checksum is recomputed below.
            print(f"[shards] reuse {filename}", flush=True)
        else:
            partial = destination.with_suffix(".tar.partial")
            with tarfile.open(partial, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for row in members:
                    image = output / row["image_relative_path"]
                    archive.add(
                        image,
                        arcname=f"{row['key']}{row['extension']}",
                        recursive=False,
                    )
                    metadata = {
                        "__key__": row["key"],
                        "source": row["source"],
                        "original_relative_path": row["original_relative_path"],
                        "view": row["view"],
                        "modality": row["modality"],
                        "sha256": row["sha256"],
                        "extension": row["extension"],
                    }
                    tar_add_bytes(
                        archive,
                        f"{row['key']}.json",
                        json.dumps(
                            metadata, ensure_ascii=False, separators=(",", ":")
                        ).encode("utf-8"),
                    )
            os.replace(partial, destination)

        shard_digest = sha256(destination)
        checksum_path = destination.with_suffix(".tar.sha256")
        checksum_path.write_text(
            f"{shard_digest}  {filename}\n", encoding="ascii"
        )
        shard_rows.append(
            {
                "shard": filename,
                "samples": len(members),
                "payload_bytes": accumulated,
                "tar_bytes": destination.stat().st_size,
                "sha256": shard_digest,
                "first_key": members[0]["key"],
                "last_key": members[-1]["key"],
            }
        )
        print(
            f"[shards] {filename}: {len(members)} samples, "
            f"{destination.stat().st_size / (1024**3):.2f} GiB, "
            f"free={free_gib(output):.1f} GiB",
            flush=True,
        )
        cursor += len(members)
        shard_index += 1

    write_csv(output / "metadata" / "shards.csv", shard_rows)
    return shard_rows


def build_report(
    output: Path,
    collected: dict[str, list[Path]],
    rows: list[dict],
    excluded: list[dict],
    shards: list[dict],
) -> None:
    report = {
        "version": "SpinePretrain-v1",
        "candidate_count": sum(map(len, collected.values())),
        "included_count": len(rows),
        "excluded_count": len(excluded),
        "source_candidate_counts": {
            source: len(paths) for source, paths in collected.items()
        },
        "source_included_counts": dict(Counter(row["source"] for row in rows)),
        "total_image_bytes": sum(int(row["size_bytes"]) for row in rows),
        "shard_count": len(shards),
        "total_shard_bytes": sum(int(row["tar_bytes"]) for row in shards),
        "image_representation": "original encoded bytes; no recoding",
        "shard_format": "uncompressed tar; image + JSON per sample",
    }
    (output / "metadata" / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"H:\SpinePretrain-v1"),
    )
    parser.add_argument("--max-samples-per-shard", type=int, default=1000)
    parser.add_argument("--max-shard-gib", type=float, default=2.0)
    parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    parser.add_argument("--skip-shards", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    guard_space(output, args.minimum_free_gib)
    collected = collect_sources()
    rows, excluded = build_images(output, collected, args.minimum_free_gib)
    build_dinov2_extra(output, rows)
    shards = []
    if not args.skip_shards:
        shards = build_shards(
            output,
            rows,
            max_samples=args.max_samples_per_shard,
            max_bytes=int(args.max_shard_gib * 1024**3),
            minimum_free_gib=args.minimum_free_gib,
        )
    build_report(output, collected, rows, excluded, shards)
    print(
        f"Completed SpinePretrain-v1: {len(rows)} images, "
        f"{len(shards)} shards at {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
