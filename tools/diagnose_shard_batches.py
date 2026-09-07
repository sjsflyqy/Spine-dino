#!/usr/bin/env python3
"""Diagnose source mixing in SpinePretrain WebDataset tar shards.

This reads JSON members only (never decodes images) and simulates the current
resampled-shard plus finite shuffle-buffer sampling policy.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
import tarfile
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards",
        required=True,
        help="Shard glob, for example: /data/SpinePretrain-v1/shards/train-*.tar",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Per-GPU batch size")
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--shuffle-buffer", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Number of the least-mixed batches to print",
    )
    return parser.parse_args()


def read_sources(shard: str) -> list[str]:
    sources: list[str] = []
    with tarfile.open(shard, mode="r:") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".json"):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            metadata = json.load(stream)
            source = str(metadata.get("source", "unknown"))
            sources.append(source)
    if not sources:
        raise RuntimeError(f"No JSON metadata found in shard: {shard}")
    return sources


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.batches <= 0 or args.shuffle_buffer <= 0:
        raise ValueError("batch-size, batches and shuffle-buffer must be positive")

    shards = sorted(glob.glob(args.shards))
    if not shards:
        raise FileNotFoundError(f"No shards match: {args.shards}")

    print(f"Found {len(shards)} shards. Reading JSON metadata only...")
    shard_sources = {shard: read_sources(shard) for shard in shards}
    total_physical = sum(map(len, shard_sources.values()))
    physical_counts = Counter(
        source for sources in shard_sources.values() for source in sources
    )
    homogeneous = sum(len(set(sources)) == 1 for sources in shard_sources.values())

    rng = random.Random(args.seed)
    buffer: list[str] = []

    def input_stream():
        while True:
            shard = rng.choice(shards)  # approximates WebDataset resampled=True
            yield from shard_sources[shard]

    stream = input_stream()
    while len(buffer) < args.shuffle_buffer:
        buffer.append(next(stream))

    batches: list[Counter[str]] = []
    sampled_counts: Counter[str] = Counter()
    for _ in range(args.batches):
        counts: Counter[str] = Counter()
        for _ in range(args.batch_size):
            index = rng.randrange(len(buffer))
            source = buffer[index]
            buffer[index] = next(stream)
            counts[source] += 1
            sampled_counts[source] += 1
        batches.append(counts)

    single_source = sum(len(counts) == 1 for counts in batches)
    distinct = [len(counts) for counts in batches]
    dominant = [max(counts.values()) / args.batch_size for counts in batches]

    print("\nPhysical dataset")
    print(f"  samples: {total_physical}")
    print(f"  homogeneous shards: {homogeneous}/{len(shards)}")
    for source, count in physical_counts.most_common():
        print(f"  {source:20s} {count:7d}  {count / total_physical:7.2%}")

    print("\nSimulated training batches")
    print(f"  batches: {args.batches}, per-GPU batch size: {args.batch_size}")
    print(f"  shuffle buffer: {args.shuffle_buffer}, seed: {args.seed}")
    print(
        f"  single-source batches: {single_source}/{args.batches} "
        f"({single_source / args.batches:.2%})"
    )
    print(f"  mean distinct sources/batch: {statistics.mean(distinct):.2f}")
    print(f"  minimum distinct sources/batch: {min(distinct)}")
    print(f"  mean dominant-source share: {statistics.mean(dominant):.2%}")
    print(f"  maximum dominant-source share: {max(dominant):.2%}")
    print("  sampled source distribution:")
    sampled_total = sum(sampled_counts.values())
    for source, count in sampled_counts.most_common():
        print(f"    {source:18s} {count:7d}  {count / sampled_total:7.2%}")

    print("\nLeast-mixed example batches")
    ranked = sorted(
        enumerate(batches, start=1),
        key=lambda item: (len(item[1]), -max(item[1].values())),
    )
    for batch_index, counts in ranked[: args.examples]:
        composition = ", ".join(
            f"{source}={count}" for source, count in counts.most_common()
        )
        print(f"  batch {batch_index:4d}: {composition}")

    print(
        "\nInterpretation: occasional source-heavy batches are normal. Repeated "
        "single-source batches or a very high dominant-source share indicate that "
        "mixed shards (global shuffle before packing) would be preferable."
    )


if __name__ == "__main__":
    main()
