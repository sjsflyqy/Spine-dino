"""Freeze downstream partitions without moving images or changing official tests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_pretrain_with_downstream import case_group, read_csv, write_csv, sha256

REPO = Path(__file__).resolve().parents[1]


def grouped_split(groups, seed, test_fraction, val_fraction):
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_test = round(len(keys) * test_fraction)
    n_val = round(len(keys) * val_fraction)
    if n_test + n_val >= len(keys):
        raise ValueError("No training groups remain")
    return {key: "test" if i < n_test else "val" if i < n_test + n_val else "train"
            for i, key in enumerate(keys)}


def make_splits(rows, seed=42):
    result = []
    for row in rows:
        if row["task"] not in {"02", "04", "05", "06"}:
            continue
        relative = Path(row["image"]).relative_to("Downstream/data").as_posix()
        group = case_group({**row, "relative": relative})
        if not group:
            raise ValueError(f"Cannot identify source group: {relative}")
        result.append(dict(task=row["task"], image=relative, split="", group=group,
                           label=row.get("label", ""), sha256=row["sha256"],
                           original_split=row.get("split", ""), reason=""))
    for task in ("02", "04", "05", "06"):
        current = [r for r in result if r["task"] == task]
        groups = defaultdict(list)
        for row in current:
            groups[row["group"]].append(row)
        if task == "02":
            for group, members in groups.items():
                if len(members) != 2 or {Path(r["image"]).parent.name for r in members} != {"AP", "LA_image"}:
                    raise ValueError(f"Expected one AP and LAT per BUU400 case: {group}")
            assigned = grouped_split(groups, seed, .15, .15)
            for row in current:
                row.update(split=assigned[row["group"]], reason="BUU400 paired case; 70/15/15; seed42")
        elif task == "04":
            official_test = {r["group"] for r in current if r["original_split"] == "test"}
            assigned = grouped_split(set(groups) - official_test, seed, 0, .2)
            for row in current:
                if row["original_split"] == "test":
                    row.update(split="test", reason="Preserved official test")
                elif row["group"] in official_test:
                    row.update(split="excluded", reason="Original-image group overlaps official test")
                else:
                    row.update(split=assigned[row["group"]], reason="Original case name groups incl. letter/flip variants; 80/20 of original train")
        elif task == "05":
            for row in current:
                parts = Path(row["image"]).parts
                split = "test" if "Test" in parts else "val" if "val" in parts else "train" if "train" in parts else None
                if split is None:
                    raise ValueError(f"Unknown task05 source split: {row['image']}")
                row.update(split=split, reason="Preserved existing nested source split")
            priority = {"train": 0, "val": 1, "test": 2}
            for members in groups.values():
                winning = max((r["split"] for r in members), key=priority.get)
                for row in members:
                    if row["split"] != winning:
                        row.update(split="excluded", reason=f"Same original image as {winning}; exclude re-encoded/augmented copy")
        else:
            for label in sorted({r["label"] for r in current}):
                subset = {r["group"] for r in current if r["label"] == label}
                assigned = grouped_split(subset, seed, .15, .15)
                for row in current:
                    if row["label"] == label:
                        row.update(split=assigned[row["group"]], reason="Class-stratified filename groups; 70/15/15; no clinical patient IDs available")
    # Fail closed for identical bytes assigned across active subsets or tasks.
    active = defaultdict(set)
    for row in result:
        if row["split"] != "excluded":
            active[row["sha256"]].add(row["split"])
    if any(len(splits) > 1 for splits in active.values()):
        raise ValueError("Identical image bytes cross the proposed splits")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "Downstream/data")
    parser.add_argument("--output", type=Path, default=REPO / "Downstream/splits/task02_04_05_06_seed42")
    args = parser.parse_args()
    manifest = args.data / "manifest_task02_06.csv"
    rows = make_splits(read_csv(manifest))
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "splits.csv", rows)
    summary = {"seed": 42, "manifest_sha256": sha256(manifest), "tasks": {}}
    for task in ("02", "04", "05", "06"):
        current = [r for r in rows if r["task"] == task]
        summary["tasks"][task] = {
            "images": dict(Counter(r["split"] for r in current)),
            "groups": {s: len({r["group"] for r in current if r["split"] == s}) for s in ("train", "val", "test", "excluded")},
            "classes": {label: dict(Counter(r["split"] for r in current if r["label"] == label)) for label in sorted({r["label"] for r in current}) if label},
        }
        directory = args.output / f"task{task}"
        directory.mkdir()
        for split in ("train", "val", "test", "excluded"):
            (directory / f"{split}.txt").write_text("".join(r["image"] + "\n" for r in current if r["split"] == split), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
