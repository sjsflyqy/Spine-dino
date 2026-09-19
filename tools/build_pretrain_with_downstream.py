"""Combine a SpineDINO index with downstream train images, excluding held-out overlaps.

Uses frozen existing task07/08 experiment splits and task09 COCO splits.
The CLI also uses the frozen task02/04/05/06 split CSV by default.
Images are referenced through relative directory symlinks, not copied or recoded.
Known cervical images are excluded directly from the original pretraining index;
no intermediate filtered index is required.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO / "Downstream/splits"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_known_cervical(row):
    if row["source"] == "csxa":
        return True
    stem = Path(row.get("original_relative_path", "").replace("\\", "/")).stem
    return row["source"] == "nhanes2" and re.fullmatch(r"C\d+", stem, re.I) is not None


def split_lookup(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload["splits"]) != {"train", "val", "test"}:
        raise ValueError(f"Expected train/val/test in {path}")
    lookup = {}
    for split, identifiers in payload["splits"].items():
        for identifier in identifiers:
            if identifier in lookup:
                raise ValueError(f"Duplicate/split overlap: {identifier} in {path}")
            lookup[identifier] = split
    for identifier in payload.get("excluded", {}):
        if identifier in lookup:
            raise ValueError(f"Excluded identifier also assigned: {identifier}")
        lookup[identifier] = "excluded"
    return lookup


def load_downstream(data, split07, split08, extra_splits=None):
    records = []
    lookups = {"07": split_lookup(split07), "08": split_lookup(split08)}
    for filename in ("manifest.csv", "manifest_task02_06.csv"):
        for row in read_csv(data / filename):
            task = row["task"]
            # Existing manifests are repository-relative, even for relocated data.
            relative = Path(row["image"]).relative_to("Downstream/data")
            path = (data / relative).resolve()
            if not path.is_relative_to(data.resolve()) or not path.is_file():
                raise ValueError(f"Missing or invalid downstream image: {path}")
            if task in lookups:
                split = lookups[task].get(row["sample_id"])
                if split is None:
                    raise ValueError(f"Image missing from frozen task{task} split: {row['sample_id']}")
            elif task == "09":
                split = Path(row["annotation"]).stem
                if split not in {"train", "val", "test"}:
                    raise ValueError(f"Invalid task09 split: {split}")
            else:
                # Original train directories may still contain downstream validation images.
                split = "test" if row.get("split") == "test" else "pending"
            records.append(dict(task=task, sample_id=row["sample_id"],
                                relative=relative.as_posix(), path=str(path),
                                sha256=row["sha256"], split=split))
    for task, lookup in lookups.items():
        if set(lookup) != {r["sample_id"] for r in records if r["task"] == task}:
            raise ValueError(f"Frozen task{task} split and manifest identifiers differ")
    # Check that the manifest still agrees with the actual COCO definitions.
    for split in ("train", "val", "test"):
        coco = json.loads((data / f"task09/annotations/{split}.json").read_text())
        expected = {str((data / "task09" / item["file_name"]).resolve()) for item in coco["images"]}
        actual = {r["path"] for r in records if r["task"] == "09" and r["split"] == split}
        if expected != actual:
            raise ValueError(f"Task09 {split} manifest and COCO paths differ")
    if extra_splits:
        overrides = {}
        for row in read_csv(extra_splits):
            task = row["task"].zfill(2)
            key = (task, row["image"])
            if task not in {"02", "04", "05", "06"} or row["split"] not in {"train", "val", "test", "excluded"} or key in overrides:
                raise ValueError(f"Invalid/duplicate additional split: {row}")
            overrides[key] = row["split"]
        tasks = {task for task, _ in overrides}
        expected = {(r["task"], r["relative"]) for r in records if r["task"] in tasks}
        if expected != set(overrides):
            raise ValueError("Additional split CSV must assign every image of each supplied task exactly once")
        for row in records:
            key = (row["task"], row["relative"])
            if key in overrides:
                if row["split"] == "test" and overrides[key] != "test":
                    raise ValueError("Cannot reassign an official test image")
                row["split"] = overrides[key]
    return records


def case_group(row):
    """Only use known naming conventions, scoped to their source cohorts."""
    if row["task"] == "07":
        return "whole_spine:" + row["sample_id"]
    if row["task"] == "08":
        match = re.fullmatch(r"(?:gq|gs)?(\d+)", row["sample_id"], re.I)
        return "lumbar:" + (str(int(match[1])) if match else row["sample_id"])
    if row["task"] == "09":
        match = re.fullmatch(r"(DR_\d+)_s\d+_e\d+", Path(row["relative"]).stem)
        if match:
            # Conservatively link numbered crops to their corresponding whole-spine case.
            return "whole_spine:" + match[1]
    if row["task"] == "02":
        return buu_case(row["sample_id"])
    if row["task"] == "04":
        return aasce_case(Path(row["relative"]).stem)
    if row["task"] == "05":
        stem = Path(row["relative"]).stem.split("_jpg.rf.")[0]
        return buu_case(stem) or "task05:" + stem
    if row["task"] == "06":
        return "task06:" + re.sub(r"(?:_\d+)+$", "", Path(row["relative"]).stem).lower()
    return None


def buu_case(stem):
    match = re.fullmatch(r"(\d+-[FM]-\d+Y)[01]", stem, re.I)
    return "buu:" + match[1].upper() if match else None


def aasce_case(stem):
    stem = re.sub(r"_flip$", "", stem, flags=re.I)
    match = re.match(r"(sunhl-1th-\d+-[A-Za-z]+-\d+-\d+)(?:\s|$)", stem)
    return "aasce:" + (match[1] if match else stem)


def base_case_group(row):
    stem = Path(row["original_relative_path"]).stem
    if row["source"] in {"buu2000_ap", "buu2000_lat"}:
        return buu_case(stem)
    if row["source"] == "ningbo":
        stem = re.sub(r"^\d{8}_", "", stem)
        if stem.startswith(("sunhl-1th-", "01-July-2019-")):
            return aasce_case(stem)
    return None


def holdout_details(records):
    """Return blocked hashes, group propagation and a concrete witness per hash."""
    parent = {r["sha256"]: r["sha256"] for r in records}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    groups = {}
    for row in records:
        group = case_group(row)
        if group:
            previous = groups.setdefault(group, row["sha256"])
            parent[find(row["sha256"])] = find(previous)
    blocked = {find(r["sha256"]) for r in records if r["split"] != "train"}
    hashes = {digest for digest in parent if find(digest) in blocked}
    anchors, exact = defaultdict(list), defaultdict(list)
    for row in records:
        if row["split"] != "train":
            anchors[find(row["sha256"])].append(row)
            exact[row["sha256"]].append(row)
    witnesses = {}
    for row in records:
        digest = row["sha256"]
        if digest not in hashes:
            continue
        candidates = exact[digest] or anchors[find(digest)]
        witness = sorted(candidates, key=lambda r: ({"test": 0, "val": 1, "excluded": 2, "pending": 3}[r["split"]], r["task"], r["relative"]))[0]
        witnesses[digest] = dict(
            conflict_evidence="identical_sha256" if exact[digest] else "case_group_connection",
            blocked_by_task=witness["task"], blocked_by_split=witness["split"],
            blocked_by_image=witness["relative"], blocked_by_sha256=witness["sha256"],
        )
    blocked_groups = {case_group(r) for r in records if r["sha256"] in hashes and case_group(r)}
    return hashes, blocked_groups, witnesses


def blocked_hashes(records):
    return holdout_details(records)[0]


def build(root, data, base_extra, output, split07, split08, *, extra_splits=None, dry_run=False):
    root, data, base_extra, output = [Path(p).resolve() for p in (root, data, base_extra, output)]
    if output.exists() and not dry_run:
        raise FileExistsError(f"Use a new output directory: {output}")
    records = load_downstream(data, split07, split08, extra_splits)
    # Verify actual downstream files against the audited manifest, once per path.
    expected = {}
    for row in records:
        if row["path"] in expected and expected[row["path"]] != row["sha256"]:
            raise ValueError(f"Conflicting manifest hashes: {row['path']}")
        expected[row["path"]] = row["sha256"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for path, actual in zip(expected, pool.map(sha256, expected)):
            if actual != expected[path]:
                raise ValueError(f"Downstream SHA-256 mismatch: {path}")
    blocked, blocked_groups, witnesses = holdout_details(records)
    manifest = read_csv(root / "metadata/manifest.csv")
    by_path = {r["dinov2_relative_path"]: r for r in manifest}
    paths = np.load(base_extra / "image_files-TRAIN.npy", allow_pickle=False)
    targets = np.load(base_extra / "class-ids-TRAIN.npy", allow_pickle=False)
    if paths.ndim != 1 or targets.shape != paths.shape or np.any(targets != 0):
        raise ValueError("Expected aligned single-class SpineDINO image/target arrays")
    if len(set(paths.tolist())) != len(paths) or len(by_path) != len(manifest):
        raise ValueError("Duplicate input paths")
    original_index_count = len(paths)
    paths = np.asarray([path for path in paths.tolist() if not is_known_cervical(by_path[path])], dtype=str)
    known_cervical_excluded = original_index_count - len(paths)
    selected, seen, base_audit, downstream_audit = [], {}, [], []
    for path in paths.tolist():
        row = by_path[path]
        digest = row["sha256"]
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "spine":
            raise ValueError(f"Invalid base path: {path}")
        image = root / "spine_dino_dataset/train" / relative
        if image.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Base image size mismatch: {image}")
        group = base_case_group(row) if "original_relative_path" in row else None
        decision = "excluded_holdout_or_pending" if digest in blocked else "excluded_case_group" if group and group in blocked_groups else "duplicate" if digest in seen else "keep"
        if decision == "keep":
            seen[digest] = path
            selected.append(path)
        base_audit.append(dict(path=path, source=row["source"], sha256=digest,
                              case_group=group or "", decision=decision))
    for row in records:
        digest = row["sha256"]
        if row["split"] != "train":
            decision = "not_train"
        elif digest in blocked:
            decision = "excluded_cross_split_or_case_overlap"
        elif digest in seen:
            decision = "already_present"
        else:
            decision = "added"
            path = "downstream/" + row["relative"]
            selected.append(path)
            seen[digest] = path
        evidence = witnesses.get(digest, {}) if decision == "excluded_cross_split_or_case_overlap" else {}
        downstream_audit.append({**row, "decision": decision, "selected_path": seen.get(digest, ""),
                                **{k: evidence.get(k, "") for k in ("conflict_evidence", "blocked_by_task", "blocked_by_split", "blocked_by_image", "blocked_by_sha256")}})
    if not selected or set(seen) & blocked:
        raise ValueError("Empty selection or holdout contamination")
    report = {
        "base_extra": str(base_extra), "base_input_count": len(paths),
        "original_index_count": original_index_count,
        "known_cervical_excluded": known_cervical_excluded,
        "base_decisions": dict(Counter(r["decision"] for r in base_audit)),
        "downstream_decisions": {task: dict(Counter(r["decision"] for r in downstream_audit if r["task"] == task)) for task in sorted({r["task"] for r in records})},
        "added_unique_images": sum(r["decision"] == "added" for r in downstream_audit),
        "total_images": len(selected), "blocked_unique_hashes": len(blocked),
        "blocked_case_groups": len(blocked_groups),
        "selected_blocked_hash_overlap": len(set(seen) & blocked),
        "split07_sha256": sha256(split07), "split08_sha256": sha256(split08),
        "extra_splits_sha256": sha256(extra_splits) if extra_splits else None,
        "base_index_sha256": sha256(base_extra / "image_files-TRAIN.npy"),
        "base_manifest_sha256": sha256(root / "metadata/manifest.csv"),
        "output_root": str(output),
        "dataset_path": f"SpineDINO:split=TRAIN:root={output.as_posix()}:extra={(output / 'extra').as_posix()}",
        "limitations": ["Base SHA-256 values come from build manifest; file sizes checked, base bytes not rehashed",
                        "Checks exact bytes and known case groups; unknown patient links or re-encoded near duplicates may remain",
                        "Direct-image mode only; no WebDataset shard filtering",
                        "Known CSXA and NHANES cervical images are excluded; mixed-source and downstream region labels are not inferred"],
    }
    if dry_run:
        return report
    output.mkdir(parents=True, exist_ok=False)
    (output / "train").mkdir()
    for name, target in {"spine": root / "spine_dino_dataset/train/spine", "downstream": data}.items():
        (output / "train" / name).symlink_to(os.path.relpath(target, output / "train"), target_is_directory=True)
    extra = output / "extra"
    extra.mkdir()
    np.save(extra / "image_files-TRAIN.npy", np.asarray(selected, dtype=str))
    np.save(extra / "class-ids-TRAIN.npy", np.zeros(len(selected), dtype=np.int64))
    np.save(extra / "class-names-TRAIN.npy", np.asarray(["spine"]))
    (extra / "train_list.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    write_csv(extra / "base_audit.csv", base_audit)
    write_csv(extra / "downstream_audit.csv", downstream_audit)
    for task, source in (("07", split07), ("08", split08)):
        (extra / f"task{task}_split.json").write_bytes(Path(source).read_bytes())
    if extra_splits:
        (extra / "extra_splits.csv").write_bytes(Path(extra_splits).read_bytes())
    write_csv(extra / "downstream_splits.csv", [{k: r[k] for k in ("task", "sample_id", "relative", "sha256", "split")} for r in records])
    (extra / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in selected:
        if not (output / "train" / path).is_file():
            raise FileNotFoundError(output / "train" / path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "SpinePretrain-v1")
    parser.add_argument("--downstream-data", type=Path, default=REPO / "Downstream/data")
    parser.add_argument("--base-extra", type=Path, help="Defaults to the original extra index; known cervical images are filtered automatically")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split07", type=Path, default=SPLIT_DIR / "task07_seed42.json")
    parser.add_argument("--split08", type=Path, default=SPLIT_DIR / "task08_seed42.json")
    parser.add_argument("--extra-splits", type=Path,
                        default=SPLIT_DIR / "task02_04_05_06_seed42/splits.csv",
                        help="Complete CSV for task02/04/05/06: task,image,split; image relative to downstream data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = build(args.root, args.downstream_data,
                   args.base_extra or args.root / "spine_dino_dataset/extra",
                   args.output or args.root / "spine_dino_no_known_cervical_with_downstream_v2",
                   args.split07, args.split08, extra_splits=args.extra_splits, dry_run=args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
