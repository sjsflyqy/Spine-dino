from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Downstream" / "data"


def main() -> None:
    manifest = DATA / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError("Run tools/build_portable_downstream.py first")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    missing = []
    for row in rows:
        for field in ("image", "annotation"):
            if not (ROOT / row[field]).exists():
                missing.append((field, row[field]))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} files; first: {missing[:5]}")

    counts = Counter(row["task"] for row in rows)
    expected = {"07": 1016, "08": 1207, "09": 1310}
    if dict(counts) != expected:
        raise RuntimeError(f"Unexpected task counts: {dict(counts)} != {expected}")

    coco_counts = {}
    for split, expected_count in (("train", 919), ("val", 253), ("test", 138)):
        path = DATA / "task09" / "annotations" / f"{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        coco_counts[split] = len(payload["images"])
        if coco_counts[split] != expected_count:
            raise RuntimeError(f"{split}: {coco_counts[split]} != {expected_count}")
        for image in payload["images"]:
            resolved = (path.parent.parent / image["file_name"]).resolve()
            if not resolved.exists():
                raise FileNotFoundError(resolved)

    print({"manifest": len(rows), "tasks": dict(counts), "task09": coco_counts})


if __name__ == "__main__":
    main()
