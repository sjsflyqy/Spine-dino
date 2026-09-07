from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Downstream" / "data"

EXPECTED = {
    "02": 800,
    "03": 2077,
    "04": 579,
    "05": 714,
    "06": 338,
    "07": 1016,
    "08": 1207,
    "09": 1310,
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    rows_02_06 = read_csv(DATA / "manifest_task02_06.csv")
    rows_07_09 = read_csv(DATA / "manifest.csv")
    counts = Counter(row["task"] for row in rows_02_06 + rows_07_09)
    if dict(counts) != EXPECTED:
        raise RuntimeError(f"Counts differ: {dict(counts)} != {EXPECTED}")

    missing = [
        row["image"]
        for row in rows_02_06 + rows_07_09
        if not (ROOT / row["image"]).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images: {missing[:5]}")

    for required in (
        DATA / "task02" / "BUU-LSPINE_400_report.xlsx",
        DATA / "task03" / "annotations" / "test.csv",
        DATA / "task03" / "LICENSE.txt",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    coco_counts = {}
    for split, expected in (("train", 919), ("val", 253), ("test", 138)):
        path = DATA / "task09" / "annotations" / f"{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        coco_counts[split] = len(payload["images"])
        if coco_counts[split] != expected:
            raise RuntimeError(f"Task09 {split}: {coco_counts[split]} != {expected}")

    print(
        {
            "total_task_records": sum(counts.values()),
            "counts": dict(counts),
            "task09_splits": coco_counts,
        }
    )


if __name__ == "__main__":
    main()
