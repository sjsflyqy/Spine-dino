"""Build a deduplicated, server-portable package for Tasks 07, 08 and 09."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "Downstream" / "data"

TASK07_SOURCE = Path(r"H:\组里的数据集\new_data\脊柱数据集(脊柱侧弯)")
TASK08_SOURCE = Path(
    r"H:\组里的数据集\new_data\脊柱数据集(腰椎滑脱)\私有数据集\data2"
)
TASK09_SOURCE = Path(
    r"H:\论文源码备份\椎骨编号识别\Vert_detect\vert_all"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Existing file differs: {destination}")
        return
    shutil.copy2(source, destination)


def stripped_labelme(source: Path, destination: Path, image_path: str) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["imageData"] = None
    payload["imagePath"] = image_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def build_task07(rows: list[dict]) -> None:
    for view in ("ap", "lat"):
        image_dir = TASK07_SOURCE / view / "rename_data"
        label_dir = TASK07_SOURCE / view / "rename_json"
        images = sorted(image_dir.glob("*.jpg"))
        for image in tqdm(images, desc=f"Task07 {view}"):
            destination = OUTPUT / "task07" / view / "images" / image.name
            label_source = label_dir / f"{image.stem}.json"
            label_destination = (
                OUTPUT / "task07" / view / "labels" / f"{image.stem}.json"
            )
            copy_file(image, destination)
            if not label_source.exists():
                raise FileNotFoundError(label_source)
            stripped_labelme(
                label_source,
                label_destination,
                f"../images/{image.name}",
            )
            rows.append(
                {
                    "task": "07",
                    "sample_id": image.stem,
                    "view": view,
                    "image": destination.relative_to(REPO_ROOT).as_posix(),
                    "annotation": label_destination.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256(destination),
                }
            )


def build_task08(rows: list[dict]) -> dict[str, Path]:
    image_dir = TASK08_SOURCE / "paired_images"
    label_dir = TASK08_SOURCE / "paired_labels"
    index: dict[str, Path] = {}
    for image in tqdm(sorted(image_dir.glob("*.png")), desc="Task08"):
        destination = OUTPUT / "task08" / "images" / image.name
        label_source = label_dir / f"{image.stem}.json"
        label_destination = OUTPUT / "task08" / "labels" / f"{image.stem}.json"
        copy_file(image, destination)
        if not label_source.exists():
            raise FileNotFoundError(label_source)
        stripped_labelme(
            label_source,
            label_destination,
            f"../images/{image.name}",
        )
        digest = sha256(destination)
        index[digest] = destination
        rows.append(
            {
                "task": "08",
                "sample_id": image.stem,
                "view": "lateral",
                "image": destination.relative_to(REPO_ROOT).as_posix(),
                "annotation": label_destination.relative_to(REPO_ROOT).as_posix(),
                "sha256": digest,
            }
        )
    return index


def build_task09(rows: list[dict], task08_hashes: dict[str, Path]) -> None:
    source_images = TASK09_SOURCE / "images"
    annotation_output = OUTPUT / "task09" / "annotations"
    annotation_output.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        source_json = TASK09_SOURCE / "annotations" / f"{split}.json"
        coco = json.loads(source_json.read_text(encoding="utf-8"))
        for item in tqdm(coco["images"], desc=f"Task09 {split}"):
            source = source_images / item["file_name"]
            if not source.exists():
                raise FileNotFoundError(source)
            digest = sha256(source)
            if digest in task08_hashes:
                destination = task08_hashes[digest]
            else:
                destination = OUTPUT / "task09" / "images" / source.name
                copy_file(source, destination)
            item["file_name"] = Path(
                os.path.relpath(destination, OUTPUT / "task09")
            ).as_posix()
            rows.append(
                {
                    "task": "09",
                    "sample_id": str(item["id"]),
                    "view": "lateral",
                    "image": destination.relative_to(REPO_ROOT).as_posix(),
                    "annotation": (
                        annotation_output / f"{split}.json"
                    ).relative_to(REPO_ROOT).as_posix(),
                    "sha256": digest,
                }
            )
        (annotation_output / f"{split}.json").write_text(
            json.dumps(coco, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()

    for source in (TASK07_SOURCE, TASK08_SOURCE, TASK09_SOURCE):
        if not source.exists():
            raise FileNotFoundError(source)

    rows: list[dict] = []
    build_task07(rows)
    task08_hashes = build_task08(rows)
    build_task09(rows, task08_hashes)

    manifest = OUTPUT / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Built {len(rows)} task records at {OUTPUT}")


if __name__ == "__main__":
    main()
