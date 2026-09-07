"""Copy Tasks 02--06 into the portable downstream package without recoding."""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Downstream" / "data"

SOURCES = {
    "02": Path(r"H:\OpenSource_Data\02BUU400"),
    "03_images": Path(r"H:\OpenSource_Data\03vindr_Spinexr\test_png"),
    "03_annotations": Path(r"H:\OpenSource_Data\03vindr_Spinexr\annotations\test.csv"),
    "03_license": Path(r"H:\OpenSource_Data\03vindr_Spinexr\LICENSE.txt"),
    "04": Path(r"H:\OpenSource_Data\05AASCE 2019"),
    "05": Path(r"H:\OpenSource_Data\06Spondylolisthesis Vertebral Landmark"),
    "06": Path(r"H:\OpenSource_Data\07Scoliosis–Spondylolisthesis–Normal"),
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Existing file differs: {destination}")
        return
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    for source_file in tqdm(files, desc=f"Copy {source.name}"):
        copy_file(source_file, destination / source_file.relative_to(source))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_metadata(task: str, relative: Path) -> tuple[str, str, str]:
    parts = [part.lower() for part in relative.parts]
    split, view, label = "", "", ""
    if task == "02":
        view = "AP" if "ap" in parts else "LAT" if any(
            part.startswith("la_") for part in parts
        ) else ""
    elif task == "03":
        split = "test"
    elif task == "04":
        split = "train" if "train" in parts else "test" if "test" in parts else ""
        view = "AP"
    elif task == "05":
        split = "train" if "train" in parts else "test" if "test" in parts else ""
        view = "LAT"
    elif task == "06":
        for candidate in ("normalfinal", "scolfinal", "spondfinal"):
            if candidate in parts:
                label = candidate
                break
    return split, view, label


def build_manifest() -> None:
    rows = []
    for task in ("02", "03", "04", "05", "06"):
        task_root = DATA / f"task{task}"
        for image in sorted(
            path
            for path in task_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            relative = image.relative_to(task_root)
            split, view, label = infer_metadata(task, relative)
            rows.append(
                {
                    "task": task,
                    "sample_id": image.stem,
                    "split": split,
                    "view": view,
                    "label": label,
                    "image": image.relative_to(ROOT).as_posix(),
                    "sha256": sha256(image),
                }
            )
    output = DATA / "manifest_task02_06.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} image records to {output}")


def main() -> None:
    missing = [str(path) for path in SOURCES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing sources: {missing}")

    copy_tree(SOURCES["02"], DATA / "task02")
    copy_tree(SOURCES["03_images"], DATA / "task03" / "images")
    copy_file(
        SOURCES["03_annotations"],
        DATA / "task03" / "annotations" / "test.csv",
    )
    copy_file(SOURCES["03_license"], DATA / "task03" / "LICENSE.txt")
    copy_tree(SOURCES["04"], DATA / "task04")
    copy_tree(SOURCES["05"], DATA / "task05")
    copy_tree(SOURCES["06"], DATA / "task06")
    build_manifest()


if __name__ == "__main__":
    main()
