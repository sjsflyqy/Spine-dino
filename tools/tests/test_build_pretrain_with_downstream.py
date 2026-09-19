import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.build_pretrain_with_downstream import blocked_hashes, build, split_lookup, is_known_cervical


def record(task, identifier, digest, split, relative=None):
    return dict(task=task, sample_id=identifier, sha256=digest, split=split,
                relative=relative or f"task{task}/images/{identifier}.png")


class HoldoutTests(unittest.TestCase):
    def test_cervical_filter_uses_source_and_original_name(self):
        for source, name, excluded in [("csxa", "123.png", True),
                                       ("nhanes2", "C123.png", True),
                                       ("nhanes2", "folder\\c123.png", True),
                                       ("nhanes2", "L123.png", False),
                                       ("ningbo", "C123.png", False),
                                       ("nhanes2", "C_AP.png", False)]:
            self.assertEqual(is_known_cervical(dict(source=source, original_relative_path=name)), excluded)

    def test_transitive_case_and_cross_task_holdout(self):
        rows = [record("08", "1", "a", "train"),
                record("08", "gq1", "b", "train"),
                record("09", "alias", "b", "test"),
                record("07", "DR_2", "c", "train"),
                record("07", "DR_2", "d", "train"),
                record("09", "DR_2_s1_e12", "e", "val"),
                record("08", "3", "f", "train"),
                record("02", "0001-F-037Y0", "g", "pending")]
        self.assertEqual(blocked_hashes(rows), set("abcdeg"))

    def test_split_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(json.dumps({"splits": {"train": ["a"], "val": ["a"], "test": []}}))
            with self.assertRaisesRegex(ValueError, "overlap"):
                split_lookup(path)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.root, self.data = root / "pretrain", root / "Downstream/data"
        self.extra = self.root / "spine_dino_dataset/extra"
        self.output = self.root / "merged"
        self.extra.mkdir(parents=True)
        self.data.mkdir(parents=True)
        self.down_rows = []
        for task, identifier, split in [("07", "DR_1", "train"), ("07", "DR_2", "test"),
                                        ("08", "1", "train"), ("08", "2", "val"),
                                        ("09", "independent", "train")]:
            relative = f"task{task}/images/{identifier}.png"
            path = self.data / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"image {task} {identifier}".encode())
            self.down_rows.append(dict(task=task, sample_id=identifier,
                                       image="Downstream/data/" + relative,
                                       annotation=f"Downstream/data/task{task}/annotations/{split}.json",
                                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        self.write_csv(self.data / "manifest.csv", self.down_rows)
        (self.data / "manifest_task02_06.csv").write_text("task,sample_id,split,image,sha256\n")
        for split in ("train", "val", "test"):
            path = self.data / f"task09/annotations/{split}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"images": [{"file_name": "images/independent.png"}] if split == "train" else []}))
        self.splits = []
        for task, train, val, test in [("07", ["DR_1"], [], ["DR_2"]), ("08", ["1"], ["2"], [])]:
            path = root / f"split{task}.json"
            path.write_text(json.dumps({"splits": dict(train=train, val=val, test=test)}))
            self.splits.append(path)
        base_rows = []
        for name, content in [("safe", b"original safe image"), ("heldout", b"image 07 DR_2"),
                              ("duplicate", b"image 09 independent")]:
            path = self.root / f"spine_dino_dataset/train/spine/{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            base_rows.append(dict(dinov2_relative_path=f"spine/{name}.png", source="test",
                                  sha256=hashlib.sha256(content).hexdigest(), size_bytes=str(len(content))))
        self.write_csv(self.root / "metadata/manifest.csv", base_rows)
        np.save(self.extra / "image_files-TRAIN.npy", [r["dinov2_relative_path"] for r in base_rows])
        np.save(self.extra / "class-ids-TRAIN.npy", np.zeros(3, dtype=np.int64))

    @staticmethod
    def write_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def run_build(self, **kwargs):
        return build(self.root, self.data, self.extra, self.output, *self.splits, **kwargs)

    def test_end_to_end_dedup_holdout_and_loader_paths(self):
        report = self.run_build()
        self.assertEqual(report["base_decisions"], {"keep": 2, "excluded_holdout_or_pending": 1})
        self.assertEqual(report["added_unique_images"], 2)
        self.assertEqual(report["total_images"], 4)
        paths = np.load(self.output / "extra/image_files-TRAIN.npy").tolist()
        self.assertEqual(len(set(paths)), 4)
        self.assertNotIn("spine/heldout.png", paths)
        for path in paths:
            self.assertTrue((self.output / "train" / path).read_bytes())
        self.assertEqual(np.load(self.extra / "image_files-TRAIN.npy").shape[0], 3)
        self.assertEqual(np.load(self.output / "extra/class-ids-TRAIN.npy").tolist(), [0] * 4)
        with self.assertRaises(FileExistsError):
            self.run_build()

    def test_dry_run_and_actual_hash_verification(self):
        self.run_build(dry_run=True)
        self.assertFalse(self.output.exists())
        (self.data / "task08/images/1.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_build()

    def test_build_filters_cervical_without_intermediate_index(self):
        manifest = self.root / "metadata/manifest.csv"
        with manifest.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["source"] = "csxa"
        self.write_csv(manifest, rows)
        report = self.run_build()
        self.assertEqual(report["known_cervical_excluded"], 1)
        self.assertEqual(report["original_index_count"], 3)
        self.assertEqual(report["total_images"], 3)
        paths = np.load(self.output / "extra/image_files-TRAIN.npy").tolist()
        self.assertNotIn("spine/safe.png", paths)


if __name__ == "__main__":
    unittest.main()
