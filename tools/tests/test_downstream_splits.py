import unittest

from tools.build_pretrain_with_downstream import aasce_case, base_case_group, case_group, holdout_details
from tools.split_downstream_02_04_05_06 import make_splits


class SplitTests(unittest.TestCase):
    def fixture(self):
        rows = []

        def add(task, name, original="", label=""):
            rows.append(dict(task=task, image=f"Downstream/data/task{task}/{name}",
                             sample_id=name.rsplit("/", 1)[-1].split(".")[0],
                             split=original, label=label, sha256=f"hash-{task}-{name}"))
        for i in range(20):
            add("02", f"AP/{i:04}-F-020Y0.jpg")
            add("02", f"LA_image/{i:04}-F-020Y1.jpg")
            for letter in "AB":
                add("04", f"train/sunhl-1th-01-Jan-2017-{i} {letter} AP.jpg", "train")
            for label in ("normal", "scoliosis", "spondylolisthesis"):
                add("06", f"{label}/N{i},{label},F,20_1_0.jpg", label=label)
        add("04", "test/01-July-2019-1.jpg", "test")
        add("05", "Test/3000-F-020Y1.jpg", "test")
        add("05", "Train/images/train/3000-F-020Y1_jpg.rf.a.jpg", "train")
        add("05", "Train/images/val/3001-F-020Y1_jpg.rf.a.jpg", "train")
        add("05", "Train/images/train/3001-F-020Y1_jpg.rf.b.jpg", "train")
        add("05", "Train/images/train/3002-F-020Y1_jpg.rf.a.jpg", "train")
        return rows

    def test_fixed_group_and_official_splits(self):
        rows = self.fixture()
        result = make_splits(rows)
        self.assertEqual(result, make_splits(rows))
        for group in {r["group"] for r in result}:
            active = {r["split"] for r in result if r["group"] == group and r["split"] != "excluded"}
            self.assertEqual(len(active), 1)
        for row in result:
            if row["original_split"] == "test":
                self.assertEqual(row["split"], "test")
        task05 = [r for r in result if r["task"] == "05"]
        self.assertEqual(sum(r["split"] == "excluded" for r in task05), 2)
        for label in ("normal", "scoliosis", "spondylolisthesis"):
            self.assertEqual({r["split"] for r in result if r["label"] == label}, {"train", "val", "test"})

    def test_aasce_variant_grouping_and_evidence(self):
        expected = aasce_case("sunhl-1th-30-Dec-2016-159 A AP")
        self.assertEqual(expected, aasce_case("sunhl-1th-30-Dec-2016-159 B AP2_flip"))
        self.assertEqual(expected, base_case_group(dict(source="ningbo", original_relative_path="00012345_sunhl-1th-30-Dec-2016-159 C AP_flip.jpg")))
        records = [dict(task="04", sample_id="a", relative="task04/sunhl-1th-30-Dec-2016-159 A AP.jpg", sha256="a", split="train"),
                   dict(task="04", sample_id="b", relative="task04/sunhl-1th-30-Dec-2016-159 B AP.jpg", sha256="b", split="val")]
        hashes, groups, witnesses = holdout_details(records)
        self.assertEqual(hashes, {"a", "b"})
        self.assertIn(expected, groups)
        self.assertEqual(witnesses["a"]["conflict_evidence"], "case_group_connection")
        self.assertEqual(witnesses["a"]["blocked_by_split"], "val")
        self.assertEqual(witnesses["b"]["conflict_evidence"], "identical_sha256")


if __name__ == "__main__":
    unittest.main()
