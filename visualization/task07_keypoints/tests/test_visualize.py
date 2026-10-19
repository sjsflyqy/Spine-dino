"""Test original-coordinate recovery, semantic connections, and saved-results CLI."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from visualization.task07_keypoints.visualize import (
    GREEN, RED, YELLOW, check_annotation, draw_overlay, load_predictions,
    main, original_coordinates, select_records, validate_points,
)


def record(identifier="DR_249"):
    gt = [[0.1, 0.2]] * 68
    prediction = [[0.2, 0.2]] * 68
    return {"id": identifier, **{f"{view}_{kind}_normalized": value
                                for view in ("ap", "lat") for kind, value in (("gt", gt), ("pred", prediction))}}


class CoordinateTest(unittest.TestCase):
    def test_width_height_convention_and_display_only_edge_clipping(self):
        normalized = np.array([[0.5, 0.25], [1., 1.]])
        mapped = original_coordinates(normalized, (101, 203))
        np.testing.assert_allclose(mapped, [[50.5, 50.75], [101, 203]])
        draw_overlay(Image.new("RGB", (101, 203)), mapped, mapped)
        # Original exported coordinates remain intact even at the outer edge.
        np.testing.assert_array_equal(mapped[-1], [101, 203])

    def test_green_gt_red_prediction_yellow_semantic_correspondence(self):
        # Index pairing produces long horizontal lines. Nearest-neighbour
        # matching would instead swap identities and produce short verticals.
        gt = np.array([[100, 100], [200, 200]], dtype=float)
        prediction = np.array([[200, 100], [100, 200]], dtype=float)
        overlay = draw_overlay(Image.new("RGB", (1000, 500)), gt, prediction)
        self.assertEqual(overlay.size, (1000, 500))
        self.assertEqual(overlay.getpixel((100, 100)), GREEN)
        self.assertEqual(overlay.getpixel((200, 100)), RED)
        self.assertEqual(overlay.getpixel((150, 100)), YELLOW)
        self.assertEqual(overlay.getpixel((150, 200)), YELLOW)
        self.assertEqual(overlay.getpixel((100, 150)), (0, 0, 0))

    def test_annotation_dimensions_and_training_corner_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "DR_249.json"
            shapes = [{"label": str(label), "points": [point]}
                      for label in range(1, 18)
                      for point in ((30, 40), (10, 20), (30, 20), (10, 40))]
            path.write_text(json.dumps({"imageWidth": 100, "imageHeight": 200, "shapes": shapes}))
            ordered = np.tile([[0.1, 0.1], [0.3, 0.1], [0.1, 0.2], [0.3, 0.2]], (17, 1))
            self.assertIn("verified", check_annotation(path, (100, 200), ordered))
            with self.assertRaisesRegex(ValueError, "dimensions"):
                check_annotation(path, (200, 100), ordered)
            with self.assertRaisesRegex(ValueError, "point ordering"):
                check_annotation(path, (100, 200), ordered[::-1])

    def test_malformed_predictions_and_case_selection(self):
        for points in (np.zeros((67, 2)), np.full((68, 2), np.nan), np.ones((68, 2)) * 1.5):
            with self.assertRaises(ValueError):
                validate_points(points, "test")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_predictions.json"
            path.write_text(json.dumps([record(), record("DR_250")]))
            indexed = load_predictions(path)
            self.assertEqual(select_records(indexed, ["DR_250"])[0]["id"], "DR_250")
            self.assertEqual(len(select_records(indexed, limit=1)), 1)
            with self.assertRaisesRegex(ValueError, "not present"):
                select_records(indexed, ["DR_999"])
            path.write_text(json.dumps([record(), record()]))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_predictions(path)


class SavedResultCliTest(unittest.TestCase):
    def test_single_model_export_and_list_without_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            experiment.mkdir()
            (experiment / "test_predictions.json").write_text(json.dumps([record()]))
            data = root / "data"
            for view, size in (("ap", (1000, 500)), ("lat", (600, 900))):
                directory = data / view / "images"
                directory.mkdir(parents=True)
                Image.new("RGB", size).save(directory / "DR_249.jpg")
            output = root / "visualization"
            base_args = ["visualize", "--result-dir", str(experiment), "--data-root", str(data)]
            with patch("sys.argv", base_args + ["--list-ids"]), redirect_stdout(io.StringIO()) as stream:
                main()
            self.assertIn("DR_249", stream.getvalue())
            self.assertFalse(output.exists())
            with patch("sys.argv", base_args + ["--output-dir", str(output), "--panel-height", "200"]), redirect_stdout(io.StringIO()):
                main()
            sample = output / "DR_249"
            with Image.open(sample / "ap.png") as overlay:
                self.assertEqual(overlay.size, (1000, 500))
                self.assertEqual(overlay.getpixel((100, 100)), GREEN)
                self.assertEqual(overlay.getpixel((200, 100)), RED)
            self.assertTrue((sample / "pair.png").is_file())
            coordinates = json.loads((sample / "coordinates.json").read_text())
            self.assertAlmostEqual(coordinates["views"]["ap"]["metrics"]["mean_error_px"], 100)
            self.assertAlmostEqual(coordinates["views"]["lat"]["metrics"]["mean_error_px"], 60)
            self.assertAlmostEqual(coordinates["views"]["ap"]["metrics"]["epsilon_dec_1024x512"], 51.2)
            with (output / "metrics.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)


if __name__ == "__main__":
    unittest.main()
