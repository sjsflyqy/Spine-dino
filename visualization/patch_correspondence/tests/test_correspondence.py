"""CPU synthetic checks; never load real checkpoints or run dataset experiments."""

import csv
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from methods.geotopo_dino.geometry.transforms import transform_points
from visualization.patch_correspondence.cli import build_parser, mine
from visualization.patch_correspondence.geometry import (
    box_iou, common_queries, grid_centres, make_view, sample_pair, sample_tokens, valid_tokens,
)
from visualization.patch_correspondence.matching import Summary, measurements, prepared_features, ranking
from visualization.patch_correspondence.rendering import encode_image, export_bundle, load_bundle


class GeometryTest(unittest.TestCase):
    def test_crop_flip_round_trip_and_complete_correspondence(self):
        source = Image.new("RGB", (112, 224), "gray")
        a = make_view(source, (0, 0, 112, 168), 224, flipped=True)
        b = make_view(source, (0, 56, 112, 224), 224, flipped=False)
        self.assertAlmostEqual(box_iou(a.metadata["box_xyxy"], b.metadata["box_xyxy"]), 0.5)
        queries = common_queries(a, b, (14, 16))
        self.assertGreater(len(queries["indices"]), 0)
        np.testing.assert_allclose(transform_points(a.transform, queries["source_xy"]), queries["xy_a"], atol=3e-5)
        np.testing.assert_allclose(transform_points(b.transform, queries["source_xy"]), queries["gt_b"], atol=3e-5)
        self.assertFalse(valid_tokens(a, 14)[0].any())
        self.assertTrue((queries["gt_b"] > 0).all())
        self.assertTrue((queries["gt_b"] < 224).all())

    def test_different_grids_sample_identical_physical_query(self):
        source = Image.new("RGB", (224, 224), "gray")
        a = make_view(source, (0, 0, 224, 224), 224)
        queries = common_queries(a, a, (14, 16))
        for patch_size in (14, 16):
            centres = grid_centres(224, patch_size)
            # An affine feature field has an independent analytic interpolation.
            features = torch.cat((centres, torch.ones(len(centres), 1)), dim=-1)
            sampled = sample_tokens(features, queries["xy_a"], 224)
            np.testing.assert_allclose(sampled[:, :2], queries["xy_a"], atol=3e-5)
            np.testing.assert_allclose(sampled[:, 2], 1, atol=1e-6)

    def test_seeded_distinct_overlapping_crops(self):
        source = Image.new("RGB", (512, 1024), "gray")
        first = sample_pair(source, seed=42, size=224)
        second = sample_pair(source, seed=42, size=224)
        self.assertEqual([v.metadata for v in first], [v.metadata for v in second])
        self.assertNotEqual(first[0].metadata["box_xyxy"], first[1].metadata["box_xyxy"])
        self.assertTrue(0.35 <= box_iou(*(v.metadata["box_xyxy"] for v in first)) <= 0.78)


def synthetic_matches():
    source = Image.new("RGB", (112, 112), "gray")
    a = make_view(source, (0, 0, 112, 112), 112)
    query = {"indices": torch.tensor([18]), "xy_a": torch.tensor([[35., 35.]]),
             "source_xy": torch.tensor([[35., 35.]]), "gt_b": torch.tensor([[35., 35.]])}
    eye = torch.eye(64)
    baseline_b = eye.reshape(8, 8, 64).roll(3, dims=0).reshape(64, 64)
    outputs, maps, masks = [], [], []
    for target in (baseline_b, eye):
        prepared = prepared_features((eye, target), query, a, a, 14, 1)
        values, heat = measurements(prepared[0], prepared[1], prepared[3], query["gt_b"], size=112,
                                    patch_size=14, near_radius=20, far_radius=35, peak_separation=28)
        outputs.append(values)
        maps.append(heat[0])
        masks.append(prepared[3].reshape(8, 8).numpy())
    return source, a, query, outputs, maps, masks


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_argmax_euclidean_error_and_actual_raw_cosine(self):
        _, _, _, (baseline, ours), maps, _ = synthetic_matches()
        self.assertAlmostEqual(float(baseline["error_px"][0]), 42)
        self.assertAlmostEqual(float(baseline["error_patch"][0]), 3)
        self.assertEqual(float(ours["error_px"][0]), 0)
        self.assertEqual(float(maps[0][5, 2]), 1)
        self.assertEqual(float(maps[1][2, 2]), 1)
        self.assertEqual(float(maps[0][2, 2]), 0)

    def test_all_query_pck_uses_shared_pixel_thresholds(self):
        summary = Summary([14, 28, 56])
        summary.add({"error_px": np.array([0., 42.])}, {"error_px": np.array([14., 28.])})
        report = summary.report()
        self.assertEqual(report["query_count"], 2)
        self.assertEqual(report["models"]["baseline"]["pck"]["14px"], 0.5)
        self.assertEqual(report["models"]["ours"]["pck"]["28px"], 1)

    def test_query_selection_does_not_discard_failures_from_statistics(self):
        _, _, queries, (baseline, ours), _, _ = synthetic_matches()
        ranks = ranking(baseline, ours, queries, (np.array([0.5]), np.array([0.05]), np.array([0.1])),
                        (112, 112), 1, size=112, reference_patch=14, baseline_min_patches=2,
                        min_gain_patches=2)
        self.assertTrue(ranks["eligible"][0])
        low_quality = ranking(baseline, ours, queries, (np.array([0.5]), np.array([0.]), np.array([0.])),
                              (112, 112), 1, size=112, reference_patch=14, baseline_min_patches=2,
                              min_gain_patches=2)
        self.assertFalse(low_quality["eligible"][0])

    def test_cache_argmax_audit_and_publication_exports(self):
        source, view, queries, values, maps, masks = synthetic_matches()
        query = {"a_xy": [35, 35], "source_xy": [35, 35], "gt_b_xy": [35, 35]}
        for name, measured in zip(("baseline", "ours"), values):
            query[f"{name}_pred_b_xy"] = [float(measured["pred_x"][0]), float(measured["pred_y"][0])]
            query[f"{name}_error_px"] = float(measured["error_px"][0])
        metadata = {"query": query, "views": {"a": view.metadata, "b": view.metadata},
                    "models": {"baseline": {"patch_size": 14}, "ours": {"patch_size": 14}}}
        arrays = {"metadata": np.asarray(json.dumps(metadata)), "source_png": encode_image(source),
                  "a_png": encode_image(view.image), "b_png": encode_image(view.image),
                  "baseline_heatmap": maps[0], "ours_heatmap": maps[1],
                  "baseline_valid_b": masks[0], "ours_valid_b": masks[1]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "case.npz"
            np.savez_compressed(bundle, **arrays)
            export_bundle(bundle, root / "figure")
            for suffix in ("png", "pdf", "svg"):
                self.assertTrue((root / f"figure/fig1.{suffix}").is_file())
                self.assertTrue((root / f"figure/panels/baseline.{suffix}").is_file())
            with Image.open(root / "figure/fig1.png") as image:
                self.assertAlmostEqual(image.info["dpi"][0], 300, delta=1)
            self.assertNotIn("<text", (root / "figure/panels/baseline.svg").read_text())
            loaded, data = load_bundle(bundle)
            np.testing.assert_array_equal(data["baseline_heatmap"], maps[0])
            metadata["query"]["baseline_pred_b_xy"] = [35, 35]
            arrays["metadata"] = np.asarray(json.dumps(metadata))
            np.savez_compressed(root / "tampered.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "argmax"):
                load_bundle(root / "tampered.npz")


class PipelineTest(unittest.TestCase):
    def test_mocked_models_keep_all_queries_without_fabricating_cases(self):
        class DummyBackbone:
            def forward_features(self, image):
                count = (image.shape[-1] // 14) ** 2
                return {"x_norm_patchtokens": torch.eye(count)[None]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = np.random.default_rng(7).integers(20, 220, (640, 320, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / "xray.png")
            (root / "weights.pth").write_bytes(b"synthetic fixture, not a real checkpoint")
            model = SimpleNamespace(model=DummyBackbone(), patch_size=14, embedding_dim=256, architecture="dummy")
            args = build_parser().parse_args(["mine", "--image", str(root / "xray.png"),
                "--baseline-weights", str(root / "weights.pth"), "--ours-weights", str(root / "weights.pth"),
                "--device", "cpu", "--view-size", "224", "--seeds", "42", "--output-dir", str(root / "output")])
            with patch("visualization.patch_correspondence.cli.load_dino_model", return_value=model), torch.inference_mode():
                mine(args)
            summary = json.loads((root / "output/summary.json").read_text())
            self.assertGreater(summary["all_valid_queries"]["query_count"], 0)
            self.assertEqual(summary["selected_case_count"], 0)
            with gzip.open(root / "output/all_queries.csv.gz", "rt") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), summary["all_valid_queries"]["query_count"])
            self.assertTrue(all(float(row["gain_px"]) == 0 for row in rows))
            self.assertEqual(json.loads((root / "output/selected_cases.json").read_text()), [])


if __name__ == "__main__":
    unittest.main()
