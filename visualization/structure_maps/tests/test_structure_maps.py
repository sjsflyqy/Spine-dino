import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

from methods.geotopo_dino.masking.feature_structure_score import compute_structure_score
from visualization.structure_maps.visualize import (
    _aligned_stability, extract_patch_features, pixel_gradient,
)


class StructureMapTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_known_split_has_only_boundary_response(self):
        tokens = torch.zeros(5, 6, 2)
        tokens[:, :3, 0] = 1
        tokens[:, 3:, 1] = 1
        result = compute_structure_score(tokens, torch.ones(5, 6, dtype=torch.bool), smooth_kernel=1)
        self.assertAlmostEqual(float(result.difference[2, 2]), 3 / 8)
        self.assertEqual(float(result.difference[2, 1]), 0)
        self.assertAlmostEqual(float(result.difference[0, 2]), 2 / 5)
        self.assertGreater(float(result.score[:, 2:4].mean()), float(result.score[:, :2].mean()))

    def test_padding_is_excluded_from_differences_quantiles_and_smoothing(self):
        tokens = torch.randn(4, 5, 7)
        valid = torch.ones(4, 5, dtype=torch.bool)
        expected = compute_structure_score(tokens, valid)
        padded = torch.full((8, 11, 7), float("nan"))
        padded[2:6, 3:8] = tokens
        padded_valid = torch.zeros(8, 11, dtype=torch.bool)
        padded_valid[2:6, 3:8] = True
        actual = compute_structure_score(padded, padded_valid)
        for name in ("difference", "normalized", "score", "neighbor_count"):
            torch.testing.assert_close(getattr(actual, name)[2:6, 3:8], getattr(expected, name))
            self.assertTrue(torch.isfinite(getattr(actual, name)).all())
            self.assertFalse(getattr(actual, name)[~padded_valid].any())

    def test_uniform_empty_and_isolated_do_not_invent_structure(self):
        for valid in (torch.ones(4, 4, dtype=torch.bool), torch.zeros(4, 4, dtype=torch.bool),
                      torch.eye(1, 4, dtype=torch.bool)):
            tokens = torch.ones(*valid.shape, 3)
            result = compute_structure_score(tokens, valid)
            self.assertTrue(result.metrics["near_constant"])
            self.assertFalse(result.score.any())
            self.assertTrue(torch.isfinite(result.difference).all())

    def test_flip_scale_detach_and_input_preservation(self):
        tokens = torch.randn(7, 6, 8, requires_grad=True)
        before = tokens.detach().clone()
        valid = torch.ones(7, 6, dtype=torch.bool)
        result = compute_structure_score(tokens, valid, radius=2)
        flipped = compute_structure_score(tokens.flip(1) * 3, valid.flip(1), radius=2)
        torch.testing.assert_close(result.difference, flipped.difference.flip(1), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(result.score, flipped.score.flip(1), atol=1e-6, rtol=1e-5)
        self.assertFalse(result.score.requires_grad)
        torch.testing.assert_close(tokens.detach(), before)

    def test_invalid_valid_features_rejected(self):
        for value in (0.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                compute_structure_score(torch.full((2, 2, 3), value), torch.ones(2, 2, dtype=torch.bool))

    def test_pixel_gradient_does_not_measure_padding_border(self):
        tensor = torch.zeros(3, 56, 56)
        tensor[:, 14:42, 14:42] = 7
        valid = torch.zeros(4, 4, dtype=torch.bool)
        valid[1:3, 1:3] = True
        raw, normalized = pixel_gradient(tensor, valid)
        self.assertFalse(raw.any())
        self.assertFalse(normalized.any())

    def test_stability_undoes_geometry_flip(self):
        tokens = torch.randn(5, 6, 3)
        valid = torch.ones(5, 6, dtype=torch.bool)
        clean = {"valid": valid, "scores": {12: compute_structure_score(tokens, valid)}}
        train = {"valid": valid.flip(1), "geometry": {"flipped": True},
                 "scores": {12: compute_structure_score(tokens.flip(1), valid.flip(1))}}
        stats = _aligned_stability(clean, train, [12])["12"]
        self.assertAlmostEqual(stats["raw_difference_pearson"], 1, places=6)
        self.assertTrue(stats["flip_undone"])

    def test_native_intermediate_tokens_match_normal_forward(self):
        os.environ["XFORMERS_DISABLED"] = "1"
        upstream = Path(__file__).resolve().parents[3] / "upstream" / "dinov2-main"
        sys.path.insert(0, str(upstream))
        from dinov2.models.vision_transformer import DinoVisionTransformer
        model = DinoVisionTransformer(img_size=56, patch_size=14, embed_dim=24,
                                      depth=2, num_heads=3, block_chunks=0).eval()
        loaded = SimpleNamespace(generation=2, patch_size=14, num_layers=2, model=model)
        image = torch.randn(1, 3, 56, 56)
        actual = extract_patch_features(loaded, image, [0, 1])[2]
        with torch.inference_mode():
            expected = model.forward_features(image)["x_norm_patchtokens"][0]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (16, 24))


if __name__ == "__main__":
    unittest.main()
