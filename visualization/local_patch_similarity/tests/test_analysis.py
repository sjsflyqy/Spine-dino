"""Check correspondence geometry, negative controls, and actual backbone tokens."""

from pathlib import Path
import random
import sys
import unittest

import numpy as np
from PIL import Image
import torch

from methods.geotopo_dino.geometry.transforms import horizontal_flip_transform
from visualization.dino_spine_maps.model_loader import LoadedDino
from visualization.local_patch_similarity.analysis import analyze, extract_features, metric_differences, pooled_local_similarity
from visualization.local_patch_similarity.views import Views, make_views, select_queries


def synthetic_views():
    return Views(Image.new("RGB", (28, 28)), torch.zeros(3, 28, 28), torch.zeros(8, 3, 28, 28),
                 {"transform": torch.eye(3), "valid_mask": torch.ones(2, 2, dtype=torch.bool)},
                 [{"transform": torch.eye(3)} for _ in range(8)])


class CorrespondenceTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_identical_crops_and_signed_cosines(self):
        views = synthetic_views()
        tokens = torch.eye(4).repeat(8, 1, 1)
        metrics, arrays = analyze(tokens, tokens[:1], views)
        self.assertAlmostEqual(metrics["local_local_aligned_cosine"], 1)
        self.assertAlmostEqual(metrics["paired_far_negative_cosine"], 0)
        self.assertAlmostEqual(metrics["paired_discrimination_gap"], 1)
        self.assertAlmostEqual(metrics["cross_local_mean_cosine"], 0.25)
        self.assertEqual(metrics["within_local_mean_cosine_excluding_self"], 0)
        self.assertAlmostEqual(metrics["anchor_retrieval_mean_error_patches"], 0, places=6)
        self.assertEqual(metrics["paired_query_count"], 8 * 7 * 4)
        np.testing.assert_allclose(arrays["patch_similarity"], arrays["patch_similarity"].T)
        changed = tokens.clone()
        changed[1] *= -1
        _, signed = analyze(changed, tokens[:1], views)
        self.assertEqual(signed["patch_similarity"][0, 4], -1)
        scaled, _ = analyze(tokens * 3, tokens[:1] * 7, views)
        for value in metric_differences(metrics, scaled).values():
            self.assertAlmostEqual(value, 0, places=6)

    def test_flip_matches_source_positions_instead_of_patch_indices(self):
        views = synthetic_views()
        views.local_geometry[1]["transform"] = horizontal_flip_transform(28)
        tokens = torch.eye(4).repeat(8, 1, 1)
        tokens[1] = tokens[1].reshape(2, 2, 4).flip(1).reshape(4, 4)
        metrics, arrays = analyze(tokens, torch.eye(4)[None], views)
        self.assertAlmostEqual(metrics["local_local_aligned_cosine"], 1)
        self.assertEqual(arrays["patch_similarity"][0, 4], 0)
        self.assertEqual(arrays["patch_similarity"][0, 5], 1)
        np.testing.assert_array_equal(arrays["anchor_retrieval_index"][1], [1, 0, 3, 2])
        self.assertAlmostEqual(metrics["anchor_retrieval_mean_error_patches"], 0, places=6)

    def test_disjoint_views_have_no_fabricated_positive_matches(self):
        views = synthetic_views()
        views.source = Image.new("RGB", (224, 28))
        for index, geometry in enumerate(views.local_geometry):
            geometry["transform"][0, 2] = -28 * index
        tokens = torch.eye(4).repeat(8, 1, 1)
        metrics, _ = analyze(tokens, tokens[:1], views)
        self.assertEqual(metrics["local_local_aligned_query_count"], 0)
        self.assertIsNone(metrics["paired_discrimination_gap"])
        self.assertIsNone(metrics["local_local_aligned_cosine"])

    def test_resized_offset_crop_samples_features_bilinearly_before_normalization(self):
        views = synthetic_views()
        # Crop source [7,21]x[7,21], resize to 28x28. Patch centres map to
        # source 10.5 and 17.5, between the target centres 7 and 21.
        views.local_geometry[1]["transform"] = torch.tensor([[2., 0., -14.], [0., 2., -14.], [0., 0., 1.]])
        tokens = torch.eye(4).repeat(8, 1, 1)
        tokens[1] = torch.tensor([[9., 3., 3., 1.], [3., 9., 1., 3.],
                                  [3., 1., 9., 3.], [1., 3., 3., 9.]]) / 16
        _, arrays = analyze(tokens, torch.eye(4)[None], views)
        np.testing.assert_allclose(arrays["local_local_aligned_cosine"][1, 0], 1, atol=1e-6)
        np.testing.assert_allclose(arrays["local_anchor_aligned_cosine"][1], 1, atol=1e-6)

    def test_padding_is_excluded_from_alignment_and_retrieval(self):
        views = synthetic_views()
        views.anchor_geometry["valid_mask"][0, 0] = False
        tokens = torch.eye(4).repeat(8, 1, 1)
        metrics, arrays = analyze(tokens, tokens[:1], views)
        self.assertEqual(metrics["local_anchor_valid_query_count"], 8 * 3)
        self.assertFalse((arrays["anchor_retrieval_index"] == 0).any())
        self.assertTrue(np.isnan(arrays["local_anchor_aligned_cosine"][:, 0]).all())


class PooledSimilarityTest(unittest.TestCase):
    def test_whole_local_cosine_pools_before_normalizing(self):
        tokens = torch.tensor([[[10., 0.], [0., 1.]], [[1., 0.], [1., 0.]],
                               [[0., 2.], [0., 4.]], [[-1., 0.], [-1., 0.]]]).repeat(2, 1, 1)
        matrix = pooled_local_similarity(tokens)
        torch.testing.assert_close(matrix.diag(), torch.ones(8), atol=0, rtol=0)
        torch.testing.assert_close(matrix, matrix.T)
        self.assertAlmostEqual(float(matrix[0, 1]), 10 / 101 ** 0.5, places=6)
        self.assertAlmostEqual(float(matrix[0, 2]), 1 / 101 ** 0.5, places=6)
        self.assertEqual(float(matrix[1, 3]), -1)
        self.assertEqual(float(matrix[1, 2]), 0)
        torch.testing.assert_close(matrix, pooled_local_similarity(tokens.flip(1)))
        torch.testing.assert_close(matrix, pooled_local_similarity(tokens * 3))
        torch.testing.assert_close(matrix - matrix, torch.zeros(8, 8))

    def test_undefined_zero_mean_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonzero"):
            pooled_local_similarity(torch.tensor([[[1., 0.], [-1., 0.]]]).repeat(8, 1, 1))


class InputAndInferenceTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_fixed_crops_rng_preservation_and_queries(self):
        pixels = np.arange(140 * 280 * 3, dtype=np.uint32).reshape(280, 140, 3).astype(np.uint8)
        source = Image.fromarray(pixels)
        random.seed(11)
        torch.manual_seed(11)
        python_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
        first = make_views(source, seed=42, anchor_size=56, local_size=28)
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        second = make_views(source, seed=42, anchor_size=56, local_size=28)
        train = make_views(source, seed=42, mode="train", anchor_size=56, local_size=28)
        self.assertTrue(torch.equal(first.locals, second.locals))
        for a, b in zip(first.local_geometry, train.local_geometry):
            self.assertTrue(torch.equal(a["box_xyxy"], b["box_xyxy"]))
        point = first.source_centres[0, 0].tolist()
        query = select_queries(first, source_queries=[point])[0]
        self.assertEqual((query["local"], query["row"], query["column"]), (0, 0, 0))
        with self.assertRaises(ValueError):
            select_queries(first, queries=[(8, 0, 0)])
        with self.assertRaises(ValueError):
            select_queries(first, source_queries=[(-1, 0)])

    def test_native_forward_features_and_batching(self):
        upstream = str(Path(__file__).resolve().parents[3] / "upstream/dinov2-main")
        sys.path.insert(0, upstream)
        from dinov2.models.vision_transformer import DinoVisionTransformer
        model = DinoVisionTransformer(img_size=28, patch_size=14, embed_dim=32, depth=2,
                                     num_heads=4, block_chunks=0).eval()
        loaded = LoadedDino(model, 2, Path("test.pth"), 14, 32, 2, 4, 0)
        views = synthetic_views()
        views.locals = torch.randn_like(views.locals)
        with torch.inference_mode():
            local, anchor = extract_features(loaded, views, torch.device("cpu"), batch_size=3)
            expected = model.forward_features(views.locals)["x_norm_patchtokens"]
        torch.testing.assert_close(local, expected, atol=2e-5, rtol=2e-5)
        self.assertEqual(anchor.shape, (1, 4, 32))


if __name__ == "__main__":
    unittest.main()
