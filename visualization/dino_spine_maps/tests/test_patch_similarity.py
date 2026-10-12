"""Check query geometry and cosine extraction against native model outputs."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from visualization.dino_spine_maps.map_extractor import _cosine_grid, extract_layer_maps
from visualization.dino_spine_maps.model_loader import LoadedDino
from visualization.dino_spine_maps.patch_similarity import select_query_patch
from visualization.dino_spine_maps.preprocessing import prepare_image


class QueryGeometryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image = Path(self.directory.name) / "test.png"
        Image.new("RGB", (101, 203)).save(self.image)
        self.prepared = prepare_image(self.image, patch_size=16, long_side=96)

    def test_rounded_resize_and_padding(self):
        # 101 * 96/203 rounds to 48. The true x scale is 48/101.
        query = select_query_patch(self.prepared, xy=(67.5, 80))
        self.assertEqual((query.row, query.column, query.index), (2, 2, 8))
        left, top, right, bottom = query.original_box_xyxy
        self.assertTrue(left <= 67.5 < right and top <= 80 < bottom)
        relative = select_query_patch(self.prepared, relative_xy=(67.5 / 101, 80 / 203))
        self.assertEqual(query, relative)

    def test_invalid_queries(self):
        for xy in ((-1, 0), (101, 0), (0, 203), (float("nan"), 0)):
            with self.subTest(xy=xy), self.assertRaises(ValueError):
                select_query_patch(self.prepared, xy=xy)
        with self.assertRaises(ValueError):
            select_query_patch(self.prepared, relative_xy=(1, 0.5))
        with self.assertRaises(ValueError):
            select_query_patch(self.prepared, patch_rc=(0, 3))
        Image.new("RGB", (1, 203)).save(self.image)
        padded = prepare_image(self.image, patch_size=16, long_side=96)
        with self.assertRaisesRegex(ValueError, "padding"):
            select_query_patch(padded, xy=(0, 80))


class PatchCosineTest(unittest.TestCase):
    def test_signed_cosine_is_not_minmax_normalized(self):
        patches = torch.tensor([[[2., 0.], [0., 3.], [-4., 0.], [2., 2.]]])
        values = _cosine_grid(patches[:, 0], patches, 2, 2)
        np.testing.assert_allclose(values, [[1, 0], [-1, 2 ** -0.5]], atol=1e-6)

    def test_native_dinov2_and_dinov3_with_extra_tokens(self):
        root = Path(__file__).resolve().parents[3]
        os.environ["XFORMERS_DISABLED"] = "1"
        for generation in (2, 3):
            sys.path.insert(0, str(root / "upstream" / f"dinov{generation}-main"))
        from dinov2.models.vision_transformer import DinoVisionTransformer as Dino2
        from dinov3.models.vision_transformer import DinoVisionTransformer as Dino3

        torch.manual_seed(7)
        torch.set_num_threads(2)
        models = [
            Dino2(img_size=32, patch_size=16, embed_dim=32, depth=2, num_heads=4, block_chunks=0, num_register_tokens=2),
            Dino3(img_size=32, patch_size=16, embed_dim=32, depth=2, num_heads=4, n_storage_tokens=2, untie_cls_and_patch_norms=True, pos_embed_rope_dtype="fp32"),
        ]
        models[1].init_weights()
        for generation, model in zip((2, 3), models):
            with self.subTest(generation=generation), torch.inference_mode():
                model.eval()
                loaded = LoadedDino(model, generation, Path("test.pth"), 16, 32, 2, 4, 2)
                batch = torch.randn(1, 3, 32, 48)
                result = extract_layer_maps(loaded, batch, query_patch=(1, 1))
                standard = model.forward_features(batch)
                patches = standard["x_norm_patchtokens"].float()
                expected = F.cosine_similarity(patches[:, 4:5], patches, dim=-1)[0].reshape(2, 3).numpy()
                np.testing.assert_allclose(result.layers[-1].patch_cosine, expected, atol=2e-5)
                np.testing.assert_allclose(result.final_patch_tokens, patches, atol=2e-5)
                for layer in result.layers:
                    self.assertAlmostEqual(float(layer.patch_cosine[1, 1]), 1.0, places=5)
                    self.assertEqual(layer.patch_cosine.shape, (2, 3))
                legacy = extract_layer_maps(loaded, batch, layers=[1])
                self.assertIsNone(legacy.layers[0].patch_cosine)
                np.testing.assert_allclose(legacy.layers[0].cosine, result.layers[-1].cosine)


if __name__ == "__main__":
    unittest.main()
