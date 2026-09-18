"""Structural invariants for the offline mask generator and training geometry."""

import random
import unittest

from PIL import Image
import torch

from methods.geotopo_dino.masking.spine_span import (
    SpanConfig, fuse_attention, generate_mask_comparison,
)


class SpineSpanTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.valid = torch.zeros(28, 28, dtype=torch.bool)
        self.valid[2:26, 4:24] = True
        y, x = torch.meshgrid(torch.arange(28), torch.arange(28), indexing="ij")
        self.expected_x = (14 + 3 * torch.sin(y[:, 0].float() / 7)).round().long()
        self.response = (0.02 + torch.exp(-((x - self.expected_x[:, None]).float() / 1.8)**2)) * self.valid
        indices = self.valid.flatten().nonzero().flatten()
        self.block = torch.zeros_like(self.valid)
        self.block.flatten()[indices[:int(len(indices) * 0.4)]] = True

    def test_curved_span_budget_contiguity_context_and_matched_control(self):
        result = generate_mask_comparison(self.response, self.valid, self.block, seed=19)
        self.assertFalse(result.diagnostics["used_fallback"], result.diagnostics)
        for mask in result.masks.values():
            self.assertEqual(int(mask.sum()), int(self.block.sum()))
            self.assertFalse((mask & ~self.valid).any())
        active = result.span.any(-1).nonzero().flatten()
        self.assertGreater(active.numel(), 0)
        self.assertEqual(active.numel(), int(active[-1] - active[0] + 1))
        self.assertTrue(torch.equal(result.span[active], result.band[active]))
        self.assertTrue(result.band[:active[0]].any())
        self.assertTrue(result.band[active[-1] + 1:].any())
        self.assertFalse((result.background & result.band).any())
        random_mask, span_mask = result.masks["band_random"], result.masks["topology_span"]
        self.assertTrue(torch.equal(random_mask & ~result.band, span_mask & ~result.band))
        self.assertEqual(int((random_mask & result.band).sum()), int((span_mask & result.band).sum()))
        self.assertEqual(int((span_mask & result.band).sum()), round(int(result.band.sum()) * 0.5))
        for mask in (random_mask, span_mask):
            self.assertFalse((mask & result.context).any())
        start, stop = int(active[0]), int(active[-1]) + 1
        self.assertTrue(torch.equal(result.context[start - 2:start], result.band[start - 2:start]))
        self.assertTrue(torch.equal(result.context[stop:stop + 2], result.band[stop:stop + 2]))
        self.assertFalse((result.span & result.supplement).any())
        self.assertTrue(torch.equal(result.span | result.supplement, span_mask))
        rows = (result.centerline >= 0).nonzero().flatten()
        self.assertLessEqual(int((result.centerline[rows] - self.expected_x[rows]).abs().max()), 1)

    def test_uniform_empty_nonfinite_maps_fall_back(self):
        for response in (self.valid.float(), torch.zeros_like(self.response),
                         torch.full_like(self.response, float("nan"))):
            result = generate_mask_comparison(response, self.valid, self.block)
            self.assertTrue(result.diagnostics["used_fallback"])
            self.assertTrue(torch.equal(result.masks["topology_span"], self.block))
            self.assertTrue(torch.isfinite(result.response).all())

    def test_border_artifact_is_rejected(self):
        response = torch.zeros_like(self.response)
        response[2:26, 4] = 1
        result = generate_mask_comparison(response, self.valid, self.block)
        self.assertIn("border_dominated_response", result.diagnostics["fallback_reasons"])
        self.assertTrue(torch.equal(result.masks["topology_span"], self.block))

    def test_budget_extremes_and_tiny_grid(self):
        for block in (torch.zeros_like(self.block), self.valid.clone()):
            result = generate_mask_comparison(self.response, self.valid, block)
            for mask in result.masks.values():
                self.assertEqual(int(mask.sum()), int(block.sum()))
            self.assertTrue(result.diagnostics["used_fallback"])
        for shape in ((1, 1), (1, 6), (6, 1)):
            valid = torch.ones(shape, dtype=torch.bool)
            result = generate_mask_comparison(valid.float(), valid, valid)
            self.assertTrue(result.diagnostics["used_fallback"])

    def test_no_valid_patches_and_padding_rejection(self):
        empty = torch.zeros_like(self.valid)
        result = generate_mask_comparison(self.response, empty, empty)
        self.assertIn("no_valid_patches", result.diagnostics["fallback_reasons"])
        invalid_block = self.block.clone()
        invalid_block[0, 0] = True
        with self.assertRaises(ValueError):
            generate_mask_comparison(self.response, self.valid, invalid_block)

    def test_reproducible_detached_and_no_mutation_or_global_rng_change(self):
        response = self.response.clone().requires_grad_(True)
        before = response.detach().clone()
        rng = torch.random.get_rng_state().clone()
        first = generate_mask_comparison(response, self.valid, self.block, seed=29)
        second = generate_mask_comparison(response, self.valid, self.block, seed=29)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(before, response))
        self.assertFalse(first.response.requires_grad)
        for name in first.masks:
            self.assertTrue(torch.equal(first.masks[name], second.masks[name]))

    def test_padding_mass_excluded_from_fusion_and_zero_heads_ignored(self):
        first = self.response.clone()
        second = first * 0.01
        second[~self.valid] = 100
        zero = torch.zeros_like(first)
        fused = fuse_attention(torch.stack((first, second, zero)), self.valid)
        torch.testing.assert_close(fused, first / first.sum())
        self.assertFalse(fused[~self.valid].any())

    def test_config_rejects_invalid_settings(self):
        for kwargs in ({"span_fraction": 0}, {"smooth_kernel": 2}, {"min_contrast": 1},
                       {"context_rows": 0}, {"max_border_mass": float("nan")},
                       {"band_mask_ratio": 0}, {"min_span_rows": 0}):
            with self.assertRaises(ValueError):
                SpanConfig(**kwargs)

    def test_multiple_segments_are_sampled_without_bridging_gap(self):
        valid = torch.ones(48, 28, dtype=torch.bool)
        y, x = torch.meshgrid(torch.arange(48), torch.arange(28), indexing="ij")
        supported = ((y >= 4) & (y < 18)) | ((y >= 28) & (y < 44))
        response = 0.001 + supported * torch.exp(-((x - 14).float() / 1.8)**2)
        block = torch.zeros_like(valid)
        block.flatten()[:int(valid.numel() * 0.4)] = True
        chosen = set()
        for seed in range(12):
            result = generate_mask_comparison(response, valid, block, seed=seed)
            self.assertFalse(result.diagnostics['used_fallback'], result.diagnostics)
            self.assertEqual(len(result.diagnostics['segments']), 2)
            self.assertFalse(result.band[21:25].any())
            chosen.add(tuple(result.diagnostics['selected_segment']))
            self.assertEqual(int((result.masks['topology_span'] & result.band).sum()),
                             round(int(result.band.sum()) * 0.5))
            self.assertTrue((result.supplement & result.band).any())
            active = result.span.any(-1).nonzero().flatten()
            self.assertEqual(active.numel(), int(active[-1] - active[0] + 1))
            for name in ('topology_span', 'band_random'):
                self.assertFalse((result.masks[name] & result.context).any())
                self.assertEqual(int(result.masks[name].sum()), int(block.sum()))
        self.assertEqual(len(chosen), 2)

    def test_context_takes_priority_over_requested_band_quota(self):
        result = generate_mask_comparison(self.response, self.valid, self.block,
                                          config=SpanConfig(band_mask_ratio=1), seed=19)
        self.assertFalse(result.diagnostics['used_fallback'])
        self.assertTrue(result.diagnostics['band_budget_adjusted'])
        self.assertLess(result.diagnostics['band_mask_ratio_actual'], 1)
        self.assertFalse((result.masks['topology_span'] & result.context).any())
        self.assertEqual(int(result.masks['topology_span'].sum()), int(self.block.sum()))


class OfflineAnchorTests(unittest.TestCase):
    def test_train_mode_matches_existing_anchor_and_preserves_rng(self):
        from methods.geotopo_dino.data.augmentations import DataAugmentationGeoTopoDINO
        from visualization.spine_masks.visualize import make_anchor, seeded_cpu_random
        source = Image.new("RGB", (91, 173), (110, 110, 110))
        augmentation = DataAugmentationGeoTopoDINO((0.5, 1), (0.2, 0.5), 8,
                                                  global_crops_size=518, patch_size=14)
        with seeded_cpu_random(12):
            expected, expected_geometry = augmentation._full_fov_anchor(source)
        python_rng, torch_rng = random.getstate(), torch.random.get_rng_state().clone()
        actual, geometry, image = make_anchor(source, size=518, mode="train", seed=12)
        self.assertEqual(python_rng, random.getstate())
        self.assertTrue(torch.equal(torch_rng, torch.random.get_rng_state()))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(geometry["transform"], expected_geometry["transform"])
        self.assertTrue(torch.equal(geometry["valid_mask"], expected_geometry["valid_mask"]))
        self.assertEqual(image.size, (518, 518))
        self.assertEqual(geometry["valid_mask"].shape, (37, 37))


if __name__ == "__main__":
    unittest.main()
