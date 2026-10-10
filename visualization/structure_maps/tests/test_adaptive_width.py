import random
import unittest

import torch

from methods.geotopo_dino.masking.adaptive_ribbon import adapt_ribbon_width
from methods.geotopo_dino.masking.candidate_mask_sampler import _components


class AdaptiveWidthTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.valid = torch.zeros(40, 37, dtype=torch.bool)
        self.valid[:, 2:35] = True
        self.path = torch.full((40,), 18, dtype=torch.long)
        self.options = dict(path_begin=0, span_start=2, span_stop=38)

    def ridge(self, narrow=3, wide=7):
        score = self.valid.float() * 0.1
        score[:20, 18 - narrow // 2:19 + narrow // 2] = 0.9
        score[20:, 18 - wide // 2:19 + wide // 2] = 0.9
        return score

    def test_local_width_tracks_known_change_in_ridge_scale(self):
        result = adapt_ribbon_width(self.ridge(), self.valid, self.path, **self.options)
        widths = result.widths
        self.assertTrue((widths[2:17] == 3).all())
        self.assertTrue((widths[23:38] == 7).all())
        self.assertTrue(((widths[1:] - widths[:-1]).abs() <= 2).all())
        self.assertEqual(len(_components(result.core)), 1)
        self.assertFalse((result.core & ~self.valid).any())
        self.assertFalse((result.core & result.context).any())
        self.assertEqual(int(result.core.sum()), int(widths[2:38].sum()))
        self.assertTrue((result.context[:2].sum(-1) == widths[:2]).all())
        self.assertTrue((result.context[38:].sum(-1) == widths[38:]).all())

    def test_image_width_changes_across_images_and_fixed_control_stays_five(self):
        for width in (3, 7):
            score = self.ridge(width, width)
            result = adapt_ribbon_width(score, self.valid, self.path, mode="image", **self.options)
            self.assertTrue((result.widths == width).all())
            fixed = adapt_ribbon_width(score, self.valid, self.path, mode="image", min_width=5, max_width=5, **self.options)
            self.assertTrue((fixed.core.sum(-1)[2:38] == 5).all())
            self.assertEqual(int(fixed.core.sum()), 180)

    def test_flat_and_weak_response_do_not_invent_width_evidence(self):
        flat = self.valid.float() * 0.4
        for mode in ("image", "local"):
            result = adapt_ribbon_width(flat, self.valid, self.path, mode=mode, **self.options)
            self.assertFalse(result.reliable.any())
            self.assertTrue((result.widths == 5).all())
        score = self.ridge(7, 7)
        score[12:16] = flat[12:16]
        local = adapt_ribbon_width(score, self.valid, self.path, **self.options)
        self.assertTrue((local.widths[12:16] == 5).all())
        self.assertFalse(local.reliable[12:16].any())

    def test_horizontal_transpose_preserves_masks_and_width_estimates(self):
        a = adapt_ribbon_width(self.ridge(), self.valid, self.path, **self.options)
        b = adapt_ribbon_width(self.ridge().T, self.valid.T, self.path, axis="horizontal", **self.options)
        torch.testing.assert_close(a.core.T, b.core)
        torch.testing.assert_close(a.context.T, b.context)
        torch.testing.assert_close(a.widths, b.widths)

    def test_padding_capacity_caps_width_without_reading_padding_scores(self):
        valid = torch.zeros_like(self.valid)
        valid[:, 16:21] = True
        score = torch.full(self.valid.shape, float("nan"))
        score[valid] = self.ridge(9, 9)[valid]
        result = adapt_ribbon_width(score, valid, self.path, **self.options)
        self.assertTrue((result.widths <= 5).all())
        self.assertFalse(((result.core | result.context) & ~valid).any())
        self.assertTrue(torch.isfinite(result.prominence).all())

    def test_curved_centerline_preserved_and_inputs_rng_untouched(self):
        path = self.path.clone()
        path[10:20] += 1
        score = self.ridge().requires_grad_()
        before = score.detach().clone()
        py_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
        result = adapt_ribbon_width(score, self.valid, path, **self.options)
        for row in range(2, 38):
            occupied = result.core[row].nonzero().flatten()
            self.assertEqual(int(occupied[0] + occupied[-1]), 2 * int(path[row]))
        self.assertFalse(result.core.requires_grad)
        self.assertFalse(result.prominence.requires_grad)
        self.assertEqual(random.getstate(), py_state)
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
        torch.testing.assert_close(score.detach(), before)
        torch.testing.assert_close(path[10:20], self.path[10:20] + 1)

    def test_invalid_span_range_scores_and_infeasible_path_rejected(self):
        for extra in ({"min_width": 4}, {"max_width": 3}, {"relative_threshold": 1}, {"min_prominence": 0}):
            with self.assertRaises(ValueError):
                adapt_ribbon_width(self.ridge(), self.valid, self.path, **self.options, **extra)
        with self.assertRaises(ValueError):
            adapt_ribbon_width(self.ridge(), self.valid, self.path, path_begin=0, span_start=0, span_stop=38)
        with self.assertRaises(ValueError):
            adapt_ribbon_width(self.ridge(), self.valid, torch.zeros_like(self.path), **self.options)
        invalid = self.ridge()
        invalid[5, 18] = float("nan")
        with self.assertRaises(ValueError):
            adapt_ribbon_width(invalid, self.valid, self.path, **self.options)


if __name__ == "__main__":
    unittest.main()
