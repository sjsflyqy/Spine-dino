import random
import unittest

import torch
import torch.nn.functional as F

from methods.geotopo_dino.masking.slender_mask_sampler import compare_slender_masks


class SlenderCandidateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.valid = torch.zeros(40, 37, dtype=torch.bool)
        self.valid[2:38, 3:34] = True
        y, x = torch.meshgrid(torch.arange(40), torch.arange(37), indexing="ij")
        self.center = (18 + 3 * torch.sin(y[:, 0].float() / 6)).round().long()
        self.score = torch.exp(-((x - self.center[:, None]).float() / 1.4)**2) * self.valid
        self.baseline = torch.zeros_like(self.valid)
        self.baseline.flatten()[self.valid.flatten().nonzero().flatten()[:int(self.valid.sum() * 0.4)]] = True

    def test_thin_width_budget_split_connectedness_and_visible_ends(self):
        result = compare_slender_masks(self.score, self.valid, self.baseline, seed=17)
        torch.testing.assert_close(result.masks[0], self.baseline)
        for index in result.diagnostics["eligible_candidate_indices"]:
            core, extra, context = result.core_masks[index], result.supplement_masks[index], result.context_masks[index]
            torch.testing.assert_close(result.masks[index], core | extra)
            self.assertEqual(int(result.masks[index].sum()), int(self.baseline.sum()))
            self.assertFalse((result.masks[index] & ~self.valid).any())
            self.assertFalse((core & extra).any())
            self.assertFalse((result.masks[index] & context).any())
            self.assertFalse((F.max_pool2d(core.float()[None, None], 3, 1, 1)[0, 0].bool() & extra).any())
            self.assertTrue(torch.equal(core.sum(-1)[core.any(-1)], torch.full((result.diagnostics['core_rows'],), 5)))
            self.assertEqual(int(core.sum()), result.diagnostics["core_target_count"])
            self.assertEqual(result.diagnostics["candidate_geometry"][index]["core_components"], 1)
            self.assertEqual(int(context.sum()), 20)

    def test_guided_curve_tracks_synthetic_ridge_and_selection_uses_only_core(self):
        result = compare_slender_masks(self.score, self.valid, self.baseline, seed=2)
        info = result.diagnostics["candidate_geometry"][3]
        start, stop = info["span_start_stop"]
        begin = result.diagnostics["supported_segment"][0]
        path = torch.tensor(info["path"])
        self.assertLessEqual(int((path[start - begin:stop - begin] - self.center[start:stop]).abs().max()), 2)
        for index in result.diagnostics["eligible_candidate_indices"]:
            self.assertAlmostEqual(float(result.scores[index]), float(self.score[result.core_masks[index]].mean()), places=6)
        self.assertEqual(float(result.probabilities[0]), 0)
        self.assertEqual(result.best_index, max(result.diagnostics["eligible_candidate_indices"], key=lambda i: float(result.scores[i])))

    def test_transposed_fov_selects_horizontal_path_with_same_width(self):
        result = compare_slender_masks(self.score.T, self.valid.T, self.baseline.T, seed=5)
        self.assertEqual(result.diagnostics["path_axis"], "horizontal")
        for index in result.diagnostics["eligible_candidate_indices"]:
            core = result.core_masks[index]
            self.assertTrue((core.sum(0)[core.any(0)] == 5).all())

    def test_default_selects_maximum_and_historical_softmax_is_explicit(self):
        for seed in (2, 17, 42):
            result = compare_slender_masks(self.score, self.valid, self.baseline, seed=seed)
            self.assertFalse(result.diagnostics["used_fallback"])
            self.assertEqual(result.adaptive_index, result.best_index)
            self.assertEqual(float(result.probabilities[result.best_index]), 1)
            legacy = compare_slender_masks(self.score, self.valid, self.baseline, seed=seed, selection_mode="softmax")
            self.assertEqual(legacy.diagnostics["selection_mode"], "softmax")
            self.assertGreater(int((legacy.probabilities > 0).sum()), 1)

    def test_core_only_matched_budget_requires_no_supplement(self):
        first = compare_slender_masks(self.score, self.valid, self.baseline, seed=9)
        budget = first.diagnostics["core_target_count"]
        control = torch.zeros_like(self.valid)
        control.flatten()[self.valid.flatten().nonzero().flatten()[:budget]] = True
        result = compare_slender_masks(self.score, self.valid, control, seed=9)
        for index in result.diagnostics["eligible_candidate_indices"]:
            torch.testing.assert_close(result.masks[index], result.core_masks[index])
            self.assertFalse(result.supplement_masks[index].any())
            self.assertEqual(int(result.masks[index].sum()), budget)

    def test_random_controls_independent_of_score_and_no_rng_or_input_mutation(self):
        state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
        before = self.score.clone()
        a = compare_slender_masks(self.score.requires_grad_(), self.valid, self.baseline, seed=71)
        b = compare_slender_masks(1 - self.score, self.valid, self.baseline, seed=71)
        for index in (1, 2, 4, 5):
            torch.testing.assert_close(a.core_masks[index], b.core_masks[index])
            torch.testing.assert_close(a.supplement_masks[index], b.supplement_masks[index])
        self.assertEqual(state, random.getstate())
        torch.testing.assert_close(torch_state, torch.random.get_rng_state())
        torch.testing.assert_close(before, self.score.detach())
        self.assertFalse(a.scores.requires_grad)

    def test_zero_full_budget_and_unreliable_response_fall_back(self):
        for baseline, unreliable in ((torch.zeros_like(self.valid), False), (self.valid, False), (self.baseline, True)):
            result = compare_slender_masks(self.score, self.valid, baseline, near_constant=unreliable)
            self.assertTrue(result.diagnostics["used_fallback"])
            self.assertEqual(result.adaptive_index, 0)
            torch.testing.assert_close(result.masks[result.adaptive_index], baseline)
            for mask in result.masks:
                self.assertEqual(int(mask.sum()), int(baseline.sum()))

    def test_short_or_too_narrow_fov_cannot_fake_slender_core(self):
        for shape in ((3, 10), (20, 2)):
            valid = torch.ones(shape, dtype=torch.bool)
            result = compare_slender_masks(valid.float(), valid, valid.clone(), strip_width=5, axis="vertical")
            self.assertTrue(result.diagnostics["used_fallback"])
            self.assertFalse(result.core_masks.any())


if __name__ == "__main__":
    unittest.main()
