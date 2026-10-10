import random
import unittest

import torch

from methods.geotopo_dino.masking.candidate_mask_sampler import (
    compare_candidate_masks, sample_candidate_masks,
)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.valid = torch.zeros(20, 24, dtype=torch.bool)
        self.valid[2:18, 3:21] = True
        self.baseline = torch.zeros_like(self.valid)
        self.baseline[3:10, 4:12] = True
        yy, xx = torch.meshgrid(torch.arange(20), torch.arange(24), indexing="ij")
        self.score = (torch.exp(-((xx - 13).float() / 4)**2) * self.valid).float()

    def test_exact_budget_padding_connectedness_and_original_baseline(self):
        result = compare_candidate_masks(self.score, self.valid, self.baseline, seed=17)
        torch.testing.assert_close(result.masks[0], self.baseline)
        self.assertEqual(result.masks.shape, (8, 20, 24))
        for index, mask in enumerate(result.masks):
            self.assertEqual(int(mask.sum()), int(self.baseline.sum()))
            self.assertFalse((mask & ~self.valid).any())
            if index:
                self.assertEqual(result.diagnostics["candidate_geometry"][index]["component_count"], 1)
        self.assertEqual(set(result.kinds), {"baseline_block", "compact", "span", "multi_region"})
        self.assertGreater(result.diagnostics["unique_candidate_count"], 1)

    def test_geometry_does_not_read_score_and_selection_is_correct(self):
        a = compare_candidate_masks(self.score, self.valid, self.baseline, seed=13, temperature=0.1, selection_mode="softmax")
        b = compare_candidate_masks(1 - self.score, self.valid, self.baseline, seed=13, temperature=0.1, selection_mode="softmax")
        torch.testing.assert_close(a.masks, b.masks)
        for index, mask in enumerate(a.masks):
            self.assertAlmostEqual(float(a.scores[index]), float(self.score[mask].mean()), places=6)
        self.assertEqual(a.best_index, int(a.scores.argmax()))
        torch.testing.assert_close(a.probabilities, torch.softmax(a.scores / 0.1, dim=0))
        self.assertGreaterEqual(a.diagnostics["adaptive_expected_score"], a.diagnostics["uniform_expected_score"] - 1e-6)

    def test_default_uses_highest_score_not_probability_draw(self):
        for seed in (1, 13, 42):
            result = compare_candidate_masks(self.score, self.valid, self.baseline, seed=seed)
            self.assertFalse(result.diagnostics["used_fallback"])
            self.assertEqual(result.adaptive_index, result.best_index)
            self.assertEqual(float(result.probabilities[result.best_index]), 1)
            self.assertEqual(float(result.probabilities.sum()), 1)

    def test_reproducible_no_global_rng_change_no_grad_no_mutation(self):
        python_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
        baseline_before, score_before = self.baseline.clone(), self.score.clone()
        a = compare_candidate_masks(self.score.requires_grad_(), self.valid, self.baseline, seed=123)
        b = compare_candidate_masks(self.score, self.valid, self.baseline, seed=123)
        torch.testing.assert_close(a.masks, b.masks)
        self.assertEqual((a.random_index, a.adaptive_index), (b.random_index, b.adaptive_index))
        self.assertEqual(random.getstate(), python_state)
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
        torch.testing.assert_close(self.baseline, baseline_before)
        torch.testing.assert_close(self.score.detach(), score_before)
        self.assertFalse(a.scores.requires_grad)

    def test_flat_unreliable_zero_and_full_budget_fall_back(self):
        for score, baseline, unreliable in (
            (torch.ones_like(self.score), self.baseline, False),
            (self.score, self.baseline, True),
            (self.score, torch.zeros_like(self.baseline), False),
            (self.score, self.valid.clone(), False),
        ):
            result = compare_candidate_masks(score, self.valid, baseline, near_constant=unreliable)
            self.assertTrue(result.diagnostics["used_fallback"])
            self.assertEqual(result.adaptive_index, 0)
            self.assertEqual(float(result.probabilities[0]), 1)
            torch.testing.assert_close(result.masks[result.adaptive_index], baseline)
            for mask in result.masks:
                self.assertEqual(int(mask.sum()), int(baseline.sum()))

    def test_fragmented_validity_reports_generation_fallback(self):
        valid = torch.zeros(5, 7, dtype=torch.bool)
        valid[1, :3] = True
        valid[3, 4:7] = True
        baseline = valid.clone()
        baseline[1, 0] = False
        baseline[3, 6] = False
        masks, _, info = sample_candidate_masks(valid, baseline, seed=0)
        for mask, diagnostics in zip(masks[1:], info[1:]):
            torch.testing.assert_close(mask, baseline)
            self.assertEqual(diagnostics["generation_fallback"], "fragmented_validity")

    def test_invalid_input_rejected_and_padding_scores_ignored(self):
        invalid = self.baseline.clone()
        invalid[0, 0] = True
        with self.assertRaises(ValueError):
            compare_candidate_masks(self.score, self.valid, invalid)
        with self.assertRaises(ValueError):
            compare_candidate_masks(self.score, self.valid, self.baseline, temperature=0)
        score = self.score.clone()
        score[~self.valid] = float("nan")
        result = compare_candidate_masks(score, self.valid, self.baseline)
        self.assertTrue(torch.isfinite(result.scores).all())


if __name__ == "__main__":
    unittest.main()
