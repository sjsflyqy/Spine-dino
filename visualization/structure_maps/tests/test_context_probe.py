import unittest

import numpy as np
import torch
from torch import nn

from visualization.structure_maps.context_metrics import (
    context_score, context_interventions, patch_errors, prototype_errors, visible_baselines, spearman,
)
from visualization.structure_maps.reconstruction_model import merge_named_state


class ContextProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_similarity_uses_only_visible_valid_neighbors(self):
        valid = torch.ones(3, 3, dtype=torch.bool)
        valid[0, 0] = False
        target = torch.zeros_like(valid)
        target[1, 1] = True
        tokens = torch.tensor([[1., 0.]]).expand(9, 2).clone()
        tokens[valid.flatten()] = torch.tensor([0., 1.])
        tokens[4] = torch.tensor([1., 0.])
        tokens[5] = torch.tensor([1., 0.])
        input_mask = target.clone()
        input_mask[1, 2] = True
        pmap, stats = context_score(tokens, target, input_mask, valid, radius=1)
        self.assertAlmostEqual(stats["P"], 0)
        self.assertEqual(stats["missing_fraction"], 0)
        self.assertEqual(stats["H"], 1)
        self.assertTrue(torch.isnan(pmap[~target]).all())
        _, visible = context_score(tokens, target, target, valid, radius=1)
        self.assertAlmostEqual(visible["P"], 1)

    def test_missing_context_is_reported_separately(self):
        valid = torch.ones(3, 3, dtype=torch.bool)
        target = valid.clone()
        pmap, stats = context_score(torch.ones(9, 2), target, target, valid)
        self.assertTrue(torch.isnan(pmap).all())
        self.assertEqual(stats["P"], 0)
        self.assertIsNone(stats["P_supported"])
        self.assertEqual(stats["missing_fraction"], 1)
        self.assertEqual(stats["H"], 1)

    def test_interventions_preserve_targets_counts_and_far_distance(self):
        valid = torch.ones(23, 23, dtype=torch.bool)
        valid[:, :2] = False
        target = torch.zeros_like(valid)
        target[8:15, 10:13] = True
        state = torch.random.get_rng_state().clone()
        near, far, metadata = context_interventions(target, valid, 2, 6, 3, 17)
        torch.testing.assert_close(state, torch.random.get_rng_state())
        self.assertFalse(metadata["near_capped"])
        for mask in [near] + far:
            self.assertFalse((target & ~mask).any())
            self.assertFalse((mask & ~valid).any())
            self.assertEqual(int(mask.sum()), int(near.sum()))
        for mask in far:
            extra = mask & ~target
            distance = (extra.nonzero()[:, None] - target.nonzero()[None]).abs().amax(-1).amin(-1)
            self.assertTrue((distance >= 6).all())
            self.assertFalse((extra & near).any())
        other, _, _ = context_interventions(target, valid, 2, 6, 3, 17)
        torch.testing.assert_close(near, other)

    def test_insufficient_far_capacity_caps_both_arms(self):
        valid = torch.ones(13, 13, dtype=torch.bool)
        target = torch.zeros_like(valid)
        target[3:10, 3:10] = True
        near, far, metadata = context_interventions(target, valid, 2, 3, 2, 5)
        self.assertTrue(metadata["near_capped"])
        self.assertEqual(int((near & ~target).sum()), metadata["far_pool_count"])
        self.assertEqual(int(near.sum()), int(far[0].sum()))

    def test_baselines_do_not_read_hidden_pixels(self):
        valid = torch.ones(3, 3, dtype=torch.bool)
        target = torch.zeros_like(valid)
        target[1, 1] = True
        input_mask = target.clone()
        input_mask[0, 1] = True
        patches = torch.arange(18).reshape(9, 2).float()
        before = visible_baselines(patches, target, input_mask, valid)
        patches[input_mask.flatten()] = 1e9
        after = visible_baselines(patches, target, input_mask, valid)
        for key in before:
            torch.testing.assert_close(before[key], after[key])

    def test_actual_error_uses_evaluation_target_not_extra_hidden_tokens(self):
        target = torch.zeros(9, 588)
        prediction = target.clone()
        prediction[4] = 2
        prediction[0] = 100
        evaluation = torch.zeros(9, dtype=torch.bool)
        evaluation[4] = True
        errors = patch_errors(prediction[evaluation], target[evaluation])
        self.assertEqual(float(errors["pixel_mse"].mean()), 4)
        self.assertEqual(float(errors["haar_l1"].mean()), 0)
        target[4, ::3] = torch.arange(196).float() % 2
        self.assertGreater(float(patch_errors(prediction[evaluation], target[evaluation])["haar_l1"].mean()), 0)

    def test_prototype_kl_and_ce_use_fixed_probability_targets(self):
        q = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
        before = q.clone()
        logits = q.log() * 0.1
        result = prototype_errors(logits, q)
        torch.testing.assert_close(result["ibot_kl"], torch.zeros(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(result["ibot_ce"], result["teacher_entropy"])
        torch.testing.assert_close(q, before)

    def test_correlation_handles_ties_and_constant_values(self):
        self.assertAlmostEqual(spearman([1, 2, 2, 4], [9, 7, 7, 1]), -1)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(spearman([1, 2], [2, 1]))


class LocalCheckpointMergeTests(unittest.TestCase):
    def reference(self):
        module = nn.Linear(3, 2)
        module.register_buffer("replicated", torch.tensor([1., 2.]))
        state = module.state_dict()
        rank0, rank1 = {}, {}
        for name, value in state.items():
            if name == "replicated":
                rank0["m." + name], rank1["m." + name] = value.clone(), value.clone()
            else:
                flat = value.flatten()
                rank0["m." + name], rank1["m." + name] = flat[:1], flat[1:]
        rank0["m._flat_param"] = torch.full((999,), 99.)
        rank1["m._flat_param"] = torch.full((999,), 99.)
        return module, [rank0, rank1]

    def test_named_fragments_merge_ignoring_flat_padding(self):
        module, ranks = self.reference()
        restored = merge_named_state(ranks, "m.", module)
        for key, value in module.state_dict().items():
            torch.testing.assert_close(restored[key], value)

    def test_incomplete_fragments_and_mismatched_buffers_rejected(self):
        module, ranks = self.reference()
        ranks[1]["m.weight"] = ranks[1]["m.weight"][:-1]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            merge_named_state(ranks, "m.", module)
        module, ranks = self.reference()
        ranks[1]["m.replicated"][0] += 1
        with self.assertRaisesRegex(ValueError, "buffer mismatch"):
            merge_named_state(ranks, "m.", module)


if __name__ == "__main__":
    unittest.main()
