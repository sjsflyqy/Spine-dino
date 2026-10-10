"""Curriculum, mixed-mask invariants, actual loss wiring and resume controls."""

import copy
from contextlib import nullcontext
from dataclasses import replace
import random
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
from torch.nn import functional as F

from methods.geotopo_dino.masking import BlockMaskPolicy, pack_masks
from methods.geotopo_dino.masking.structure_mask import (
    StructureMaskSettings, StructureMaxAMaskPolicy, build_mask_policy, build_maxa_ribbon_mask,
    core_context_metrics, policy_seed, structure_probability,
)
from methods.geotopo_dino.models.ssl_meta_arch import GeoTopoSSLMetaArch
from methods.geotopo_dino.tests.test_pixel_reconstruction import tiny_config, build_tiny_backbones, pack_heads
from methods.geotopo_dino.train.checkpoint import GeoTopoCheckpointer, validate_masking_checkpoint


def synthetic_geometry(height=37, width=37, count=429):
    valid = torch.zeros(height, width, dtype=torch.bool)
    valid[:, 3:width - 3] = True
    score = valid.float() * 0.1
    columns = torch.arange(width).float()
    for row in range(height):
        center = width // 2 + round(2 * torch.sin(torch.tensor(row / 8.)).item())
        score[row] += torch.exp(-(columns - center).square() / 8) * 0.8 * valid[row]
    baseline = torch.zeros_like(valid)
    ids = valid.flatten().nonzero().flatten()[:count]
    baseline.flatten()[ids] = True
    return score, valid, baseline


class StructurePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_schedule_boundaries_and_restart_offset(self):
        settings = StructureMaskSettings()
        for step, expected in ((0, 0), (5999, 0), (6000, 0), (9000, .15), (12000, .3), (29419, .3)):
            self.assertAlmostEqual(structure_probability(step, settings), expected)
        offset = replace(settings, start_iteration=10000)
        self.assertEqual(structure_probability(15999, offset), 0)
        self.assertAlmostEqual(structure_probability(19000, offset), .15)
        self.assertEqual(structure_probability(0, replace(settings, warmup_iterations=0, ramp_iterations=0)), .3)

    def test_invalid_settings_and_policy_names_rejected(self):
        for fields in ({"max_probability": 1.1}, {"max_probability": float("nan")}, {"warmup_iterations": True},
                       {"ramp_iterations": -1}, {"strip_width": 4}, {"min_width": 7}, {"smooth_kernel": 2},
                       {"guard_radius": 0}, {"context_rows": 0}, {"width_mode": "bad"}, {"num_candidates": 1}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                StructureMaskSettings(**fields)
        for fields in ({"anchor_policy": "unknown"}, {"random_global_policy": "structure_maxa"}):
            with self.assertRaises(NotImplementedError):
                build_mask_policy(fields)
        self.assertIsInstance(build_mask_policy({}), BlockMaskPolicy)

    def test_adaptive_core_supplement_budget_context_and_max_score(self):
        score, valid, baseline = synthetic_geometry()
        result = build_maxa_ribbon_mask(score, valid, baseline, StructureMaskSettings(), seed=42)
        self.assertTrue(result.diagnostics["applied"], result.diagnostics)
        self.assertEqual(int(result.mask.sum()), int(baseline.sum()))
        self.assertFalse((result.mask & ~valid).any())
        self.assertFalse((result.mask & result.context).any())
        guard = F.max_pool2d(result.core.float()[None, None], 3, 1, 1)[0, 0].bool()
        self.assertFalse((result.supplement & guard).any())
        torch.testing.assert_close(result.mask, result.core | result.supplement)
        diag = result.diagnostics
        self.assertEqual(diag["best_index"], max(diag["eligible_candidate_indices"], key=lambda i: diag["candidate_scores"][i]))
        self.assertAlmostEqual(diag["final_core_A"], float(score[result.core].mean()))
        packed = pack_masks(torch.stack((result.mask.flatten(), baseline.flatten())), upperbound=2 * int(baseline.sum()))
        self.assertEqual(packed.n_masked_patches, 2 * int(baseline.sum()))

    def test_flat_short_narrow_and_over_budget_fall_back_exactly(self):
        score, valid, baseline = synthetic_geometry(count=100)
        result = build_maxa_ribbon_mask(score, valid, baseline, StructureMaskSettings(), seed=42, near_constant=True)
        self.assertEqual(result.diagnostics["reason"], "near_constant_structure_map")
        torch.testing.assert_close(result.mask, baseline)
        for height, width in ((4, 13), (20, 9)):
            score, valid, baseline = synthetic_geometry(height, width, 10)
            result = build_maxa_ribbon_mask(score, valid, baseline, StructureMaskSettings(), seed=42)
            self.assertFalse(result.diagnostics["applied"])
            torch.testing.assert_close(result.mask, baseline)
        score, valid, baseline = synthetic_geometry(count=100)
        score = valid.float() * .1
        score[:, 14:23] = .9
        result = build_maxa_ribbon_mask(score, valid, baseline, StructureMaskSettings(), seed=42)
        self.assertFalse(result.diagnostics["applied"])
        self.assertEqual(result.diagnostics["reason"], "adaptive_core_exceeds_budget")
        torch.testing.assert_close(result.mask, baseline)

    def test_horizontal_ribbons_and_empty_masks(self):
        score, valid, baseline = synthetic_geometry()
        settings = replace(StructureMaskSettings(), width_mode="fixed", path_axis="horizontal")
        result = build_maxa_ribbon_mask(score.T, valid.T, baseline.T, settings, seed=42)
        self.assertTrue(result.diagnostics["applied"])
        self.assertEqual(result.diagnostics["axis"], "horizontal")
        self.assertEqual(int(result.mask.sum()), int(baseline.sum()))
        empty = build_maxa_ribbon_mask(score, valid, torch.zeros_like(valid), settings, seed=42)
        self.assertFalse(empty.mask.any())

    def batch(self, batch=3):
        score, valid, baseline = synthetic_geometry()
        masks = baseline.flatten().expand(batch * 2, -1).clone()
        masks[1].zero_()
        # Random high-dimensional tokens produce a nonflat score map.
        tokens = torch.randn(batch, valid.numel(), 16, generator=torch.Generator().manual_seed(123))
        return masks, tokens, valid.expand(batch, -1, -1).clone()

    def test_noop_warmup_does_not_score_or_consume_rng(self):
        masks, tokens, valid = self.batch()
        policy = StructureMaxAMaskPolicy()
        state, py_state = torch.random.get_rng_state().clone(), random.getstate()
        with patch("methods.geotopo_dino.masking.structure_mask.compute_structure_score", side_effect=AssertionError("warmup scored")):
            final, topology = policy.select(masks, teacher_anchor_tokens=tokens, anchor_valid_mask=valid, progress=.1, iteration=5999)
        self.assertIs(final, masks)
        self.assertIsNone(topology)
        self.assertEqual(policy.last_metrics["mask_structure_attempted_anchors"], 0)
        torch.testing.assert_close(state, torch.random.get_rng_state())
        self.assertEqual(py_state, random.getstate())

    def test_only_masked_anchors_change_rng_independence_and_resume(self):
        masks, tokens, valid = self.batch()
        settings = replace(StructureMaskSettings(), warmup_iterations=0, ramp_iterations=0, max_probability=1,
                           width_mode="fixed", visualization_period=1, context_log_period=1)
        policy = StructureMaxAMaskPolicy(settings)
        original = masks.clone()
        state, py_state = torch.random.get_rng_state().clone(), random.getstate()
        final, topology = policy.select(masks, teacher_anchor_tokens=tokens, anchor_valid_mask=valid, progress=.5, iteration=15000)
        self.assertIsNone(topology)
        self.assertEqual(policy.last_metrics["mask_structure_applied_anchors"], 2)
        self.assertEqual(policy.last_metrics["mask_structure_context_scored_cores"], 2)
        torch.testing.assert_close(final[3:], masks[3:])
        torch.testing.assert_close(final[1], masks[1])
        torch.testing.assert_close(final.sum(-1), masks.sum(-1))
        torch.testing.assert_close(original, masks)
        torch.testing.assert_close(state, torch.random.get_rng_state())
        self.assertEqual(py_state, random.getstate())
        fresh = StructureMaxAMaskPolicy(settings)
        resumed, _ = fresh.select(masks, teacher_anchor_tokens=tokens, anchor_valid_mask=valid, progress=0, iteration=15000)
        torch.testing.assert_close(final, resumed)
        self.assertEqual(policy.last_records, fresh.last_records)
        self.assertNotEqual(policy_seed(42, 15000, 0, 0, "gate"), policy_seed(42, 15000, 1, 0, "gate"))

    def test_context_diagnostic_uses_full_mask_visibility(self):
        from visualization.structure_maps.context_metrics import context_score
        _, valid, full = synthetic_geometry()
        core = torch.zeros_like(valid);core[4:20, 16:21] = True
        full |= core
        tokens = torch.randn(valid.numel(), 8)
        _, expected = context_score(tokens, core, full, valid, 5, .7)
        actual = core_context_metrics(tokens, core, full, valid, 5, .7)
        self.assertAlmostEqual(actual["P"], expected["P"])
        self.assertAlmostEqual(actual["H"], expected["H"])
        self.assertAlmostEqual(actual["missing"], expected["missing_fraction"])


def structure_tiny_config(enabled=True):
    cfg = tiny_config(True)
    cfg.crops.global_crops_size = 168
    cfg.gcvd.anchor_size = 168
    cfg.gcvd.update({"enabled": True, "loss_type": "prototype_ce", "projection_dim": 8,
        "projection_hidden_dim": 16, "head_n_prototypes": 8, "head_hidden_dim": 16,
        "head_bottleneck_dim": 8, "head_nlayers": 2, "teacher_centering": "sinkhorn_knopp"})
    cfg.pixel_reconstruction.visualization_period = 0
    if enabled:
        cfg.masking.anchor_policy = "structure_maxa"
        cfg.masking.structure = {"warmup_iterations": 0, "ramp_iterations": 0, "max_probability": 1,
            "strip_width": 3, "width_mode": "fixed", "span_fraction": .4, "context_rows": 1,
            "context_log_period": 1, "visualization_period": 0}
    return cfg


def training_batch():
    batch, grid = 2, 12
    masks = torch.zeros(2 * batch, grid * grid, dtype=torch.bool)
    masks[:, :48] = True
    return {"collated_global_crops": torch.randn(2 * batch, 3, 168, 168),
            "collated_local_crops": torch.randn(batch, 3, 28, 28), "collated_masks": masks,
            "anchor_transforms": torch.eye(3).expand(batch, -1, -1),
            "anchor_valid_masks": torch.ones(batch, grid, grid, dtype=torch.bool),
            "local_transforms": torch.eye(3).expand(batch, -1, -1),
            "local_sample_ids": torch.arange(batch), "upperbound": int(masks.sum())}


class StructureTrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.builder = patch("dinov2.train.ssl_meta_arch.build_model_from_cfg", side_effect=build_tiny_backbones)
        self.builder.start()
        self.addCleanup(self.builder.stop)

    def model(self, enabled=True):
        model = GeoTopoSSLMetaArch(structure_tiny_config(enabled))
        model.need_to_synchronize_fsdp_streams = False
        model.teacher.load_state_dict(model.student.state_dict())
        return model

    def forward(self, model, batch, iteration=9000):
        with patch.object(torch.Tensor, "cuda", lambda x, **kwargs: x), \
             patch("methods.geotopo_dino.models.ssl_meta_arch.fmha.BlockDiagonalMask.from_tensor_list", side_effect=pack_heads):
            return model.forward_backward(batch, teacher_temp=.07, iteration=iteration)

    def test_final_masks_reach_student_ibot_pixel_with_gcvd_and_optimizer(self):
        model = self.model()
        batch = training_batch()
        original = batch["collated_masks"].clone()
        seen = {}
        original_forward = model.student.backbone.forward
        def student_forward(images, masks=None, is_training=False):
            if isinstance(images, list):
                seen["student_masks"] = masks[0].clone()
            return original_forward(images, masks, is_training)
        model.student.backbone.forward = student_forward
        original_ibot = model.ibot_patch_loss.forward_masked
        def ibot_forward(*args, **kwargs):
            seen["ibot_masks"] = kwargs["student_masks_flat"].clone()
            seen["ibot_count"] = kwargs["n_masked_patches"]
            return original_ibot(*args, **kwargs)
        hook = model.pixel_loss.register_forward_pre_hook(lambda module, args: seen.update(pixel_masks=args[2].clone()))
        with patch.object(model.ibot_patch_loss, "forward_masked", side_effect=ibot_forward):
            result = self.forward(model, batch)
        hook.remove()
        self.assertGreater(result["mask_structure_applied_anchors"].item(), 0)
        self.assertIn("gcvd_dense_raw_loss", result)
        self.assertTrue(all(torch.isfinite(value).all() for value in result.values()))
        torch.testing.assert_close(seen["student_masks"], seen["ibot_masks"])
        torch.testing.assert_close(seen["student_masks"], seen["pixel_masks"])
        torch.testing.assert_close(seen["student_masks"].sum(-1), original.sum(-1))
        torch.testing.assert_close(seen["student_masks"][2:], original[2:])
        self.assertEqual(seen["ibot_count"], int(original.sum()))
        torch.testing.assert_close(batch["collated_masks"], original)
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        self.assertGreater(model.student_aux.pixel_decoder.pred.weight.grad.abs().sum().item(), 0)
        optimizer = torch.optim.AdamW(model.get_params_groups(), lr=1e-3)
        before = model.student.backbone.mix.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.student.backbone.mix.weight))

    def test_warmup_preserves_parameters_rng_losses_and_gradients(self):
        torch.manual_seed(17);baseline = self.model(False)
        rng = torch.random.get_rng_state().clone()
        torch.manual_seed(17);structured = self.model(True)
        torch.testing.assert_close(rng, torch.random.get_rng_state())
        self.assertEqual(baseline.state_dict().keys(), structured.state_dict().keys())
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, structured.state_dict()[key], rtol=0, atol=0)
        structured.mask_policy.settings = replace(structured.mask_policy.settings, warmup_iterations=6000, ramp_iterations=6000)
        batch = training_batch()
        left, right = self.forward(baseline, batch, 10), self.forward(structured, batch, 10)
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        for a, b in zip(baseline.student.parameters(), structured.student.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_training_snapshot_contains_actual_masks_and_diagnostics(self):
        import json
        from pathlib import Path
        from PIL import Image
        model = self.model()
        model.mask_policy.settings = replace(model.mask_policy.settings, visualization_period=1)
        with tempfile.TemporaryDirectory() as temp:
            model.cfg.train.output_dir = temp
            result = self.forward(model, training_batch(), iteration=9000)
            self.assertGreater(result["mask_structure_applied_anchors"].item(), 0)
            path = Path(temp) / "structure_masks" / "step_0009001.png"
            with Image.open(path) as image:
                self.assertEqual(image.width, 5 * 330)
            metadata = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(metadata["iteration"], 9000)
            self.assertEqual(metadata["probability"], 1)
            for example in metadata["examples"]:
                if example["applied"]:
                    self.assertEqual(example["core_count"] + example["supplement_count"], 48)

    def test_checkpoint_roundtrip_continues_schedule_and_rejects_policy_change(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.get_params_groups(), lr=1e-3)
        batch = training_batch();self.forward(model, batch);optimizer.step()
        restored = self.model()
        restored_optimizer = torch.optim.AdamW(restored.get_params_groups(), lr=1e-3)
        with tempfile.TemporaryDirectory() as temp, \
             patch("dinov2.fsdp.FSDP.state_dict_type", side_effect=lambda *args: nullcontext()):
            checkpointer = GeoTopoCheckpointer(model, temp, optimizer=optimizer)
            checkpointer.save("structure", iteration=9000)
            state = GeoTopoCheckpointer(restored, temp, optimizer=restored_optimizer).resume_or_load("", resume=True)
            self.assertEqual(state["iteration"], 9000)
            with self.assertRaisesRegex(ValueError, "Mask policy.*mismatch"):
                GeoTopoCheckpointer(self.model(False), temp).resume_or_load("", resume=True)
        self.assertEqual(len(optimizer.state), len(restored_optimizer.state))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])
        self.forward(model, batch, 9001);self.forward(restored, batch, state["iteration"] + 1)
        self.assertEqual(model.mask_policy.last_records, restored.mask_policy.last_records)
        changed = copy.deepcopy(model.masking_signature);changed["settings"]["max_probability"] = .4
        with self.assertRaisesRegex(ValueError, "Mask policy.*mismatch"):
            validate_masking_checkpoint({"masking_signature": model.masking_signature}, changed)
        with self.assertRaisesRegex(ValueError, "Mask policy.*mismatch"):
            validate_masking_checkpoint({}, model.masking_signature)
        validate_masking_checkpoint({}, BlockMaskPolicy().signature)
        settings = replace(model.mask_policy.settings, visualization_period=300, context_log_period=0)
        self.assertEqual(StructureMaxAMaskPolicy(settings).signature, model.masking_signature)

    def test_paired_config_only_changes_masking_and_output(self):
        from pathlib import Path
        configs = Path(__file__).parents[1] / "configs"
        baseline = OmegaConf.to_container(OmegaConf.load(configs / "maira2_f025_2gpu.yaml"))
        structured = OmegaConf.to_container(OmegaConf.load(configs / "maira2_f025_structure_maxa_2gpu.yaml"))
        structured["masking"] = baseline["masking"]
        structured["train"]["output_dir"] = baseline["train"]["output_dir"]
        self.assertEqual(baseline, structured)


if __name__ == "__main__":
    unittest.main()
