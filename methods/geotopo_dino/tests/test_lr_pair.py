"""Check optimizer behavior, exact schedules, and paired experiment controls."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from dinov2.configs import dinov2_default_config
from dinov2.train.train import apply_optim_scheduler, build_schedulers as upstream_schedulers
from dinov2.utils.config import apply_scaling_rules_to_cfg
from methods.geotopo_dino.models.ssl_meta_arch import GeoTopoSSLMetaArch
from methods.geotopo_dino.train.schedules import build_schedulers, get_schedule_iterations
from methods.geotopo_dino.tests.test_pixel_reconstruction import TinyBackbone, tiny_config


def build_backbones(cfg):
    student = TinyBackbone()
    student.blocks = nn.ModuleList([nn.Linear(12, 12), nn.Linear(12, 12)])
    return student, copy.deepcopy(student), 12


def parameter_groups(model):
    return {
        id(parameter): group
        for group in model.get_params_groups()
        for parameter in group["params"]
    }


class BackboneLRTests(unittest.TestCase):
    @patch("dinov2.train.ssl_meta_arch.build_model_from_cfg", side_effect=build_backbones)
    def test_only_backbone_is_reduced_with_heads_and_decoder_enabled(self, _builder):
        cfg = tiny_config(True)
        cfg.gcvd.enabled = True
        for key, value in {"projection_dim": 8, "projection_hidden_dim": 16,
                           "loss_type": "prototype_ce", "head_n_prototypes": 8,
                           "head_hidden_dim": 16, "head_bottleneck_dim": 8,
                           "head_nlayers": 2}.items():
            cfg.gcvd[key] = value
        model = GeoTopoSSLMetaArch(cfg)
        original = parameter_groups(model)
        cfg.optim.backbone_lr_mult = 0.25
        scaled = parameter_groups(model)
        backbone_ids = set(map(id, model.student.backbone.parameters()))
        trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(set(scaled), trainable_ids)
        for parameter_id in scaled:
            expected = 0.25 if parameter_id in backbone_ids else 1.0
            self.assertAlmostEqual(scaled[parameter_id]["lr_multiplier"],
                                   original[parameter_id]["lr_multiplier"] * expected)
        # Verify applied rates, including existing patch/layer decay, and ensure
        # repeated group collection does not multiply the factor a second time.
        groups = model.get_params_groups()
        optimizer = torch.optim.AdamW(groups)
        apply_optim_scheduler(optimizer, 1e-3, 0.04, 1e-3)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                self.assertAlmostEqual(group["lr"], 1e-3 * scaled[id(parameter)]["lr_multiplier"])
        patch_id = id(model.student.backbone.patch_embed.proj.weight)
        self.assertAlmostEqual(scaled[patch_id]["lr_multiplier"], 0.25 * 0.2 * 0.9**3)
        # A real optimizer update on a scalar head gradient keeps the intended LR.
        parameter = model.student.geom_head.parameters().__next__()
        parameter.grad = torch.ones_like(parameter)
        before = parameter.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, parameter))

    def test_invalid_backbone_multiplier_fails_before_model_construction(self):
        for value in (0, -1, float("nan"), float("inf")):
            cfg = tiny_config(False)
            cfg.optim.backbone_lr_mult = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "backbone_lr_mult"):
                GeoTopoSSLMetaArch(cfg)

    @patch("dinov2.train.ssl_meta_arch.build_model_from_cfg", side_effect=build_backbones)
    def test_missing_multiplier_preserves_original_groups(self, _builder):
        cfg = tiny_config(False)
        del cfg.optim.backbone_lr_mult
        model = GeoTopoSSLMetaArch(cfg)
        missing = parameter_groups(model)
        cfg.optim.backbone_lr_mult = 1.0
        explicit = parameter_groups(model)
        self.assertEqual({p: g["lr_multiplier"] for p, g in missing.items()},
                         {p: g["lr_multiplier"] for p, g in explicit.items()})


class PairedScheduleTests(unittest.TestCase):
    def config(self, name="v025", gpu_count=4):
        return OmegaConf.merge(OmegaConf.create(dinov2_default_config),
                               OmegaConf.load(Path(__file__).parents[1] / f"configs/maira2_{name}_{gpu_count}gpu.yaml"))

    def test_pair_differs_only_in_objectives_and_output(self):
        for gpu_count in (2, 4):
            vanilla = OmegaConf.to_container(self.config(gpu_count=gpu_count))
            full = OmegaConf.to_container(self.config("f025", gpu_count=gpu_count))
            full["gcvd"]["enabled"] = False
            full["pixel_reconstruction"]["enabled"] = False
            full["train"]["output_dir"] = vanilla["train"]["output_dir"]
            self.assertEqual(full, vanilla)

    def test_two_gpu_sample_budget_and_actual_lr_scaling(self):
        cfg = self.config(gpu_count=2)
        self.assertEqual(cfg.train.batch_size_per_gpu, 8)
        self.assertEqual(cfg.train.stop_after_iterations, 29420)
        self.assertEqual(get_schedule_iterations(cfg), 29420)
        self.assertEqual(29420 * 16, 14710 * 32)
        with patch("dinov2.distributed.get_global_size", return_value=2):
            apply_scaling_rules_to_cfg(cfg)
        self.assertAlmostEqual(cfg.optim.lr, 0.0005)
        lr, wd, momentum, _, _ = build_schedulers(cfg)
        self.assertEqual(lr.total_iters, 29420)
        self.assertAlmostEqual(lr[5999], 0.0005)
        self.assertAlmostEqual(lr[5999] * cfg.optim.backbone_lr_mult, 0.000125)
        self.assertLess(lr[29419], 1.01e-6)
        self.assertGreater(wd[29419], 0.399999)
        self.assertGreater(momentum[29419], 0.999999)

    def test_exact_step_schedule_and_warmup(self):
        cfg = self.config()
        cfg.optim.lr = float(0.004 * np.sqrt(32 / 1024))
        original = OmegaConf.to_container(cfg)
        for steps in (14710, 20000):
            cfg.optim.total_iterations = steps
            lr, wd, mom, temp, last = build_schedulers(cfg)
            for schedule in (lr, wd, mom, last):
                self.assertEqual(schedule.total_iters, steps)
                self.assertEqual(len(schedule.schedule), steps)
            self.assertEqual(lr[0], 0)
            self.assertAlmostEqual(lr[5999], cfg.optim.lr)
            self.assertAlmostEqual(temp[5999], 0.07)
            self.assertEqual(last[999], 0)
            self.assertGreater(last[1000], 0)
            self.assertLess(lr[steps - 1], 1.01e-6)
            self.assertGreater(mom[steps - 1], 0.999999)
            self.assertGreater(wd[steps - 1], 0.399999)
            self.assertEqual(get_schedule_iterations(cfg), steps)
        cfg.optim.total_iterations = original["optim"]["total_iterations"]
        self.assertEqual(OmegaConf.to_container(cfg), original)

    def test_legacy_schedule_unchanged_and_invalid_budget_rejected(self):
        cfg = self.config()
        cfg.optim.total_iterations = 0
        self.assertEqual(get_schedule_iterations(cfg), 14800)
        for left, right in zip(build_schedulers(cfg), upstream_schedulers(cfg)):
            np.testing.assert_array_equal(left.schedule, right.schedule)
        for value in (-1, 1.5, True):
            cfg.optim.total_iterations = value
            with self.assertRaises(ValueError):
                get_schedule_iterations(cfg)
        cfg.optim.total_iterations = 6000
        with self.assertRaisesRegex(ValueError, "warmup"):
            build_schedulers(cfg)


if __name__ == "__main__":
    unittest.main()
