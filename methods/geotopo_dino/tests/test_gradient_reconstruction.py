import copy
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from methods.geotopo_dino.losses.gradient_reconstruction_loss import (
    GradientReconstructionLoss, sobel_gradients,
)
from methods.geotopo_dino.losses.pixel_reconstruction_loss import (
    PixelReconstructionLoss, global_reconstruction_mask, patchify,
)
from methods.geotopo_dino.models.ssl_meta_arch import GeoTopoSSLMetaArch
from methods.geotopo_dino.tests.test_pixel_reconstruction import (
    build_tiny_backbones, pack_heads, synthetic_batch, tiny_config,
)
from methods.geotopo_dino.tests.test_wavelet_reconstruction import wavelet_config
from methods.geotopo_dino.train.checkpoint import GeoTopoCheckpointer, validate_gradient_checkpoint


def gradient_config(enabled=True, **overrides):
    cfg = tiny_config(True)
    cfg.gradient_reconstruction = {
        "enabled": enabled, "loss_weight": 0.1, "warmup_iterations": 0, **overrides,
    }
    return cfg


def distributed_inputs(rank, case):
    images = torch.zeros(1, 3, 8, 8)
    ramp = torch.arange(4).float()[None].expand(4, 4).repeat(3, 2, 2)[None] * (rank + 1)
    prediction = patchify(ramp, 4).requires_grad_()
    count = ((1, 3), (0, 3), (0, 0))[case][rank]
    return prediction, images, torch.arange(4)[None] < count


def distributed_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        results = []
        for case in range(3):
            prediction, images, masks = distributed_inputs(rank, case)
            loss = GradientReconstructionLoss(4)(prediction, images, masks)
            loss.backward()
            results.append({"loss": loss.detach(), "gradient": prediction.grad})
        torch.save(results, Path(directory) / f"rank{rank}.pth")
    finally:
        dist.destroy_process_group()


class GradientLossTests(unittest.TestCase):
    def test_unit_ramps_direction_sign_and_normalization(self):
        y, x = torch.meshgrid(torch.arange(4).float(), torch.arange(5).float(), indexing="ij")
        image = torch.stack((x, y, -2 * x + 3 * y))[None]
        expected = torch.tensor([[[1., 0., -2.], [0., 1., 3.]]])[..., None, None].expand(1, 2, 3, 2, 3)
        torch.testing.assert_close(sobel_gradients(image), expected, rtol=0, atol=0)
        target = x[:, :4][None, None]
        # Opposite signed gradients must be penalized even at equal magnitude.
        loss = GradientReconstructionLoss(4)(patchify(-target, 4), target, torch.ones(1, 1, dtype=torch.bool))
        self.assertAlmostEqual(loss.item(), 1)

    def test_rgb_patch_layout_with_analytic_independent_slopes(self):
        p = 14
        images = torch.randn(2, 3, 28, 42)
        prediction_images = images.clone()
        slopes = torch.arange(2 * 6 * 3 * 2).reshape(2, 6, 3, 2).float() / 20 - 1
        y, x = torch.meshgrid(torch.arange(p).float(), torch.arange(p).float(), indexing="ij")
        for batch in range(2):
            for row in range(2):
                for col in range(3):
                    sx, sy = slopes[batch, row * 3 + col].unbind(-1)
                    prediction_images[batch, :, row*p:(row+1)*p, col*p:(col+1)*p] += sx[:, None, None] * x + sy[:, None, None] * y
        masks = torch.tensor([[True, False, True, False, False, True],
                              [False, True, True, False, True, False]])
        expected = slopes.abs().mean(dim=(2, 3))[masks].mean()
        actual = GradientReconstructionLoss(p)(patchify(prediction_images, p), images, masks)
        torch.testing.assert_close(actual, expected)

    def test_patch_offsets_do_not_create_boundary_edges(self):
        images = torch.zeros(1, 3, 8, 8)
        prediction = patchify(images, 4) + torch.tensor([1., 100., -20., 40.])[None, :, None]
        masks = torch.ones(1, 4, dtype=torch.bool)
        self.assertEqual(GradientReconstructionLoss(4)(prediction, images, masks).item(), 0)
        self.assertGreater(PixelReconstructionLoss(4)(prediction, images, masks).item(), 0)

    def test_final_mask_padding_target_detach_and_unmasked_gradients(self):
        images = torch.zeros(2, 3, 8, 8, requires_grad=True)
        raw_mask = torch.ones(2, 4, dtype=torch.bool)
        raw_mask[1, 3] = False
        masks = global_reconstruction_mask(raw_mask, torch.tensor([[[False, True], [True, False]]]))
        ramp = torch.arange(4).float()[None].expand(4, 4).repeat(2, 3, 2, 2)
        prediction = patchify(ramp, 4)
        prediction[~masks] = torch.randn_like(prediction[~masks]) * 100
        prediction.requires_grad_()
        loss = GradientReconstructionLoss(4)(prediction, images, masks)
        self.assertAlmostEqual(loss.item(), 0.5)
        loss.backward()
        self.assertEqual(prediction.grad[~masks].count_nonzero().item(), 0)
        self.assertGreater(prediction.grad[masks].abs().sum().item(), 0)
        self.assertIsNone(images.grad)

    def test_empty_masks_keep_zero_gradient_graph(self):
        images = torch.randn(2, 3, 8, 8, requires_grad=True)
        prediction = torch.randn(2, 4, 48, requires_grad=True)
        loss = GradientReconstructionLoss(4)(prediction, images, torch.zeros(2, 4, dtype=torch.bool))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertEqual(prediction.grad.count_nonzero().item(), 0)
        self.assertIsNone(images.grad)

    def test_sobel_and_loss_stay_fp32_inside_autocast(self):
        images = torch.randn(2, 3, 8, 8, dtype=torch.float16, requires_grad=True)
        prediction = torch.randn(2, 4, 48, dtype=torch.float16, requires_grad=True)
        masks = torch.ones(2, 4, dtype=torch.bool)
        expected = GradientReconstructionLoss(4)(prediction.float(), images.float(), masks)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            gradients = sobel_gradients(images)
            actual = GradientReconstructionLoss(4)(prediction, images, masks)
        self.assertEqual(gradients.dtype, torch.float32)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertIsNone(images.grad)

    def test_invalid_patch_sizes_and_shapes(self):
        for size in (-1, 0, 2):
            with self.assertRaises(ValueError):
                GradientReconstructionLoss(size)
        with self.assertRaises(ValueError):
            sobel_gradients(torch.zeros(1, 3, 2, 4))
        with self.assertRaisesRegex(ValueError, "same patch grid"):
            GradientReconstructionLoss(4)(torch.zeros(1, 3, 48), torch.zeros(1, 3, 8, 8), torch.ones(1, 4))

    @unittest.skipUnless(dist.is_gloo_available(), "two-rank normalization requires Gloo")
    def test_real_two_rank_unequal_and_empty_counts_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=(str(Path(directory) / "rendezvous"), directory), nprocs=2, join=True)
            results = [torch.load(Path(directory) / f"rank{rank}.pth", weights_only=True) for rank in range(2)]
        for case in range(3):
            inputs = [distributed_inputs(rank, case) for rank in range(2)]
            prediction = torch.cat([item[0].detach() for item in inputs]).requires_grad_()
            reference = GradientReconstructionLoss(4)(prediction, torch.cat([item[1] for item in inputs]),
                                                       torch.cat([item[2] for item in inputs]))
            reference.backward()
            torch.testing.assert_close(sum(result[case]["loss"] for result in results) / 2, reference)
            for rank in range(2):
                torch.testing.assert_close(results[rank][case]["gradient"] / 2, prediction.grad[rank:rank+1])


class GradientIntegrationTests(unittest.TestCase):
    def setUp(self):
        builder = patch("dinov2.train.ssl_meta_arch.build_model_from_cfg", side_effect=build_tiny_backbones)
        builder.start()
        self.addCleanup(builder.stop)

    def model(self, cfg=None):
        model = GeoTopoSSLMetaArch(gradient_config() if cfg is None else cfg)
        model.need_to_synchronize_fsdp_streams = False
        model.teacher.load_state_dict(model.student.state_dict())
        return model

    def forward(self, model, batch, iteration=10):
        with patch.object(torch.Tensor, "cuda", lambda x, **kwargs: x), \
             patch("methods.geotopo_dino.models.ssl_meta_arch.fmha.BlockDiagonalMask.from_tensor_list", side_effect=pack_heads):
            return model.forward_backward(batch, teacher_temp=0.07, iteration=iteration)

    def test_disabled_preserves_f025_and_fw_state_rng_losses_and_gradients(self):
        for cfg in (tiny_config(True), wavelet_config()):
            torch.manual_seed(123)
            baseline = self.model(cfg)
            rng = torch.get_rng_state()
            disabled_cfg = copy.deepcopy(cfg)
            disabled_cfg.gradient_reconstruction = {"enabled": False}
            torch.manual_seed(123)
            disabled = self.model(disabled_cfg)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(baseline.state_dict().keys(), disabled.state_dict().keys())
            for key, value in baseline.state_dict().items():
                torch.testing.assert_close(value, disabled.state_dict()[key], rtol=0, atol=0)
            batch = synthetic_batch()
            left, right = self.forward(baseline, batch), self.forward(disabled, batch)
            self.assertEqual(left.keys(), right.keys())
            self.assertFalse(any(key.startswith("gradient_") for key in right))
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            for a, b in zip(baseline.parameters(), disabled.parameters()):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_enabled_reuses_prediction_target_mask_without_new_parameters(self):
        torch.manual_seed(123)
        baseline = self.model(gradient_config(False))
        rng = torch.get_rng_state()
        torch.manual_seed(123)
        enabled = self.model()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(baseline.state_dict().keys(), enabled.state_dict().keys())
        batch = synthetic_batch()
        batch["anchor_valid_masks"][0, 0, 0] = False
        seen = {}
        handles = [loss.register_forward_pre_hook(
            lambda _module, args, name=name: seen.update({name: args})
        ) for name, loss in (("pixel", enabled.pixel_loss), ("gradient", enabled.gradient_loss))]
        left, right = self.forward(baseline, batch), self.forward(enabled, batch)
        for handle in handles:
            handle.remove()
        for i in range(3):
            self.assertIs(seen["pixel"][i], seen["gradient"][i])
        self.assertFalse(seen["gradient"][2][0, 0])
        self.assertGreater(right["gradient_raw_loss"].item(), 0)
        torch.testing.assert_close(right["optimization_loss"], left["optimization_loss"] + right["gradient_weighted_loss"])
        self.assertFalse(torch.equal(baseline.student.backbone.mix.weight.grad, enabled.student.backbone.mix.weight.grad))
        self.assertTrue(all(p.grad is not None for p in enabled.student_aux.parameters()))
        self.assertTrue(all(p.grad is None for p in enabled.teacher.parameters()))

    def test_zero_weight_and_initial_warmup_preserve_f025_gradients(self):
        for weight, warmup, iteration in ((0, 0, 10), (0.1, 1000, 0)):
            torch.manual_seed(123)
            baseline = self.model(gradient_config(False))
            torch.manual_seed(123)
            enabled = self.model(gradient_config(loss_weight=weight, warmup_iterations=warmup))
            batch = synthetic_batch()
            left, right = self.forward(baseline, batch, iteration), self.forward(enabled, batch, iteration)
            self.assertEqual(right["gradient_weighted_loss"].item(), 0)
            torch.testing.assert_close(left["optimization_loss"], right["optimization_loss"], rtol=0, atol=0)
            for a, b in zip(baseline.parameters(), enabled.parameters()):
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_warmup_and_gcvd_sinkhorn_combination(self):
        cfg = gradient_config(warmup_iterations=1000)
        cfg.gcvd.update({"enabled": True, "loss_type": "prototype_ce", "projection_hidden_dim": 16,
                         "projection_dim": 8, "head_hidden_dim": 16, "head_bottleneck_dim": 8,
                         "head_n_prototypes": 8, "head_nlayers": 2,
                         "teacher_centering": "sinkhorn_knopp", "sinkhorn_iterations": 3})
        for iteration, expected in ((500, 0.5), (1000, 1), (3000, 1)):
            model = self.model(cfg)
            result = self.forward(model, synthetic_batch(), iteration)
            self.assertEqual(result["gradient_warmup_scale"].item(), expected)
            torch.testing.assert_close(result["gradient_weighted_loss"], result["gradient_raw_loss"] * 0.1 * expected)
            self.assertIn("gcvd_dense_raw_loss", result)
            self.assertNotIn("wavelet_raw_loss", result)
            self.assertTrue(torch.isfinite(result["optimization_loss"]))

    def test_invalid_configuration_and_mutual_exclusion(self):
        configs = [gradient_config(loss_weight=w) for w in (-1, float("nan"), float("inf"))]
        configs += [gradient_config(warmup_iterations=w) for w in (-1, 0.5)]
        for field, value in (("enabled", False), ("norm_pix_loss", True)):
            cfg = gradient_config()
            cfg.pixel_reconstruction[field] = value
            configs.append(cfg)
        cfg = gradient_config()
        cfg.wavelet_reconstruction = {"enabled": True}
        configs.append(cfg)
        for cfg in configs:
            with self.subTest(cfg=cfg.gradient_reconstruction), self.assertRaises(ValueError):
                self.model(cfg)

    def test_checkpoint_roundtrip_legacy_compatibility_and_mismatches(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.get_params_groups(), lr=1e-3)
        self.forward(model, synthetic_batch())
        optimizer.step()
        restored = self.model()
        restored_optimizer = torch.optim.AdamW(restored.get_params_groups(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory, \
             patch("dinov2.fsdp.FSDP.state_dict_type", side_effect=lambda *args: nullcontext()):
            checkpointer = GeoTopoCheckpointer(model, directory, optimizer=optimizer)
            checkpointer.save("model", iteration=10)
            loaded = torch.load(checkpointer.get_checkpoint_file(), weights_only=True)
            result = GeoTopoCheckpointer(restored, directory, optimizer=restored_optimizer).resume_or_load("", resume=True)
            self.assertEqual(result["iteration"], 10)
            self.assertEqual(loaded["gradient_reconstruction_signature"], model.gradient_reconstruction_signature)
            for cfg in (gradient_config(False), gradient_config(loss_weight=0.2), gradient_config(warmup_iterations=1000)):
                incompatible = self.model(cfg)
                before = copy.deepcopy(incompatible.state_dict())
                with self.assertRaisesRegex(ValueError, "Gradient reconstruction checkpoint/config mismatch"):
                    GeoTopoCheckpointer(incompatible, directory).resume_or_load("", resume=True)
                for key, value in incompatible.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])
        left_state, right_state = optimizer.state_dict(), restored_optimizer.state_dict()
        self.assertEqual(left_state["param_groups"], right_state["param_groups"])
        for index, values in left_state["state"].items():
            for key, value in values.items():
                torch.testing.assert_close(value, right_state["state"][index][key])
        validate_gradient_checkpoint({}, {"enabled": False})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_gradient_checkpoint({}, model.gradient_reconstruction_signature)

    def test_config_preserves_f025_except_output_and_gradient_objective(self):
        directory = Path(__file__).parents[1] / "configs"
        baseline = OmegaConf.load(directory / "maira2_f025_2gpu.yaml")
        grad = OmegaConf.load(directory / "maira2_fgrad025_2gpu.yaml")
        self.assertEqual(grad.pop("wavelet_reconstruction"), {"enabled": False})
        self.assertEqual(grad.pop("gradient_reconstruction"),
                         {"enabled": True, "loss_weight": 0.1, "warmup_iterations": 1000})
        del baseline.train.output_dir
        del grad.train.output_dir
        self.assertEqual(OmegaConf.to_container(baseline), OmegaConf.to_container(grad))


if __name__ == "__main__":
    unittest.main()
