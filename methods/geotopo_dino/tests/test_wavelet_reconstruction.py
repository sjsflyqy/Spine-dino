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
from torch.nn import functional as F

from methods.geotopo_dino.losses.pixel_reconstruction_loss import (
    PixelReconstructionLoss, global_reconstruction_mask, patchify,
)
from methods.geotopo_dino.losses.wavelet_reconstruction_loss import (
    WaveletReconstructionLoss, haar_details,
)
from methods.geotopo_dino.models.ssl_meta_arch import GeoTopoSSLMetaArch
from methods.geotopo_dino.tests.test_pixel_reconstruction import (
    build_tiny_backbones, pack_heads, synthetic_batch, tiny_config,
)
from methods.geotopo_dino.train.checkpoint import (
    GeoTopoCheckpointer, validate_wavelet_checkpoint,
)


def wavelet_config(enabled=True, **overrides):
    cfg = tiny_config(True)
    cfg.wavelet_reconstruction = {
        "enabled": enabled, "loss_weight": 0.1, "warmup_iterations": 0, **overrides,
    }
    return cfg


def distributed_inputs(rank, case):
    images = torch.zeros(1, 3, 4, 4)
    pattern = torch.tensor([[1., 1.], [-1., -1.]]).repeat(3, 2, 2)[None] * (rank + 1)
    prediction = patchify(pattern, 2).requires_grad_()
    count = ((1, 3), (0, 3), (0, 0))[case][rank]
    masks = torch.arange(4)[None] < count
    return prediction, images, masks


def distributed_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        results = []
        for case in range(3):
            prediction, images, masks = distributed_inputs(rank, case)
            loss = WaveletReconstructionLoss(2)(prediction, images, masks)
            loss.backward()
            results.append({"loss": loss.detach(), "gradient": prediction.grad})
        torch.save(results, Path(directory) / f"rank{rank}.pth")
    finally:
        dist.destroy_process_group()


class WaveletLossTests(unittest.TestCase):
    def test_analytic_haar_bands_and_orthonormal_scaling(self):
        patches = torch.tensor([[[[1., 2.], [3., 4.]],
                                 [[1., 1.], [-1., -1.]],
                                 [[1., -1.], [-1., 1.]]]])
        expected = torch.tensor([[[-2., 2., 0.], [-1., 0., 0.], [0., 0., 2.]]])
        torch.testing.assert_close(haar_details(patches)[..., 0, 0], expected)
        ll = patches.sum(dim=(-1, -2)) / 2
        torch.testing.assert_close(patches.square().sum(), ll.square().sum() + expected.square().sum())

    def test_rgb_layout_matches_independent_full_image_convolution(self):
        torch.manual_seed(7)
        images = torch.randn(2, 3, 28, 42)
        prediction_images = torch.randn_like(images)
        prediction = patchify(prediction_images, 14)
        masks = torch.tensor([[True, False, True, False, False, True],
                              [False, True, True, False, True, False]])
        # Independent whole-image DWT reference: boundaries align since P=14
        # is even, and no filter footprint crosses a patch boundary at level 1.
        filters = torch.tensor([[[1., 1.], [-1., -1.]],
                                [[1., -1.], [1., -1.]],
                                [[1., -1.], [-1., 1.]]])[:, None] / 2
        delta = (prediction_images - images).reshape(6, 1, 28, 42)
        error = F.conv2d(delta, filters, stride=2).abs().reshape(2, 3, 3, 14, 21)
        spatial_mask = masks.reshape(2, 2, 3).repeat_interleave(7, 1).repeat_interleave(7, 2)
        expected = error.permute(0, 3, 4, 1, 2)[spatial_mask].mean()
        actual = WaveletReconstructionLoss(14)(prediction, images, masks)
        torch.testing.assert_close(actual, expected)

    def test_constant_error_has_no_detail_penalty(self):
        images = torch.randn(2, 3, 4, 4)
        masks = torch.ones(2, 4, dtype=torch.bool)
        prediction = patchify(images, 2) + 2
        self.assertAlmostEqual(WaveletReconstructionLoss(2)(prediction, images, masks).item(), 0, places=6)
        self.assertAlmostEqual(PixelReconstructionLoss(2)(prediction, images, masks).item(), 4)

    def test_final_mask_excludes_anchor_padding_and_unmasked_predictions(self):
        images = torch.zeros(2, 3, 4, 4, requires_grad=True)
        raw_mask = torch.ones(2, 4, dtype=torch.bool)
        raw_mask[1, 3] = False
        valid = torch.tensor([[[False, True], [True, False]]])
        masks = global_reconstruction_mask(raw_mask, valid)
        prediction = patchify(torch.tensor([[1., 1.], [-1., -1.]]).repeat(2, 3, 2, 2), 2)
        prediction[~masks] = 100 * torch.randn_like(prediction[~masks])
        prediction.requires_grad_()
        loss = WaveletReconstructionLoss(2)(prediction, images, masks)
        self.assertAlmostEqual(loss.item(), 2 / 3, places=6)
        loss.backward()
        self.assertEqual(prediction.grad[~masks].count_nonzero().item(), 0)
        self.assertGreater(prediction.grad[masks].abs().sum().item(), 0)
        self.assertIsNone(images.grad)

    def test_empty_masks_keep_graph_and_fp32(self):
        images = torch.randn(2, 3, 4, 4, dtype=torch.float16, requires_grad=True)
        prediction = torch.randn(2, 4, 12, dtype=torch.float16, requires_grad=True)
        masks = torch.zeros(2, 4, dtype=torch.bool)
        loss = WaveletReconstructionLoss(2)(prediction, images, masks)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertEqual(prediction.grad.count_nonzero().item(), 0)
        self.assertIsNone(images.grad)

    def test_half_precision_targets_match_fp32_and_are_detached(self):
        images = torch.randn(2, 3, 4, 4, dtype=torch.float16, requires_grad=True)
        prediction = torch.randn(2, 4, 12, dtype=torch.float16, requires_grad=True)
        masks = torch.ones(2, 4, dtype=torch.bool)
        loss = WaveletReconstructionLoss(2)(prediction, images, masks)
        expected = WaveletReconstructionLoss(2)(prediction.float(), images.float(), masks)
        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertIsNone(images.grad)

    def test_invalid_patch_sizes_and_shapes(self):
        for size in (0, -2, 7):
            with self.assertRaisesRegex(ValueError, "positive even"):
                WaveletReconstructionLoss(size)
        with self.assertRaises(ValueError):
            haar_details(torch.zeros(1, 3, 7, 7))
        with self.assertRaisesRegex(ValueError, "same patch grid"):
            WaveletReconstructionLoss(2)(torch.zeros(1, 3, 12), torch.zeros(1, 3, 4, 4), torch.ones(1, 4))

    @unittest.skipUnless(dist.is_gloo_available(), "two-rank normalization requires Gloo")
    def test_real_two_rank_unequal_and_empty_counts_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=(str(Path(directory) / "rendezvous"), directory),
                     nprocs=2, join=True)
            results = [torch.load(Path(directory) / f"rank{rank}.pth", weights_only=True) for rank in range(2)]
        for case in range(3):
            inputs = [distributed_inputs(rank, case) for rank in range(2)]
            prediction = torch.cat([item[0].detach() for item in inputs]).requires_grad_()
            images = torch.cat([item[1] for item in inputs])
            masks = torch.cat([item[2] for item in inputs])
            reference = WaveletReconstructionLoss(2)(prediction, images, masks)
            reference.backward()
            torch.testing.assert_close(sum(result[case]["loss"] for result in results) / 2, reference)
            for rank in range(2):
                # FSDP averages these local gradients across the two ranks.
                torch.testing.assert_close(results[rank][case]["gradient"] / 2, prediction.grad[rank:rank+1])


class WaveletIntegrationTests(unittest.TestCase):
    def setUp(self):
        builder = patch("dinov2.train.ssl_meta_arch.build_model_from_cfg", side_effect=build_tiny_backbones)
        builder.start()
        self.addCleanup(builder.stop)

    def model(self, cfg=None):
        model = GeoTopoSSLMetaArch(wavelet_config() if cfg is None else cfg)
        model.need_to_synchronize_fsdp_streams = False
        model.teacher.load_state_dict(model.student.state_dict())
        return model

    def forward(self, model, batch, iteration=10):
        with patch.object(torch.Tensor, "cuda", lambda x, **kwargs: x), \
             patch("methods.geotopo_dino.models.ssl_meta_arch.fmha.BlockDiagonalMask.from_tensor_list", side_effect=pack_heads):
            return model.forward_backward(batch, teacher_temp=0.07, iteration=iteration)

    def test_disabled_is_identical_to_legacy_f025(self):
        torch.manual_seed(123)
        legacy = self.model(tiny_config(True))
        rng = torch.get_rng_state()
        torch.manual_seed(123)
        disabled = self.model(wavelet_config(False))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(legacy.state_dict().keys(), disabled.state_dict().keys())
        for key, value in legacy.state_dict().items():
            torch.testing.assert_close(value, disabled.state_dict()[key], rtol=0, atol=0)
        batch = synthetic_batch()
        left, right = self.forward(legacy, batch), self.forward(disabled, batch)
        self.assertEqual(left.keys(), right.keys())
        self.assertFalse(any(key.startswith("wavelet_") for key in right))
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        for a, b in zip(legacy.parameters(), disabled.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_enabled_reuses_prediction_target_mask_and_changes_gradients(self):
        torch.manual_seed(123)
        baseline = self.model(wavelet_config(False))
        rng = torch.get_rng_state()
        torch.manual_seed(123)
        enabled = self.model()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(baseline.state_dict().keys(), enabled.state_dict().keys())
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), sum(p.numel() for p in enabled.parameters()))
        batch = synthetic_batch()
        batch["anchor_valid_masks"][0, 0, 0] = False
        seen = {}
        handles = [loss.register_forward_pre_hook(
            lambda _module, args, name=name: seen.update({name: args})
        ) for name, loss in (("pixel", enabled.pixel_loss), ("wavelet", enabled.wavelet_loss))]
        left = self.forward(baseline, batch)
        right = self.forward(enabled, batch)
        for handle in handles:
            handle.remove()
        for i in range(3):
            self.assertIs(seen["pixel"][i], seen["wavelet"][i])
        self.assertIs(seen["wavelet"][1], batch["collated_global_crops"])
        self.assertFalse(seen["wavelet"][2][0, 0])
        self.assertGreater(right["wavelet_raw_loss"].item(), 0)
        torch.testing.assert_close(right["wavelet_weighted_loss"], right["wavelet_raw_loss"] * 0.1)
        torch.testing.assert_close(right["optimization_loss"], left["optimization_loss"] + right["wavelet_weighted_loss"])
        self.assertFalse(torch.equal(baseline.student.backbone.mix.weight.grad, enabled.student.backbone.mix.weight.grad))
        self.assertTrue(all(p.grad is not None for p in enabled.student_aux.parameters()))
        self.assertTrue(all(p.grad is None for p in enabled.teacher.parameters()))
        self.assertFalse(any("wavelet" in name for name in enabled.state_dict()))

    def test_weight_zero_reproduces_f025_gradients(self):
        torch.manual_seed(123)
        baseline = self.model(wavelet_config(False))
        torch.manual_seed(123)
        enabled = self.model(wavelet_config(loss_weight=0))
        batch = synthetic_batch()
        left, right = self.forward(baseline, batch), self.forward(enabled, batch)
        torch.testing.assert_close(left["optimization_loss"], right["optimization_loss"], rtol=0, atol=0)
        for a, b in zip(baseline.parameters(), enabled.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_warmup_follows_iteration_including_resume(self):
        for iteration, expected in ((0, 0), (500, 0.5), (1000, 1), (3000, 1)):
            model = self.model(wavelet_config(warmup_iterations=1000))
            result = self.forward(model, synthetic_batch(), iteration)
            self.assertEqual(result["wavelet_warmup_scale"].item(), expected)
            torch.testing.assert_close(result["wavelet_weighted_loss"], result["wavelet_raw_loss"] * 0.1 * expected)

    def test_gcvd_sinkhorn_pixel_wavelet_combination(self):
        cfg = wavelet_config()
        cfg.gcvd.update({"enabled": True, "loss_type": "prototype_ce", "projection_hidden_dim": 16,
                         "projection_dim": 8, "head_hidden_dim": 16, "head_bottleneck_dim": 8,
                         "head_n_prototypes": 8, "head_nlayers": 2,
                         "teacher_centering": "sinkhorn_knopp", "sinkhorn_iterations": 3})
        model = self.model(cfg)
        result = self.forward(model, synthetic_batch())
        for name in ("pixel_raw_loss", "wavelet_raw_loss", "gcvd_dense_raw_loss", "optimization_loss"):
            self.assertIn(name, result)
            self.assertTrue(torch.isfinite(result[name]))
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))

    def test_invalid_configuration(self):
        configs = []
        for weight in (-1, float("nan"), float("inf")):
            configs.append(wavelet_config(loss_weight=weight))
        for warmup in (-1, 0.5):
            configs.append(wavelet_config(warmup_iterations=warmup))
        cfg = wavelet_config()
        cfg.pixel_reconstruction.enabled = False
        configs.append(cfg)
        cfg = wavelet_config()
        cfg.pixel_reconstruction.norm_pix_loss = True
        configs.append(cfg)
        for cfg in configs:
            with self.subTest(cfg=cfg.wavelet_reconstruction), self.assertRaises(ValueError):
                self.model(cfg)

    def test_checkpoint_roundtrip_and_objective_mismatch_before_state_load(self):
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
            self.assertEqual(loaded["wavelet_reconstruction_signature"], model.wavelet_reconstruction_signature)
            for cfg in (wavelet_config(False), wavelet_config(loss_weight=0.2), wavelet_config(warmup_iterations=1000)):
                incompatible = self.model(cfg)
                before = copy.deepcopy(incompatible.state_dict())
                with self.assertRaisesRegex(ValueError, "Wavelet reconstruction checkpoint/config mismatch"):
                    GeoTopoCheckpointer(incompatible, directory).resume_or_load("", resume=True)
                for key, value in incompatible.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])
        self.assertEqual(len(optimizer.state), len(restored_optimizer.state))
        validate_wavelet_checkpoint({}, {"enabled": False})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_wavelet_checkpoint({}, model.wavelet_reconstruction_signature)

    def test_fw_configuration_only_changes_wavelet_objective_and_output(self):
        directory = Path(__file__).parents[1] / "configs"
        baseline = OmegaConf.load(directory / "maira2_f025_2gpu.yaml")
        fw = OmegaConf.load(directory / "maira2_fw025_2gpu.yaml")
        wave = fw.pop("wavelet_reconstruction")
        self.assertEqual(wave, {"enabled": True, "loss_weight": 0.1, "warmup_iterations": 1000})
        del baseline.train.output_dir
        del fw.train.output_dir
        self.assertEqual(OmegaConf.to_container(baseline), OmegaConf.to_container(fw))


if __name__ == "__main__":
    unittest.main()
