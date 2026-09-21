import copy
from contextlib import nullcontext
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch import nn

from dinov2.configs import dinov2_default_config
from methods.geotopo_dino.losses.pixel_reconstruction_loss import (
    PixelReconstructionLoss, global_reconstruction_mask, patchify, pixel_warmup_scale, unpatchify,
)
from methods.geotopo_dino.models.pixel_decoder import PixelDecoder, sincos_position_embedding
from methods.geotopo_dino.models.ssl_meta_arch import GeoTopoSSLMetaArch
from methods.geotopo_dino.train.checkpoint import GeoTopoCheckpointer, validate_pixel_checkpoint


def tiny_config(pixel=None):
    cfg = OmegaConf.merge(
        OmegaConf.create(dinov2_default_config),
        OmegaConf.load(Path(__file__).parents[1] / "configs/gcvd_mvp.yaml"),
        OmegaConf.create({
            "compute_precision": {"grad_scaler": False},
            "student": {"arch": "vit_small", "pretrained_weights": ""},
            "crops": {"global_crops_size": 28, "local_crops_size": 28, "local_crops_number": 1},
            "gcvd": {"enabled": False, "anchor_size": 28},
            "dino": {"head_n_prototypes": 8, "head_hidden_dim": 16, "head_bottleneck_dim": 8,
                     "head_nlayers": 2, "koleo_loss_weight": 0.0},
        }),
    )
    del cfg.pixel_reconstruction
    if pixel is not None:
        cfg.pixel_reconstruction = {
            "enabled": pixel, "decoder_dim": 16, "decoder_depth": 1,
            "decoder_num_heads": 4, "warmup_iterations": 0, "loss_weight": 0.1,
        }
    return cfg


class TinyBackbone(nn.Module):
    """CPU stand-in for the CUDA/xFormers encoder; retains contextual masking."""
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 12, 14, stride=14)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 12))
        self.mix = nn.Linear(12, 12)

    def forward(self, images, masks=None, is_training=False):
        if isinstance(images, list):
            return [self.forward(x, m, is_training) for x, m in zip(images, masks)]
        x = self.patch_embed.proj(images).flatten(2).transpose(1, 2)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token, x)
        x = self.mix(x + x.mean(1, keepdim=True))
        return {"x_norm_patchtokens": x, "x_norm_clstoken": x.mean(1)}


def build_tiny_backbones(cfg):
    student = TinyBackbone()
    return student, copy.deepcopy(student), 12


class PackedHeads:
    """Head packing needs no attention; replace CUDA bias metadata on CPU."""
    def __init__(self, tensors):
        self.lengths = [tensor.shape[1] for tensor in tensors]

    def split(self, tensor):
        return list(tensor.split(self.lengths, dim=1))


def pack_heads(tensors):
    return PackedHeads(tensors), torch.cat(tensors, dim=1)


def synthetic_batch(device="cpu", empty_masks=False):
    masks = torch.tensor([[True, False, True, False], [False, True, False, True]], device=device)
    if empty_masks:
        masks.zero_()
    return {
        "collated_global_crops": torch.randn(2, 3, 28, 28, device=device),
        "collated_local_crops": torch.randn(1, 3, 28, 28, device=device),
        "collated_masks": masks,
        "anchor_transforms": torch.eye(3, device=device)[None],
        "anchor_valid_masks": torch.ones(1, 2, 2, dtype=torch.bool, device=device),
        "local_transforms": torch.eye(3, device=device)[None],
        "local_sample_ids": torch.zeros(1, dtype=torch.long, device=device),
        "upperbound": 8,
    }


class PixelLossTests(unittest.TestCase):
    def test_patch_order_channels_rectangular_and_roundtrip(self):
        images = torch.arange(2 * 3 * 4 * 6).reshape(2, 3, 4, 6).float()
        patches = patchify(images, 2)
        for batch in range(2):
            for row in range(2):
                for col in range(3):
                    expected = images[batch, :, row*2:row*2+2, col*2:col*2+2].permute(1, 2, 0).flatten()
                    torch.testing.assert_close(patches[batch, row*3+col], expected)
        torch.testing.assert_close(unpatchify(patches, 2, (2, 3)), images)

    def test_loss_only_uses_mask_and_valid_pixels(self):
        images = torch.randn(4, 3, 4, 4)
        candidates = torch.ones(4, 4, dtype=torch.bool)
        valid = torch.tensor([[[False, True], [True, False]], [[True, True], [False, False]]])
        masks = global_reconstruction_mask(candidates, valid)
        torch.testing.assert_close(masks[:2], valid.flatten(1))
        self.assertTrue(masks[2:].all())
        masks[3, 1] = False
        prediction = patchify(images, 2).clone()
        prediction[~masks] += 100
        prediction.requires_grad_()
        criterion = PixelReconstructionLoss(2)
        self.assertEqual(criterion(prediction, images, masks).item(), 0.0)
        altered = prediction + masks.unsqueeze(-1) * 2
        loss = criterion(altered, images, masks)
        self.assertAlmostEqual(loss.item(), 4.0)
        loss.backward()
        self.assertEqual(prediction.grad[~masks].abs().sum().item(), 0.0)
        self.assertGreater(prediction.grad[masks].abs().sum().item(), 0.0)

    def test_global_views_are_not_swapped(self):
        images = torch.stack([torch.full((3, 4, 4), v) for v in (1., 2., 10., 20.)])
        masks = torch.ones(4, 4, dtype=torch.bool)
        criterion = PixelReconstructionLoss(2)
        target = patchify(images, 2)
        self.assertEqual(criterion(target, images, masks).item(), 0.0)
        self.assertGreater(criterion(target.roll(2, 0), images, masks).item(), 0.0)

    def test_empty_mask_preserves_zero_gradient_graph(self):
        images = torch.randn(2, 3, 4, 4, requires_grad=True)
        prediction = torch.randn(2, 4, 12, requires_grad=True)
        loss = PixelReconstructionLoss(2)(prediction, images, torch.zeros(2, 4, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertEqual(prediction.grad.abs().sum().item(), 0.0)
        self.assertIsNone(images.grad)

    def test_pixel_normalization_and_fp32(self):
        images = torch.randn(2, 3, 4, 4).half()
        target = patchify(images.float(), 2)
        target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6).sqrt()
        loss = PixelReconstructionLoss(2, True)(target, images, torch.ones(2, 4, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(loss.dtype, torch.float32)

    def test_distributed_count_scaling(self):
        images = torch.zeros(1, 1, 2, 2)
        prediction = torch.full((1, 4, 1), 2., requires_grad=True)
        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.all_reduce", side_effect=lambda count: count.fill_(4)):
            loss = PixelReconstructionLoss(1)(prediction, images, torch.tensor([[True, False, False, False]]))
        self.assertEqual(loss.item(), 2.0)  # 4 local error * 2 ranks / 4 global patches
        loss.backward()
        torch.testing.assert_close(prediction.grad.flatten(), torch.tensor([2., 0., 0., 0.]))

    def test_warmup_uses_restored_iteration(self):
        self.assertEqual(pixel_warmup_scale(0, 1000), 0)
        self.assertEqual(pixel_warmup_scale(500, 1000), 0.5)
        self.assertEqual(pixel_warmup_scale(3000, 1000), 1)
        self.assertEqual(pixel_warmup_scale(0, 0), 1)
        with self.assertRaises(ValueError):
            pixel_warmup_scale(0, -1)


class PixelDecoderTests(unittest.TestCase):
    def test_shape_backward_and_checkpoint(self):
        decoder = PixelDecoder(12, (2, 3), 2, decoder_dim=16, decoder_depth=1, decoder_num_heads=4)
        features = torch.randn(2, 6, 12, requires_grad=True)
        prediction = decoder(features)
        self.assertEqual(prediction.shape, (2, 6, 12))
        prediction.square().mean().backward()
        self.assertGreater(features.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None for p in decoder.parameters()))
        restored = copy.deepcopy(decoder)
        restored.load_state_dict(decoder.state_dict())
        torch.testing.assert_close(restored(features), prediction)
        self.assertFalse(decoder.pos_embed.requires_grad)
        positions = sincos_position_embedding(2, 3, 16)
        self.assertFalse(torch.equal(positions[:, 0], positions[:, 1]))
        self.assertFalse(torch.equal(positions[:, 0], positions[:, 3]))

    def test_actual_518_grid(self):
        decoder = PixelDecoder(12, (37, 37), 14, decoder_dim=16, decoder_depth=1, decoder_num_heads=4)
        with torch.no_grad():
            prediction = decoder(torch.randn(1, 1369, 12))
        self.assertEqual(prediction.shape, (1, 1369, 588))


class PixelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.builder = patch("dinov2.train.ssl_meta_arch.build_model_from_cfg", side_effect=build_tiny_backbones)
        self.builder.start()
        self.addCleanup(self.builder.stop)

    def model(self, enabled):
        model = GeoTopoSSLMetaArch(tiny_config(enabled))
        model.need_to_synchronize_fsdp_streams = False
        model.teacher.load_state_dict(model.student.state_dict())
        return model

    def forward(self, model, batch):
        with patch.object(torch.Tensor, "cuda", lambda x, **kwargs: x), \
             patch("methods.geotopo_dino.models.ssl_meta_arch.fmha.BlockDiagonalMask.from_tensor_list", side_effect=pack_heads):
            return model.forward_backward(batch, teacher_temp=0.07, iteration=10)

    def test_missing_and_disabled_configs_match_parameters_rng_and_forward(self):
        torch.manual_seed(123)
        legacy = self.model(None)
        legacy_rng = torch.get_rng_state()
        torch.manual_seed(123)
        disabled = self.model(False)
        self.assertTrue(torch.equal(legacy_rng, torch.get_rng_state()))
        self.assertEqual(len(disabled.student_aux), 0)
        self.assertEqual(legacy.state_dict().keys(), disabled.state_dict().keys())
        for key, value in legacy.state_dict().items():
            torch.testing.assert_close(value, disabled.state_dict()[key], rtol=0, atol=0)
        batch = synthetic_batch()
        left, right = self.forward(legacy, batch), self.forward(disabled, batch)
        self.assertFalse(any(key.startswith("pixel_") for key in right))
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        for a, b in zip(legacy.student.parameters(), disabled.student.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_enabled_branch_optimizer_teacher_and_gradients(self):
        torch.manual_seed(123)
        baseline = self.model(False)
        torch.manual_seed(123)
        enabled = self.model(True)
        batch = synthetic_batch()
        raw_mask = batch["collated_masks"].clone()
        baseline_result = self.forward(baseline, batch)
        seen = {}
        def capture(_module, args):
            seen["images"] = args[1]
            seen["masks"] = args[2].clone()
        hook = enabled.pixel_loss.register_forward_pre_hook(capture)
        result = self.forward(enabled, batch)
        hook.remove()
        self.assertIs(seen["images"], batch["collated_global_crops"])
        torch.testing.assert_close(seen["masks"], raw_mask)
        torch.testing.assert_close(batch["collated_masks"], raw_mask)
        self.assertGreater(result["pixel_raw_loss"].item(), 0)
        torch.testing.assert_close(result["pixel_weighted_loss"], result["pixel_raw_loss"] * 0.1)
        torch.testing.assert_close(result["optimization_loss"], baseline_result["optimization_loss"] + result["pixel_weighted_loss"])
        self.assertFalse(torch.equal(baseline.student.backbone.mix.weight.grad, enabled.student.backbone.mix.weight.grad))
        self.assertTrue(all(p.grad is not None for p in enabled.student_aux.parameters()))
        self.assertTrue(all(p.grad is None for p in enabled.teacher.parameters()))
        self.assertFalse(any("pixel" in name for name, _ in enabled.teacher.named_parameters()))
        groups = enabled.get_params_groups()
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(set(map(id, enabled.student_aux.parameters())).issubset(set(ids)))
        optimizer = torch.optim.AdamW(groups, lr=1e-3)
        before = enabled.student_aux.pixel_decoder.pred.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, enabled.student_aux.pixel_decoder.pred.weight))

    def test_checkpoint_signature_and_state_roundtrip(self):
        model = self.model(True)
        optimizer = torch.optim.AdamW(model.get_params_groups(), lr=1e-3)
        self.forward(model, synthetic_batch())
        optimizer.step()
        restored = self.model(True)
        restored_optimizer = torch.optim.AdamW(restored.get_params_groups(), lr=1e-3)
        # Exercise the real checkpointer orchestration, replacing only the CUDA
        # FSDP state-dict context for these unwrapped CPU modules.
        with tempfile.TemporaryDirectory() as temp, \
             patch("dinov2.fsdp.FSDP.state_dict_type", side_effect=lambda *args: nullcontext()):
            checkpointer = GeoTopoCheckpointer(model, temp, optimizer=optimizer)
            checkpointer.save("model", iteration=10)
            loaded = torch.load(checkpointer.get_checkpoint_file(), weights_only=True)
            result = GeoTopoCheckpointer(restored, temp, optimizer=restored_optimizer).resume_or_load("", resume=True)
            self.assertEqual(result["iteration"], 10)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                GeoTopoCheckpointer(self.model(False), temp).resume_or_load("", resume=True)
        self.assertEqual(len(optimizer.state), len(restored_optimizer.state))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])
        validate_pixel_checkpoint({"model": {}}, {"enabled": False})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_pixel_checkpoint(loaded, {"enabled": False})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_pixel_checkpoint({"model": {}}, model.pixel_reconstruction_signature)

    def test_gcvd_and_pixel_can_run_together(self):
        cfg = tiny_config(True)
        cfg.gcvd.enabled = True
        cfg.gcvd.projection_hidden_dim = 16
        cfg.gcvd.projection_dim = 8
        model = GeoTopoSSLMetaArch(cfg)
        model.need_to_synchronize_fsdp_streams = False
        result = self.forward(model, synthetic_batch())
        self.assertIn("pixel_raw_loss", result)
        self.assertIn("gcvd_dense_raw_loss", result)
        self.assertTrue(torch.isfinite(result["optimization_loss"]))

    def test_auxiliary_fsdp_wrapper_is_only_used_when_enabled(self):
        for enabled in (False, True):
            model = self.model(enabled)
            with patch("dinov2.train.ssl_meta_arch.SSLMetaArch.prepare_for_distributed_training"), \
                 patch("methods.geotopo_dino.models.ssl_meta_arch.get_fsdp_wrapper", return_value=lambda module: module) as wrapper:
                model.prepare_for_distributed_training()
            self.assertEqual(wrapper.call_count, int(enabled))

    def test_zero_warmup_weight_keeps_decoder_in_graph(self):
        model = self.model(True)
        model.pixel_warmup_iterations = 1000
        with patch.object(torch.Tensor, "cuda", lambda x, **kwargs: x), \
             patch("methods.geotopo_dino.models.ssl_meta_arch.fmha.BlockDiagonalMask.from_tensor_list", side_effect=pack_heads):
            result = model.forward_backward(synthetic_batch(), teacher_temp=0.07, iteration=0)
        self.assertEqual(result["pixel_weighted_loss"].item(), 0)
        self.assertTrue(all(p.grad is not None for p in model.student_aux.parameters()))
        self.assertTrue(all(p.grad.count_nonzero().item() == 0 for p in model.student_aux.parameters()))

    def test_visualization(self):
        from methods.geotopo_dino.tools.visualize_reconstruction import save_reconstruction_grid
        from PIL import Image
        images = torch.randn(2, 3, 28, 28)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "reconstruction.png"
            save_reconstruction_grid(images, patchify(images, 14), torch.ones(2, 4, dtype=torch.bool),
                                     patch_size=14, output_path=path)
            with Image.open(path) as image:
                self.assertEqual(image.size, (245 * 5, 60 * 2))


if __name__ == "__main__":
    unittest.main()
