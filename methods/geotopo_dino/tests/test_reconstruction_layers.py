"""CPU checks for masked intermediate features and reconstruction gradients."""

import copy
import unittest
from unittest.mock import patch

import torch

from dinov2.layers.block import Block
from dinov2.models.vision_transformer import DinoVisionTransformer
from methods.geotopo_dino.losses.pixel_reconstruction_loss import PixelReconstructionLoss
from methods.geotopo_dino.models.pixel_decoder import PixelDecoder
from methods.geotopo_dino.models.ssl_meta_arch import GeoTopoSSLMetaArch
from methods.geotopo_dino.tests.test_pixel_reconstruction import pack_heads, synthetic_batch, tiny_config
from methods.geotopo_dino.train.checkpoint import validate_pixel_checkpoint


class CPUListBlock(Block):
    """Use native PyTorch attention for the encoder's global/local list on CPU."""

    def forward(self, x):
        if isinstance(x, list):
            return [super(CPUListBlock, self).forward(tokens) for tokens in x]
        return super().forward(x)


def backbone(chunks=0, registers=0):
    return DinoVisionTransformer(
        img_size=28, patch_size=14, embed_dim=12, depth=12, num_heads=3,
        block_fn=CPUListBlock, block_chunks=chunks, num_register_tokens=registers,
    )


def build_cpu_backbones(cfg):
    student = backbone(chunks=cfg.student.block_chunks)
    return student, copy.deepcopy(student), 12


def make_model(layer=None, chunks=0, gcvd=False):
    cfg = tiny_config(True)
    cfg.student.block_chunks = chunks
    if gcvd:
        cfg.gcvd.update({
            "enabled": True, "loss_type": "prototype_ce", "head_n_prototypes": 8,
            "head_hidden_dim": 16, "head_bottleneck_dim": 8, "head_nlayers": 2,
        })
    if layer is not None:
        cfg.pixel_reconstruction.feature_layer = layer
    with patch("dinov2.train.ssl_meta_arch.build_model_from_cfg", side_effect=build_cpu_backbones):
        model = GeoTopoSSLMetaArch(cfg)
    model.teacher.load_state_dict(model.student.state_dict())
    model.need_to_synchronize_fsdp_streams = False
    return model


def train_batch(model, batch):
    with patch.object(torch.Tensor, "cuda", lambda x, **kwargs: x), \
         patch("methods.geotopo_dino.models.ssl_meta_arch.fmha.BlockDiagonalMask.from_tensor_list",
               side_effect=pack_heads):
        return model.forward_backward(batch, teacher_temp=0.07, iteration=10)


class IntermediateFeatureTests(unittest.TestCase):
    def test_layer_numbers_masks_registers_and_single_forward(self):
        for chunks in (0, 3):
            model = backbone(chunks=chunks, registers=2)
            images = torch.randn(2, 3, 28, 28)
            masks = synthetic_batch()["collated_masks"]
            for layer in (1, 4, 8, 12):
                with self.subTest(chunks=chunks, layer=layer):
                    seen, counts, handles = {}, {}, []
                    blocks = (list(model.blocks) if not chunks else
                              [b for chunk in model.blocks for b in chunk if not isinstance(b, torch.nn.Identity)])
                    for index, block in enumerate(blocks):
                        def capture(_module, _args, output, index=index):
                            counts[index] = counts.get(index, 0) + 1
                            if index + 1 == layer:
                                seen["tokens"] = output
                        handles.append(block.register_forward_hook(capture))
                    try:
                        result = model(images, masks=masks, is_training=True, return_patch_layer=layer)
                    finally:
                        for handle in handles:
                            handle.remove()
                    self.assertEqual(counts, dict.fromkeys(range(12), 1))
                    self.assertEqual(result["x_norm_patchtokens_intermediate"].shape, (2, 4, 12))
                    torch.testing.assert_close(result["x_norm_patchtokens_intermediate"],
                                               model.norm(seen["tokens"])[:, 3:], rtol=0, atol=0)
                    # Hidden input pixels must never leak into the intermediate branch.
                    hidden = masks.reshape(2, 2, 2).repeat_interleave(14, 1).repeat_interleave(14, 2)
                    changed_images = images.masked_fill(hidden[:, None], 1000)
                    changed = model(changed_images, masks=masks, is_training=True, return_patch_layer=layer)
                    torch.testing.assert_close(result["x_norm_patchtokens_intermediate"],
                                               changed["x_norm_patchtokens_intermediate"], rtol=0, atol=0)

    def test_list_forward_preserves_final_features_and_rng(self):
        for chunks in (0, 3):
            model = backbone(chunks=chunks)
            batch = synthetic_batch()
            inputs = [batch["collated_global_crops"], batch["collated_local_crops"]]
            masks = [batch["collated_masks"], None]
            rng = torch.get_rng_state()
            expected = model(inputs, masks=masks, is_training=True)
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            actual = model(inputs, masks=masks, is_training=True, return_patch_layer=8)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            for old, new in zip(expected, actual):
                for key in ("x_norm_patchtokens", "x_norm_clstoken", "x_prenorm"):
                    torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
            # List-mode global tokens agree with the single-tensor masked path.
            single = model(inputs[0], masks=masks[0], is_training=True, return_patch_layer=8)
            torch.testing.assert_close(actual[0]["x_norm_patchtokens_intermediate"],
                                       single["x_norm_patchtokens_intermediate"], rtol=0, atol=0)

    def test_pixel_gradients_stop_at_selected_block(self):
        for chunks in (0, 3):
            model = backbone(chunks=chunks)
            batch = synthetic_batch()
            output = model([batch["collated_global_crops"], batch["collated_local_crops"]],
                           masks=[batch["collated_masks"], None], is_training=True, return_patch_layer=8)
            decoder = PixelDecoder(12, (2, 2), 14, decoder_dim=16, decoder_depth=1, decoder_num_heads=4)
            loss = PixelReconstructionLoss(14)(decoder(output[0]["x_norm_patchtokens_intermediate"]),
                                               batch["collated_global_crops"], batch["collated_masks"])
            loss.backward()
            blocks = (list(model.blocks) if not chunks else
                      [b for chunk in model.blocks for b in chunk if not isinstance(b, torch.nn.Identity)])
            for index, block in enumerate(blocks):
                grads = [p.grad for p in block.parameters()]
                if index < 8:
                    self.assertTrue(all(g is not None for g in grads))
                    self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
                else:
                    self.assertTrue(all(g is None for g in grads))
            self.assertIsNotNone(model.norm.weight.grad)
            self.assertTrue(all(p.grad is not None for p in decoder.parameters()))


class ReconstructionLayerIntegrationTests(unittest.TestCase):
    def test_last_layer_defaults_are_exactly_compatible(self):
        models = []
        for layer in (None, -1, 12):
            torch.manual_seed(123)
            models.append(make_model(layer))
        batch = synthetic_batch()
        results = [train_batch(model, batch) for model in models]
        for model, result in zip(models[1:], results[1:]):
            self.assertFalse(model.pixel_use_intermediate)
            self.assertEqual(model.pixel_reconstruction_signature, models[0].pixel_reconstruction_signature)
            for key in results[0]:
                torch.testing.assert_close(result[key], results[0][key], rtol=0, atol=0)
            for key, value in models[0].state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
            for old, new in zip(models[0].parameters(), model.parameters()):
                if old.grad is not None:
                    torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)

    def test_decoder_consumes_selected_global_features_and_other_losses_unchanged(self):
        for chunks in (0, 3):
            torch.manual_seed(123)
            original = make_model(chunks=chunks, gcvd=True)
            torch.manual_seed(123)
            selected = make_model(8, chunks=chunks, gcvd=True)
            seen = {}
            def capture_backbone(_module, _args, output):
                seen["global"] = output[0]["x_norm_patchtokens_intermediate"].detach().clone()
                seen["calls"] = seen.get("calls", 0) + 1
            def capture_decoder(_module, args):
                seen["decoder"] = args[0].detach().clone()
            hooks = [selected.student.backbone.register_forward_hook(capture_backbone),
                     selected.student_aux.pixel_decoder.register_forward_pre_hook(capture_decoder)]
            try:
                batch = synthetic_batch()
                expected = train_batch(original, batch)
                result = train_batch(selected, batch)
            finally:
                for hook in hooks:
                    hook.remove()
            self.assertEqual(seen["calls"], 1)
            torch.testing.assert_close(seen["global"], seen["decoder"], rtol=0, atol=0)
            for key in ("dino_local_crops_loss", "dino_global_crops_loss", "ibot_loss",
                        "gcvd_dense_raw_loss", "gcvd_region_raw_loss"):
                torch.testing.assert_close(expected[key], result[key], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(result["pixel_raw_loss"]))
            self.assertTrue(all(p.grad is None for p in selected.teacher.parameters()))

    def test_checkpoint_rejects_cross_layer_resume_and_accepts_legacy_final(self):
        legacy = make_model().pixel_reconstruction_signature
        final = make_model(12).pixel_reconstruction_signature
        layer8 = make_model(8).pixel_reconstruction_signature
        layer4 = make_model(4).pixel_reconstruction_signature
        validate_pixel_checkpoint({"pixel_reconstruction_signature": legacy}, final)
        validate_pixel_checkpoint({"pixel_reconstruction_signature": layer8}, layer8)
        for saved, requested in ((legacy, layer8), (layer8, final), (layer8, layer4)):
            with self.assertRaisesRegex(ValueError, "feature_layer"):
                validate_pixel_checkpoint({"pixel_reconstruction_signature": saved}, requested)

    def test_invalid_layers_fail_early(self):
        for layer in (0, -2, 13, True, 8.0, "8", [8]):
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "feature_layer"):
                make_model(layer)


if __name__ == "__main__":
    unittest.main()
