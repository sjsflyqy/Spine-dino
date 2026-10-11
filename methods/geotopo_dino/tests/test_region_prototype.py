"""Region prototype supervision through the actual GCVD training path."""

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
from torch.nn import functional as F

from methods.geotopo_dino.losses.gcvd_loss import GCVDPrototypeLoss
from methods.geotopo_dino.models.projection_head import GCVDPrototypeHead
from methods.geotopo_dino.models.ssl_meta_arch import GeoTopoSSLMetaArch
from methods.geotopo_dino.tests.test_pixel_reconstruction import (
    build_tiny_backbones, pack_heads, synthetic_batch, tiny_config,
)
from methods.geotopo_dino.train.checkpoint import GeoTopoCheckpointer


def region_config(teacher_centering="sinkhorn_knopp"):
    cfg = tiny_config(False)
    cfg.crops.local_crops_number = 3
    cfg.gcvd.update({
        "enabled": True, "loss_type": "prototype_ce", "head_n_prototypes": 8,
        "head_hidden_dim": 16, "head_bottleneck_dim": 8, "head_nlayers": 2,
        "teacher_centering": teacher_centering, "sinkhorn_iterations": 3,
        "warmup_type": "no_warmup", "dense_weight": 0.0,
    })
    return cfg


def region_batch():
    batch = synthetic_batch()
    batch["collated_local_crops"] = torch.randn(3, 3, 28, 28)
    batch["local_transforms"] = torch.eye(3).repeat(3, 1, 1)
    # First region: four valid patches; second: two; third: none.
    batch["local_transforms"][1, 0, 2] = 14
    batch["local_transforms"][2, 0, 2] = 100
    batch["local_sample_ids"] = torch.zeros(3, dtype=torch.long)
    return batch


def distributed_region_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        results = []
        for counts in ((1, 2), (0, 2), (0, 0)):
            torch.manual_seed(42)
            with patch("dinov2.train.ssl_meta_arch.build_model_from_cfg", side_effect=build_tiny_backbones):
                model = GeoTopoSSLMetaArch(region_config())
            model.need_to_synchronize_fsdp_streams = False
            model.teacher.load_state_dict(model.student.state_dict())
            batch = region_batch()
            batch["local_transforms"][counts[rank]:, 0, 2] = 100
            seen = {}

            def teacher_hook(module, inputs, output):
                seen["teacher_logits"] = output.detach().clone()

            def loss_hook(module, inputs):
                seen["student_logits"] = inputs[0].detach().clone()
                seen["probabilities"] = inputs[1].detach().clone()

            model.teacher.geom_head.register_forward_hook(teacher_hook)
            model.gcvd_region_prototype_loss.register_forward_pre_hook(loss_hook)
            with patch.object(torch.Tensor, "cuda", lambda x, **kwargs: x), \
                 patch("methods.geotopo_dino.models.ssl_meta_arch.fmha.BlockDiagonalMask.from_tensor_list",
                       side_effect=pack_heads):
                output = model.forward_backward(batch, teacher_temp=0.07, iteration=10)
            seen["loss"] = output["gcvd_region_raw_loss"]
            gradients = [p.grad for p in model.student.geom_head.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert all(p.grad is None for p in model.teacher.parameters())
            if counts[rank] == 0:
                assert all(g.count_nonzero() == 0 for g in gradients)
            results.append(seen)
        torch.save(results, Path(directory) / f"rank{rank}.pth")
    finally:
        dist.destroy_process_group()


class DistributedRegionPrototypeTests(unittest.TestCase):
    def test_two_rank_region_balancing_with_unequal_and_empty_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_region_worker,
                     args=(str(Path(directory) / "rendezvous"), directory), nprocs=2, join=True)
            results = [torch.load(Path(directory) / f"rank{rank}.pth", weights_only=True) for rank in range(2)]
        reference = GCVDPrototypeLoss(8, teacher_centering="sinkhorn_knopp")
        for case in range(3):
            teacher_logits = torch.cat([rank[case]["teacher_logits"] for rank in results])
            expected = reference.sinkhorn_knopp_teacher(teacher_logits)
            offset = 0
            for rank in results:
                seen = rank[case]
                count = seen["probabilities"].shape[0]
                torch.testing.assert_close(seen["probabilities"], expected[offset:offset + count])
                offset += count
                if count:
                    ce = -(seen["probabilities"] * F.log_softmax(seen["student_logits"] / 0.1, dim=-1)).sum(-1).mean()
                    torch.testing.assert_close(seen["loss"], ce)
                else:
                    self.assertEqual(seen["loss"].item(), 0)


class RegionPrototypeTests(unittest.TestCase):
    def setUp(self):
        builder = patch("dinov2.train.ssl_meta_arch.build_model_from_cfg", side_effect=build_tiny_backbones)
        builder.start()
        self.addCleanup(builder.stop)

    def model(self, teacher_centering="sinkhorn_knopp"):
        model = GeoTopoSSLMetaArch(region_config(teacher_centering))
        # Retain standard SSL heads/forward, isolate the region contribution.
        model.dino_loss_weight = 0.0
        model.ibot_loss_weight = 0.0
        model.need_to_synchronize_fsdp_streams = False
        model.teacher.load_state_dict(model.student.state_dict())
        return model

    def forward(self, model, batch):
        with patch.object(torch.Tensor, "cuda", lambda x, **kwargs: x), \
             patch("methods.geotopo_dino.models.ssl_meta_arch.fmha.BlockDiagonalMask.from_tensor_list",
                   side_effect=pack_heads):
            return model.forward_backward(batch, teacher_temp=0.07, iteration=10)

    def test_region_uses_sinkhorn_ce_valid_pooling_and_backbone_gradient(self):
        model = self.model()
        batch = region_batch()
        seen = {}

        def capture(name):
            def hook(module, inputs, output):
                seen[name] = (inputs[0].detach().clone(), output.detach().clone())
            return hook

        handles = [
            model.teacher.geom_head.register_forward_hook(capture("teacher")),
            model.student.geom_head.register_forward_hook(capture("student")),
        ]
        with torch.no_grad():
            teacher_tokens = model.teacher.backbone(batch["collated_global_crops"])["x_norm_patchtokens"][:1]
            student_tokens = model.student.backbone(batch["collated_local_crops"])["x_norm_patchtokens"]
        result = self.forward(model, batch)
        for handle in handles:
            handle.remove()

        self.assertIsInstance(model.student.geom_head, GCVDPrototypeHead)
        self.assertIsNot(model.student.geom_head, model.student.gcvd_proto_head)
        torch.testing.assert_close(seen["teacher"][0], torch.stack([
            teacher_tokens[0].mean(0), teacher_tokens[0, [0, 2]].mean(0),
        ]))
        torch.testing.assert_close(seen["student"][0], torch.stack([
            student_tokens[0].mean(0), student_tokens[1, [1, 3]].mean(0),
        ]))
        probabilities = model.gcvd_region_prototype_loss.sinkhorn_knopp_teacher(seen["teacher"][1])
        reference_ce = -(probabilities * F.log_softmax(
            seen["student"][1] / model.cfg.gcvd.student_temp, dim=-1,
        )).sum(-1).mean()
        torch.testing.assert_close(result["gcvd_region_raw_loss"], reference_ce)
        torch.testing.assert_close(result["gcvd_region_weighted_loss"], reference_ce * 0.5)
        torch.testing.assert_close(result["optimization_loss"], result["gcvd_region_weighted_loss"])
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean()
        torch.testing.assert_close(result["gcvd_region_raw_kl"], reference_ce - entropy)
        self.assertGreater(model.student.backbone.mix.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.student.geom_head.last_layer.weight_v.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        torch.testing.assert_close(model.gcvd_region_prototype_loss.center, torch.zeros(1, 8))

    def test_all_invalid_regions_have_zero_loss_and_finite_zero_gradients(self):
        model = self.model()
        batch = region_batch()
        batch["local_transforms"][:, 0, 2] = 100
        result = self.forward(model, batch)
        for key in ("gcvd_dense_raw_loss", "gcvd_region_raw_loss", "gcvd_region_raw_kl",
                    "gcvd_region_teacher_entropy", "gcvd_region_student_entropy"):
            self.assertEqual(result[key].item(), 0.0)
        for parameter in model.student.geom_head.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(parameter.grad.count_nonzero().item(), 0)

    def test_region_center_is_independent_and_checkpoint_roundtrip_restores_training(self):
        model = self.model("centering")
        optimizer = torch.optim.AdamW(model.get_params_groups(), lr=1e-3)
        self.forward(model, region_batch())
        optimizer.step()
        self.assertGreater(model.gcvd_region_prototype_loss.center.abs().sum().item(), 0)
        self.assertFalse(torch.equal(model.gcvd_region_prototype_loss.center, model.gcvd_prototype_loss.center))
        restored = self.model("centering")
        restored_optimizer = torch.optim.AdamW(restored.get_params_groups(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory, \
             patch("dinov2.fsdp.FSDP.state_dict_type", side_effect=lambda *args: nullcontext()):
            checkpointer = GeoTopoCheckpointer(model, directory, optimizer=optimizer)
            checkpointer.save("model", iteration=10)
            GeoTopoCheckpointer(restored, directory, optimizer=restored_optimizer).resume_or_load("", resume=True)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])
        batch = region_batch()
        optimizer.zero_grad()
        restored_optimizer.zero_grad()
        left, right = self.forward(model, copy.deepcopy(batch)), self.forward(restored, batch)
        torch.testing.assert_close(left["optimization_loss"], right["optimization_loss"])
        torch.testing.assert_close(model.student.geom_head.last_layer.weight_v.grad,
                                   restored.student.geom_head.last_layer.weight_v.grad)


if __name__ == "__main__":
    unittest.main()
