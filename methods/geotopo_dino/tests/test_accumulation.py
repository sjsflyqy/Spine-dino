"""Exercise the real training loop, AMP and checkpoint resume on small CPU models."""

import copy
from contextlib import ExitStack, nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
import torch
from torch import nn

from methods.geotopo_dino.train import train as trainer
from methods.geotopo_dino.train.checkpoint import GeoTopoCheckpointer
from methods.geotopo_dino.tests.test_pixel_reconstruction import (
    build_tiny_backbones, pack_heads, synthetic_batch, tiny_config,
)
from methods.geotopo_dino.models.ssl_meta_arch import GeoTopoSSLMetaArch


class ClippedLinear(nn.Linear):
    def __init__(self, value):
        super().__init__(1, 1, bias=False)
        nn.init.constant_(self.weight, value)
        self.clip_calls = 0

    def clip_grad_norm_(self, max_norm):
        self.clip_calls += 1
        return nn.utils.clip_grad_norm_(self.parameters(), max_norm)


class TinyTrainingModel(nn.Module):
    masking_signature = {"anchor_policy": "block", "random_global_policy": "block"}
    pixel_reconstruction_signature = {"enabled": False}
    wavelet_reconstruction_signature = {"enabled": False}
    gradient_reconstruction_signature = {"enabled": False}

    def __init__(self, scaler=None):
        super().__init__()
        self.student = nn.ModuleDict({"backbone": ClippedLinear(0.2)})
        self.student_aux = nn.ModuleDict({"decoder": ClippedLinear(-0.1)})
        self.register_buffer("teacher_weight", torch.tensor([[0.2]]))
        self.fp16_scaler = scaler
        self.teacher_updates = []
        self.forward_calls = []

    def get_params_groups(self):
        return [{"params": list(self.parameters()), "is_last_layer": False,
                 "lr_multiplier": 1.0, "wd_multiplier": 1.0}]

    def objective(self, x, target):
        prediction = self.student.backbone(x) + self.student_aux.decoder(x)
        return (prediction - target).square().mean()

    def forward_backward(self, data, *, iteration, teacher_temp,
                         schedule_max_iterations, loss_scale=1.0):
        loss = self.objective(data["x"], data["target"])
        scaled_loss = loss * loss_scale
        if self.fp16_scaler is None:
            scaled_loss.backward()
        else:
            self.fp16_scaler.scale(scaled_loss).backward()
        self.forward_calls.append((iteration, teacher_temp, loss_scale, loss.item()))
        return {"optimization_loss": loss.detach()}

    @torch.no_grad()
    def update_teacher(self, momentum):
        self.teacher_weight.mul_(momentum).add_(self.student.backbone.weight, alpha=1 - momentum)
        self.teacher_updates.append(momentum)


def batches(count):
    return [{"x": torch.tensor([[float(i + 1)]]),
             "target": torch.tensor([[float(i % 3)]]),
             "collated_global_crops": torch.zeros(2, 1)} for i in range(count)]


class AccumulationTests(unittest.TestCase):
    def config(self, output_dir, steps=3, total=7):
        return OmegaConf.create({
            "MODEL": {"WEIGHTS": ""},
            "train": {"output_dir": output_dir, "gradient_accumulation_steps": steps,
                      "batch_size_per_gpu": 1, "checkpoint_period": 2,
                      "checkpoint_max_to_keep": 10, "stop_after_iterations": total,
                      "dataset_path": "synthetic", "num_workers": 0,
                      "prefetch_factor": 1, "seed": 42},
            "optim": {"total_iterations": total, "clip_grad": 0.7},
            "evaluation": {"eval_period_iterations": 2},
            "student": {"patch_size": 14},
            "crops": {"global_crops_size": 28, "local_crops_size": 28,
                      "global_crops_scale": [0.5, 1.0], "local_crops_scale": [0.2, 0.5],
                      "local_crops_number": 1, "crop_ratio": [0.75, 1.33],
                      "horizontal_flip_probability": 0.5},
            "ibot": {"mask_ratio_min_max": [0.1, 0.5], "mask_sample_probability": 0.5},
        })

    def train(self, cfg, model, data, resume=False):
        optimizer = torch.optim.SGD(model.get_params_groups(), lr=0.01, momentum=0.9)
        total = cfg.optim.total_iterations
        schedules = ([0.01 * (i + 1) for i in range(total)], [0.0] * total,
                     [0.8 + 0.01 * i for i in range(total)],
                     [0.04 + 0.001 * i for i in range(total)], [0.01] * total)
        saves = []
        original_save = GeoTopoCheckpointer.save

        def record_save(checkpointer, name, **kwargs):
            saves.append((name, kwargs.copy(), len(model.teacher_updates)))
            return original_save(checkpointer, name, **kwargs)

        with ExitStack() as stack:
            for name, value in (("build_optimizer", optimizer), ("build_schedulers", schedules),
                                ("make_dataset", None), ("make_data_loader", data),
                                ("DataAugmentationGeoTopoDINO", None), ("MaskingGenerator", None)):
                stack.enter_context(patch.object(trainer, name, return_value=value))
            stack.enter_context(patch("dinov2.distributed.get_global_size", return_value=1))
            stack.enter_context(patch("dinov2.fsdp.FSDP.state_dict_type", side_effect=lambda *a: nullcontext()))
            stack.enter_context(patch.object(GeoTopoCheckpointer, "save", autospec=True, side_effect=record_save))
            stack.enter_context(patch.object(torch.cuda, "synchronize"))
            evaluate = stack.enter_context(patch.object(trainer, "do_test"))
            metrics = trainer.do_train(cfg, model, resume=resume)
        return optimizer, metrics, saves, [call.args[2] for call in evaluate.call_args_list]

    def test_gradients_match_grouped_batches_including_short_final_group(self):
        data = batches(7)
        for steps in (1, 3, 20):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as temp:
                model = TinyTrainingModel()
                reference = TinyTrainingModel()
                cfg = self.config(temp, steps)
                _, metrics, _, _ = self.train(cfg, model, data)
                optimizer = torch.optim.SGD(reference.get_params_groups(), lr=0.01, momentum=0.9)
                for start in range(0, len(data), steps):
                    group = data[start:start + steps]
                    end = start + len(group) - 1
                    optimizer.zero_grad(set_to_none=True)
                    loss = reference.objective(torch.cat([b["x"] for b in group]),
                                               torch.cat([b["target"] for b in group]))
                    loss.backward()
                    for module in (*reference.student.values(), *reference.student_aux.values()):
                        module.clip_grad_norm_(cfg.optim.clip_grad)
                    optimizer.param_groups[0]["lr"] = 0.01 * (end + 1)
                    optimizer.step()
                    reference.update_teacher(0.8 + 0.01 * end)
                for name, value in reference.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[name], value)
                self.assertEqual(len(model.teacher_updates), (7 + steps - 1) // steps)
                self.assertEqual(model.student.backbone.clip_calls, len(model.teacher_updates))
                self.assertEqual(model.student_aux.decoder.clip_calls, len(model.teacher_updates))
                self.assertEqual([call[0] for call in model.forward_calls], list(range(7)))
                self.assertAlmostEqual(metrics["total_loss"], sum(c[3] for c in model.forward_calls) / 7)
                if steps == 3:
                    self.assertEqual([c[2] for c in model.forward_calls], [1 / 3] * 6 + [1.0])

    def test_eval_and_checkpoints_defer_to_completed_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            model = TinyTrainingModel()
            _, _, saves, evaluations = self.train(self.config(temp), model, batches(7))
            self.assertEqual(evaluations, ["training_2", "training_5", "training_6"])
            self.assertEqual([s[1]["iteration"] for s in saves if s[0] != "model_final"], [2, 5])
            self.assertTrue(all(s[1]["optimizer_updates"] == s[2] for s in saves))
            checkpoint = torch.load(Path(temp) / "model_final.rank_0.pth", weights_only=True)
            self.assertEqual(checkpoint["iteration"], 6)
            self.assertEqual(checkpoint["optimizer_updates"], 3)
            self.assertEqual(checkpoint["gradient_accumulation_steps"], 3)

    def test_amp_unscales_clips_and_updates_once_per_group(self):
        scaler = Mock(wraps=torch.amp.GradScaler("cpu", init_scale=128.0, growth_interval=2))
        with tempfile.TemporaryDirectory() as temp:
            model = TinyTrainingModel(scaler)
            reference = TinyTrainingModel()
            self.train(self.config(temp), model, batches(7))
            with tempfile.TemporaryDirectory() as other:
                self.train(self.config(other), reference, batches(7))
            torch.testing.assert_close(model.student.backbone.weight, reference.student.backbone.weight)
            torch.testing.assert_close(model.student_aux.decoder.weight, reference.student_aux.decoder.weight)
            self.assertEqual(scaler.scale.call_count, 7)
            for method in (scaler.unscale_, scaler.step, scaler.update):
                self.assertEqual(method.call_count, 3)

    def test_amp_overflow_does_not_update_teacher(self):
        class OverflowScaler(torch.amp.GradScaler):
            def scale(self, loss):
                # Simulate an overflow in backward while keeping logged losses finite.
                return super().scale(loss) * float("inf")

        with tempfile.TemporaryDirectory() as temp:
            model = TinyTrainingModel(OverflowScaler("cpu", init_scale=128.0))
            before = copy.deepcopy(model.state_dict())
            self.train(self.config(temp), model, batches(7))
            self.assertEqual(model.teacher_updates, [])
            for name, value in before.items():
                torch.testing.assert_close(model.state_dict()[name], value)

    def test_resume_restores_optimizer_scaler_and_microbatch_progress(self):
        data = batches(7)
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as other:
            cfg = self.config(temp)
            cfg.train.stop_after_iterations = 3
            first = TinyTrainingModel(torch.amp.GradScaler("cpu", init_scale=128.0, growth_interval=2))
            self.train(cfg, first, data[:3])
            cfg.train.stop_after_iterations = 7
            restored = TinyTrainingModel(torch.amp.GradScaler("cpu", init_scale=16.0, growth_interval=2))
            _, _, saves, _ = self.train(cfg, restored, data[3:], resume=True)
            reference = TinyTrainingModel(torch.amp.GradScaler("cpu", init_scale=128.0, growth_interval=2))
            self.train(self.config(other), reference, data)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value)
            self.assertEqual(restored.fp16_scaler.state_dict(), reference.fp16_scaler.state_dict())
            self.assertEqual([c[0] for c in restored.forward_calls], [3, 4, 5, 6])
            self.assertEqual(saves[-1][1]["optimizer_updates"], 3)
            # Evaluation ignores the accumulation setting; resuming training must match.
            with patch("dinov2.fsdp.FSDP.state_dict_type", side_effect=lambda *a: nullcontext()):
                GeoTopoCheckpointer(TinyTrainingModel(), temp).resume_or_load("", resume=True)
                with self.assertRaisesRegex(ValueError, "accumulation checkpoint/config mismatch"):
                    GeoTopoCheckpointer(TinyTrainingModel(), temp, gradient_accumulation_steps=1).resume_or_load("", resume=True)

    def test_default_and_invalid_accumulation_settings(self):
        cfg = self.config("/tmp/unused")
        del cfg.train.gradient_accumulation_steps
        self.assertEqual(trainer.get_accumulation_steps(cfg), 1)
        for value in (0, -1, 1.5, True, "4", None):
            cfg.train.gradient_accumulation_steps = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                trainer.get_accumulation_steps(cfg)

    def test_unexpected_data_exhaustion_does_not_silently_drop_partial_gradients(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "accumulation group completed"):
                self.train(self.config(temp), TinyTrainingModel(), batches(2))
            self.assertEqual(list(Path(temp).glob("*.pth")), [])

    def test_legacy_checkpoint_resumes_with_default_one(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as other:
            cfg = self.config(temp, steps=1)
            cfg.train.stop_after_iterations = 3
            self.train(cfg, TinyTrainingModel(), batches(3))
            path = Path(temp) / "model_final.rank_0.pth"
            checkpoint = torch.load(path, weights_only=True)
            del checkpoint["gradient_accumulation_steps"], checkpoint["optimizer_updates"]
            torch.save(checkpoint, path)
            del cfg.train.gradient_accumulation_steps
            cfg.train.stop_after_iterations = 7
            restored = TinyTrainingModel()
            self.train(cfg, restored, batches(7)[3:], resume=True)
            reference = TinyTrainingModel()
            self.train(self.config(other, steps=1), reference, batches(7))
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value)

    @patch("dinov2.train.ssl_meta_arch.build_model_from_cfg", side_effect=build_tiny_backbones)
    def test_real_objective_scales_all_gradients_but_preserves_logged_losses(self, _builder):
        cfg = tiny_config(True)
        cfg.gcvd.update({"enabled": True, "loss_type": "prototype_ce", "head_n_prototypes": 8,
                         "head_bottleneck_dim": 8, "head_hidden_dim": 16, "head_nlayers": 2})
        full = GeoTopoSSLMetaArch(cfg)
        scaled = GeoTopoSSLMetaArch(cfg)
        scaled.load_state_dict(full.state_dict())
        full.need_to_synchronize_fsdp_streams = scaled.need_to_synchronize_fsdp_streams = False
        data = synthetic_batch()
        with patch.object(torch.Tensor, "cuda", lambda x, **kwargs: x), \
             patch("methods.geotopo_dino.models.ssl_meta_arch.fmha.BlockDiagonalMask.from_tensor_list", side_effect=pack_heads):
            torch.manual_seed(42)
            full_metrics = full.forward_backward(data, teacher_temp=0.07, iteration=10)
            torch.manual_seed(42)
            scaled_metrics = scaled.forward_backward(data, teacher_temp=0.07, iteration=10, loss_scale=0.25)
        for key, value in full_metrics.items():
            torch.testing.assert_close(scaled_metrics[key], value)
        for name, parameter in full.named_parameters():
            other = dict(scaled.named_parameters())[name]
            if parameter.grad is None:
                self.assertIsNone(other.grad)
            else:
                torch.testing.assert_close(other.grad, parameter.grad * 0.25)


if __name__ == "__main__":
    unittest.main()
