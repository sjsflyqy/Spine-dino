"""Compatibility, frozen-feature semantics and end-to-end task-07 probe checks."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[1]
sys.path[:0] = [str(TASK), str(ROOT / "Downstream"), str(ROOT)]

import model as legacy
import probe_model
import train_probe
from keypoint_linear_common import deterministic_split, gaussian_heatmap_loss
from keypoint_linear_common import build_downstream_optimizer, trainable_state_dict
from backbone.lora import inject_lora


TINY_ADAPTER = {"enabled": True, "type": "lora", "rank": 2, "alpha": 4, "dropout": 0,
                "target_modules": ["query"]}


class TinyHFEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 8, 2, stride=2)
        self.query = nn.Linear(8, 8)
        self.layernorm = nn.LayerNorm(8)
        self.config = SimpleNamespace(num_hidden_layers=12)

    def forward(self, pixel_values, **kwargs):
        tokens = self.query(self.projection(pixel_values).flatten(2).transpose(1, 2))
        # Include CLS and two register tokens to check patch-token selection.
        tokens = torch.cat((tokens.new_zeros(tokens.shape[0], 3, 8), tokens), dim=1)
        states = tuple(tokens * (index + 1) + index for index in range(13))
        return SimpleNamespace(hidden_states=states, last_hidden_state=self.layernorm(states[-1]))


class TinyNativeEncoder(TinyHFEncoder):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])

    def get_intermediate_layers(self, images, *, n, reshape):
        states = self(images).hidden_states
        return tuple(
            self.layernorm(states[index + 1])[:, 3:].transpose(1, 2).reshape(
                images.shape[0], 8, images.shape[-2] // 2, images.shape[-1] // 2
            )
            for index in n
        )


class TinyBackbone(nn.Module):
    embedding_dim = 8
    patch_size = 2
    adapter_enabled = False

    def __init__(self, native=True):
        super().__init__()
        self.model = (TinyNativeEncoder() if native else TinyHFEncoder()).requires_grad_(False)

    def forward(self, images):
        tokens = self.model(images).last_hidden_state[:, 3:]
        return SimpleNamespace(feature_map=tokens.transpose(1, 2).reshape(
            images.shape[0], 8, images.shape[-2] // 2, images.shape[-1] // 2
        ))


class TinyPairedDataset(Dataset):
    def __init__(self, root, identifiers, clean, height, width, train):
        self.identifiers = identifiers
        self.height, self.width = height, width

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(int(self.identifiers[index]))
        return {
            "id": self.identifiers[index],
            "ap_image": torch.rand(3, self.height, self.width, generator=generator),
            "lat_image": torch.rand(3, self.height, self.width, generator=generator),
            "ap_points": torch.rand(68, 2, generator=generator),
            "lat_points": torch.rand(68, 2, generator=generator),
        }


def tiny_backbone(*args, adapter=None, **kwargs):
    backbone = TinyBackbone()
    if adapter and adapter.get("enabled", False):
        backbone.adapter_modules = inject_lora(
            backbone.model, rank=adapter.get("rank", 8), alpha=adapter.get("alpha", 16),
            dropout=adapter.get("dropout", 0.05), target_modules=adapter.get("target_modules", ["query"]),
        )
        backbone.adapter_enabled = True
    return backbone


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.legacy_patch = patch.object(legacy, "build_backbone", side_effect=tiny_backbone)
        self.probe_patch = patch.object(probe_model, "build_backbone", side_effect=tiny_backbone)
        self.legacy_patch.start()
        self.probe_patch.start()
        self.addCleanup(self.legacy_patch.stop)
        self.addCleanup(self.probe_patch.stop)
        self.images = torch.randn(2, 3, 16, 8)

    def build(self, kind, adapter=None, **settings):
        return probe_model.build_probe({"backbone": "tiny", "probe": {"type": kind, **settings}, "adapter": adapter})

    def test_legacy_exact_initialization_outputs_and_bidirectional_checkpoint_loading(self):
        torch.manual_seed(17)
        original = legacy.PairedKeypointLinearProbe("tiny", None)
        torch.manual_seed(17)
        selected = probe_model.build_probe({"backbone": "tiny"})
        self.assertIs(type(selected), legacy.PairedKeypointLinearProbe)
        self.assertEqual(set(original.state_dict()), set(selected.state_dict()))
        for name, expected in original.state_dict().items():
            torch.testing.assert_close(selected.state_dict()[name], expected, rtol=0, atol=0)
        for view, expected in original(self.images, self.images.flip(-1)).items():
            torch.testing.assert_close(selected(self.images, self.images.flip(-1))[view], expected, rtol=0, atol=0)
        probe_model.restore_probe(selected, {"probe": original.probe_state_dict()})
        original.load_probe_state_dict(selected.probe_state_dict())

    def test_all_heads_train_only_the_readout_and_roundtrip_checkpoint(self):
        for kind in probe_model.PROBE_TYPES:
            with self.subTest(kind=kind):
                settings = {"hidden_channels": 16} if kind in {"mlp_probe", "cnn_probe"} else {}
                model = self.build(kind, **settings).train()
                self.assertFalse(model.backbone.training)
                self.assertFalse(model.backbone.model.training)
                before = {name: p.detach().clone() for name, p in model.backbone.named_parameters()}
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
                output = model(self.images, self.images.flip(-1))
                loss = sum(gaussian_heatmap_loss(value, torch.rand(2, 68, 2)) for value in output.values())
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
                self.assertTrue(all(p.grad is not None for head in (model.ap_head, model.lat_head) for p in head.parameters()))
                optimizer.step()
                for name, p in model.backbone.named_parameters():
                    torch.testing.assert_close(p, before[name], rtol=0, atol=0)
                model.eval()
                expected = model(self.images, self.images.flip(-1))
                payload = {"probe_spec": model.probe_spec, "probe": model.probe_state_dict()}
                stream = io.BytesIO()
                torch.save(payload, stream)
                stream.seek(0)
                restored = self.build(kind, **settings).eval()
                restored.backbone.load_state_dict(model.backbone.state_dict())
                probe_model.restore_probe(restored, torch.load(stream, weights_only=True))
                for view, value in restored(self.images, self.images.flip(-1)).items():
                    self.assertEqual(tuple(value.shape), (2, 68, 8, 4))
                    torch.testing.assert_close(value, expected[view], rtol=0, atol=0)

    def test_linear_heads_obey_affine_superposition(self):
        for kind in ("legacy_linear", "multilayer_linear", "spatial_linear"):
            with self.subTest(kind=kind):
                head = self.build(kind).ap_head
                channels = head.in_channels
                a, b = torch.randn(2, channels, 8, 4), torch.randn(2, channels, 8, 4)
                torch.testing.assert_close(head(a + b), head(a) + head(b) - head(torch.zeros_like(a)), atol=2e-6, rtol=1e-5)

    def test_hf_and_native_multilayer_features_use_frozen_normalization_and_correct_layers(self):
        native = self.build("multilayer_linear")
        with patch.object(probe_model, "build_backbone", return_value=TinyBackbone(native=False)):
            hf = self.build("multilayer_linear")
        hf.backbone.model.load_state_dict(native.backbone.model.state_dict())
        torch.testing.assert_close(native._features(self.images), hf._features(self.images), rtol=0, atol=0)
        # The last selected block agrees with the ordinary normalized final map.
        torch.testing.assert_close(hf._features(self.images)[:, -8:], hf.backbone(self.images).feature_map)
        with torch.no_grad():
            expected = torch.cat([
                hf.backbone.model.layernorm(hf.backbone.model(self.images).hidden_states[index + 1])[:, 3:]
                .transpose(1, 2).reshape(2, 8, 8, 4)
                for index in (2, 5, 8, 11)
            ], dim=1)
        torch.testing.assert_close(hf._features(self.images), expected)
        self.assertFalse(hf._features(self.images.requires_grad_()).requires_grad)

    def test_mismatched_architecture_and_invalid_configs_are_rejected(self):
        model = self.build("multilayer_linear")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            probe_model.restore_probe(model, {"probe": model.probe_state_dict()})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            probe_model.restore_probe(model, {
                "probe_spec": {"type": "multilayer_linear", "feature_layers": [1, 4, 7, 10]},
                "probe": model.probe_state_dict(),
            })
        for config in (
            {"type": "unknown"}, {"feature_layers": [2, 11]},
            {"type": "multilayer_linear", "feature_layers": "last"},
            {"type": "multilayer_linear", "feature_layers": [2, 2]},
            {"type": "multilayer_linear", "feature_layers": [11, 2]},
            {"type": "cnn_probe", "hidden_channels": 0}, {"unexpected": 1},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                probe_model.normalize_probe_config(config)
        with self.assertRaisesRegex(ValueError, "block"):
            self.build("multilayer_linear", feature_layers=[2, 12])(self.images, self.images)

    def test_presets_and_cli_selection_keep_output_directories_separate(self):
        presets = sorted((TASK / "configs/probes").glob("*/*.yaml"))
        self.assertEqual(len(presets), 14)
        for path in presets:
            config = train_probe.load_probe_config(["--config", str(path)])
            self.assertIn(config["model"]["probe"]["type"], config["output_dir"])
        base = str(TASK / "configs/dinov2.yaml")
        config = train_probe.load_probe_config(["--config", base])
        self.assertEqual(config["model"]["probe"]["type"], "legacy_linear")
        self.assertIn("task07_keypoint_probe/legacy_linear", config["output_dir"])
        config = train_probe.load_probe_config([
            "--config", str(TASK / "configs/probes/multilayer_linear/dinov2.yaml"),
            "--probe-type", "cnn_probe", "--hidden-channels", "16",
        ])
        self.assertEqual(config["model"]["probe"], {"type": "cnn_probe", "feature_layers": "last", "hidden_channels": 16})
        self.assertIn("cnn_probe", config["output_dir"])
        for kind in ("mlp_probe", "cnn_probe"):
            config = train_probe.load_probe_config([
                "--config", str(TASK / "configs/lora_dinov2.yaml"), "--probe-type", kind,
            ])
            self.assertTrue(config["model"]["adapter"]["enabled"])
            self.assertEqual(config["model"]["probe"]["type"], kind)
            self.assertIn(f"task07_keypoint_lora_probe/{kind}", config["output_dir"])

    def test_lora_gradients_optimizer_groups_and_full_trainable_checkpoint_roundtrip(self):
        for kind in probe_model.PROBE_TYPES:
            with self.subTest(kind=kind):
                settings = {"hidden_channels": 16} if kind in {"mlp_probe", "cnn_probe"} else {}
                torch.manual_seed(39)
                model = self.build(kind, adapter=TINY_ADAPTER, **settings).train()
                self.assertTrue(model.backbone.training)
                self.assertTrue(model.backbone.model.training)
                before = {name: p.detach().clone() for name, p in model.named_parameters()}
                optimizer, scheduler = build_downstream_optimizer(model, {
                    "learning_rate": 0.001, "adapter_learning_rate": 0.0001,
                    "head_learning_rate": 0.002, "epochs": 3,
                })
                self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.0001, 0.002])
                adapter_ids = {id(p) for p in model.backbone.parameters() if p.requires_grad}
                head_ids = {id(p) for head in (model.ap_head, model.lat_head) for p in head.parameters()}
                self.assertEqual({id(p) for p in optimizer.param_groups[0]["params"]}, adapter_ids)
                self.assertEqual({id(p) for p in optimizer.param_groups[1]["params"]}, head_ids)
                target = torch.rand(2, 68, 2)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    output = model(self.images, self.images.flip(-1))
                    loss = sum(gaussian_heatmap_loss(value, target) for value in output.values())
                    loss.backward()
                    self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))
                    self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
                    self.assertTrue(any(p.grad.abs().sum() > 0 for p in model.backbone.parameters() if p.requires_grad))
                    optimizer.step()
                    scheduler.step()
                self.assertTrue(any(
                    not torch.equal(p, before[name])
                    for name, p in model.backbone.named_parameters(prefix="backbone") if p.requires_grad
                ))
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
                model.eval()
                expected = model(self.images, self.images.flip(-1))
                state = trainable_state_dict(model)
                self.assertEqual(set(state), {name for name, p in model.named_parameters() if p.requires_grad})
                payload = {"probe_spec": model.probe_spec, "adapter_spec": model.adapter_spec, "trainable_state": state}
                stream = io.BytesIO()
                torch.save(payload, stream)
                stream.seek(0)
                torch.manual_seed(39)
                restored = self.build(kind, adapter=TINY_ADAPTER, **settings).eval()
                probe_model.restore_probe(restored, torch.load(stream, weights_only=True))
                for view, value in restored(self.images, self.images.flip(-1)).items():
                    torch.testing.assert_close(value, expected[view], rtol=0, atol=0)
                if kind != "legacy_linear":
                    self.assertFalse(restored._features(self.images).requires_grad)

    def test_lora_restore_rejects_incomplete_state_and_incompatible_adapters(self):
        model = self.build("mlp_probe", adapter=TINY_ADAPTER, hidden_channels=16)
        payload = {"probe_spec": model.probe_spec, "adapter_spec": model.adapter_spec,
                   "trainable_state": trainable_state_dict(model)}
        for key in ("backbone.model.query.lora_B.weight", "ap_head.0.weight"):
            incomplete = {name: value for name, value in payload["trainable_state"].items() if name != key}
            with self.assertRaisesRegex(ValueError, "Incomplete LoRA checkpoint"):
                probe_model.restore_probe(model, {**payload, "trainable_state": incomplete})
        with self.assertRaisesRegex(ValueError, "Incomplete LoRA checkpoint"):
            probe_model.restore_probe(model, {**payload, "trainable_state": {
                **payload["trainable_state"], "backbone.model.query.base.weight": model.backbone.model.query.base.weight,
            }})
        with self.assertRaisesRegex(ValueError, "adapter mismatch"):
            probe_model.restore_probe(self.build("mlp_probe", hidden_channels=16), payload)
        other = self.build("mlp_probe", adapter={**TINY_ADAPTER, "alpha": 8}, hidden_channels=16)
        with self.assertRaisesRegex(ValueError, "adapter mismatch"):
            probe_model.restore_probe(other, payload)
        with self.assertRaisesRegex(ValueError, "must contain trainable_state"):
            probe_model.restore_probe(model, {"probe_spec": model.probe_spec, "adapter_spec": model.adapter_spec})

    def test_original_lora_checkpoint_remains_loadable(self):
        original = self.build("legacy_linear", adapter=TINY_ADAPTER)
        with torch.no_grad():
            original.backbone.model.query.lora_B.weight.add_(0.1)
        payload = {"config": {"model": {"adapter": TINY_ADAPTER}},
                   "trainable_state": trainable_state_dict(original)}
        restored = self.build("legacy_linear", adapter=TINY_ADAPTER)
        probe_model.restore_probe(restored, payload)
        for name, value in trainable_state_dict(restored).items():
            torch.testing.assert_close(value, payload["trainable_state"][name], rtol=0, atol=0)

    def test_lora_mlp_and_cnn_training_save_adapters_and_step_scheduler(self):
        clean = {str(index): {} for index in range(10)}
        for kind in ("mlp_probe", "cnn_probe"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                config = {
                    "model": {"backbone": "tiny", "weights": "unused.pth", "image_height": 16, "image_width": 8,
                              "probe": {"type": kind, "hidden_channels": 16}, "adapter": TINY_ADAPTER},
                    "training": {"batch_size": 2, "workers": 0, "epochs": 3, "learning_rate": 0.001,
                                 "adapter_learning_rate": 0.0001, "head_learning_rate": 0.002,
                                 "warmup_epochs": 1, "gradient_accumulation_steps": 2, "seed": 42},
                    "device": "cpu", "output_dir": directory,
                }
                schedulers = []

                def optimizer_builder(model, training):
                    optimizer, scheduler = build_downstream_optimizer(model, training)
                    schedulers.append(scheduler)
                    return optimizer, scheduler

                with patch.object(train_probe, "audit_pairs", return_value=(clean, {})), patch.object(
                    train_probe, "PairedWholeSpineDataset", TinyPairedDataset
                ), patch.object(train_probe, "build_downstream_optimizer", side_effect=optimizer_builder), redirect_stdout(io.StringIO()):
                    train_probe.main(config)
                self.assertEqual(schedulers[0].last_epoch, 3)
                output = Path(directory)
                summary = json.loads((output / "training_summary.json").read_text())
                self.assertGreater(summary["backbone_trainable"], 0)
                self.assertEqual(summary["protocol"], "task07_lora_finetune")
                self.assertEqual(summary["checkpoint"], f"best_lora_{kind}.pt")
                checkpoint = torch.load(output / summary["checkpoint"], weights_only=False)
                self.assertEqual(checkpoint["adapter_spec"], probe_model.adapter_spec(TINY_ADAPTER))
                self.assertIn("backbone.model.query.lora_B.weight", checkpoint["trainable_state"])
                self.assertIn("lat_head.0.weight", checkpoint["trainable_state"])
                self.assertFalse(any(".base." in name for name in checkpoint["trainable_state"]))
                self.assertNotIn("probe", checkpoint)
                self.assertEqual(len(json.loads((output / "test_predictions.json").read_text())), 2)

    def test_training_validation_checkpoint_and_test_outputs(self):
        clean = {str(index): {} for index in range(10)}
        for kind in probe_model.PROBE_TYPES:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                spec = {"type": kind}
                if kind in {"cnn_probe", "mlp_probe"}:
                    spec["hidden_channels"] = 16
                config = {
                    "model": {"backbone": "tiny", "weights": "unused.pth", "image_height": 16, "image_width": 8, "probe": spec},
                    "training": {"batch_size": 2, "workers": 0, "epochs": 1, "learning_rate": 0.001, "seed": 42},
                    "device": "cpu", "output_dir": directory,
                }
                with patch.object(train_probe, "audit_pairs", return_value=(clean, {})), patch.object(
                    train_probe, "PairedWholeSpineDataset", TinyPairedDataset
                ), redirect_stdout(io.StringIO()):
                    train_probe.main(config)
                output = Path(directory)
                split = json.loads((output / "split_and_audit.json").read_text())
                self.assertEqual(split["splits"], deterministic_split(list(clean), 42))
                summary = json.loads((output / "training_summary.json").read_text())
                self.assertEqual(summary["backbone_trainable"], 0)
                checkpoint = torch.load(output / summary["checkpoint"], weights_only=False)
                self.assertEqual(checkpoint["probe_spec"]["type"], kind)
                self.assertEqual(set(checkpoint["probe"]), {"ap_head", "lat_head"})
                self.assertEqual(len(json.loads((output / "test_predictions.json").read_text())), 2)
                metrics = json.loads((output / "test_metrics.json").read_text())
                self.assertIn("mean_epsilon_dec", metrics)
                self.assertEqual(metrics["probe_spec"]["type"], kind)
                with self.assertRaisesRegex(FileExistsError, "not empty"):
                    train_probe.main(config)


if __name__ == "__main__":
    unittest.main()
