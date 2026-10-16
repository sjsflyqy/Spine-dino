"""Regression tests for native DINO wrappers and incorrect architecture configs."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from backbone.checkpoints import extract_state_dict, load_checkpoint, validate_native_architecture
from backbone.registry import NativeDinoBackbone


class TinyDino(nn.Module):
    def __init__(self, generation):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 8))
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 8, 14 if generation == 2 else 16, stride=14 if generation == 2 else 16)
        if generation == 2:
            self.pos_embed = nn.Parameter(torch.randn(1, 5, 8))
        else:
            self.storage_tokens = nn.Parameter(torch.randn(1, 4, 8))


class CheckpointTests(unittest.TestCase):
    def test_wrapper_extraction_preserves_tensors_and_prefers_teacher(self):
        expected = TinyDino(3).state_dict()
        for wrapper in (None, "model", "teacher", "state_dict", "backbone"):
            checkpoint = expected if wrapper is None else {wrapper: expected, "meta": {"architecture": "dinov3"}}
            actual = extract_state_dict(checkpoint, Path("init.pth"))
            self.assertEqual(set(actual), set(expected))
            for key, value in expected.items():
                self.assertIs(actual[key], value)
        actual = extract_state_dict({"teacher": expected, "model": TinyDino(2).state_dict()}, Path("init.pth"))
        self.assertIs(actual["cls_token"], expected["cls_token"])

    def test_backbone_prefixes_drop_ssl_heads(self):
        tensor = torch.ones(1, 1, 8)
        for prefix in (
            "backbone.", "teacher.backbone.", "backbone._fsdp_wrapped_module.",
            "teacher.backbone._fsdp_wrapped_module.",
        ):
            state = {f"model.module.{prefix}cls_token": tensor, "model.module.dino_head.weight": torch.zeros(4, 8)}
            actual = extract_state_dict({"teacher": state}, Path("eval.pth"))
            self.assertEqual(set(actual), {"cls_token"})
            self.assertIs(actual["cls_token"], tensor)

    def test_malformed_wrappers_are_rejected(self):
        for checkpoint in ({}, {"teacher": {"nested": {}}, "meta": {}}, {"meta": {"architecture": "dinov3"}}):
            with self.subTest(checkpoint=checkpoint), self.assertRaisesRegex(ValueError, "extract_teacher_backbone"):
                extract_state_dict(checkpoint, Path("invalid.pth"))
        with self.assertRaises(TypeError):
            extract_state_dict([torch.ones(1)], Path("invalid.pth"))

    def test_architecture_and_patch_mismatches_give_actionable_errors(self):
        for generation in (2, 3):
            state = TinyDino(generation).state_dict()
            validate_native_architecture(state, generation, Path("init.pth"))
            with self.assertRaisesRegex(ValueError, f"model.backbone: dinov{generation}"):
                validate_native_architecture(state, 5 - generation, Path("init.pth"))
        state = TinyDino(2).state_dict()
        state["patch_embed.proj.weight"] = torch.zeros(8, 3, 16, 16)
        with self.assertRaisesRegex(ValueError, "expects patch size 14"):
            validate_native_architecture(state, 2, Path("init.pth"))

    def test_native_loader_strictly_loads_raw_and_wrapped_weights(self):
        for generation in (2, 3):
            state = TinyDino(generation).state_dict()
            for wrapper in (None, "teacher", "model", "state_dict", "backbone"):
                with self.subTest(generation=generation, wrapper=wrapper), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "init.pth"
                    torch.save(state if wrapper is None else {wrapper: state, "meta": {}}, path)
                    calls = []

                    def factory(**kwargs):
                        calls.append(kwargs)
                        model = TinyDino(generation)
                        if "weights" in kwargs:
                            model.load_state_dict(torch.load(kwargs["weights"], weights_only=True), strict=True)
                        return model

                    name = "dinov2.models.vision_transformer" if generation == 2 else "dinov3.hub.backbones"
                    module = SimpleNamespace(**{"vit_base" if generation == 2 else "dinov3_vitb16": factory})
                    with patch.dict(sys.modules, {name: module}), patch.object(sys, "path", sys.path.copy()):
                        backbone = NativeDinoBackbone(generation, str(path))
                    self.assertEqual(backbone.patch_size, 14 if generation == 2 else 16)
                    self.assertFalse(backbone.model.training)
                    self.assertTrue(all(not p.requires_grad for p in backbone.parameters()))
                    for key, value in backbone.model.state_dict().items():
                        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
                    if generation == 3:
                        self.assertEqual(calls, [{"weights": str(path)}] if wrapper is None else [{"pretrained": False}])

    def test_wrong_architecture_fails_before_constructing_backbone(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "init.pth"
            torch.save({"teacher": TinyDino(3).state_dict(), "meta": {"architecture": "dinov3"}}, path)
            constructor = Mock(side_effect=AssertionError("should not build DINOv2"))
            with patch.dict(sys.modules, {"dinov2.models.vision_transformer": SimpleNamespace(vit_base=constructor)}):
                with self.assertRaisesRegex(ValueError, "config selects DINOv2.*DINOv3"):
                    NativeDinoBackbone(2, str(path))
            constructor.assert_not_called()

    def test_missing_tensors_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pth"
            state = TinyDino(3).state_dict()
            del state["cls_token"]
            torch.save({"teacher": state}, path)
            module = SimpleNamespace(dinov3_vitb16=lambda **kwargs: TinyDino(3))
            with patch.dict(sys.modules, {"dinov3.hub.backbones": module}), patch.object(sys, "path", sys.path.copy()):
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    NativeDinoBackbone(3, str(path))

    def test_legacy_serialization_remains_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            state = TinyDino(3).state_dict()
            torch.save(state, path, _use_new_zipfile_serialization=False)
            actual = load_checkpoint(path)
            for key, value in state.items():
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
