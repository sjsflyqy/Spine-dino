"""Regression checks for teacher initialization checkpoint wrappers."""

from pathlib import Path
import unittest

import torch

from visualization.dino_spine_maps.model_loader import _extract_state_dict


class CheckpointWrapperTest(unittest.TestCase):
    def test_training_initialization_teacher_with_metadata(self):
        state = {"cls_token": torch.zeros(1, 1, 8), "storage_tokens": torch.zeros(1, 4, 8)}
        actual = _extract_state_dict({"teacher": state, "meta": {"architecture": "dinov3"}}, Path("init.pth"))
        self.assertEqual(set(actual), set(state))
        self.assertIs(actual["cls_token"], state["cls_token"])

    def test_prefers_explicit_teacher_over_model(self):
        teacher = {"cls_token": torch.ones(1, 1, 8)}
        student = {"cls_token": torch.zeros(1, 1, 8)}
        actual = _extract_state_dict({"teacher": teacher, "model": student}, Path("ssl.pth"))
        self.assertIs(actual["cls_token"], teacher["cls_token"])

    def test_teacher_backbone_prefix_excludes_projection_head(self):
        tensor = torch.zeros(1, 1, 8)
        wrapped = {"module.backbone.cls_token": tensor, "module.dino_head.weight": torch.zeros(4, 8)}
        actual = _extract_state_dict({"teacher": wrapped}, Path("teacher.pth"))
        self.assertEqual(set(actual), {"cls_token"})
        self.assertIs(actual["cls_token"], tensor)

    def test_unsupported_teacher_does_not_accept_metadata(self):
        with self.assertRaisesRegex(ValueError, "extract_teacher_backbone"):
            _extract_state_dict({"teacher": {"nested": {}}, "meta": {"architecture": "dinov3"}}, Path("invalid.pth"))


if __name__ == "__main__":
    unittest.main()
