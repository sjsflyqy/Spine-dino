import random
import unittest

import torch

from dinov2.data.masking import MaskingGenerator

from methods.geotopo_dino.data.collate import collate_data_and_cast_gcvd
from methods.geotopo_dino.losses.gcvd_loss import GeometryDistillationLoss


def _sample(sample_id: int, n_locals: int = 2, anchor_valid_mask=None):
    if anchor_valid_mask is None:
        anchor_valid_mask = torch.ones((2, 2), dtype=torch.bool)
    anchor = torch.full((3, 2, 2), float(10 + sample_id))
    random_global = torch.full((3, 2, 2), float(20 + sample_id))
    locals_ = [torch.full((3, 2, 2), float(100 * view + sample_id)) for view in range(n_locals)]
    return (
        {
            "global_crops": [anchor, random_global],
            "local_crops": locals_,
            "anchor_geometry": {
                "transform": torch.eye(3),
                "valid_mask": anchor_valid_mask,
                "source_size": torch.tensor([2, 2]),
            },
            "random_global_geometry": {"transform": torch.eye(3)},
            "local_geometry": [
                {
                    "transform": torch.eye(3),
                    "box_xyxy": torch.tensor([0.0, 0.0, 2.0, 2.0]),
                    "flipped": False,
                }
                for _ in range(n_locals)
            ],
        },
        (),
    )


class CollateTests(unittest.TestCase):
    def test_view_major_order_and_metadata(self):
        random.seed(0)
        result = collate_data_and_cast_gcvd(
            [_sample(0), _sample(1)],
            mask_ratio_tuple=(0.1, 0.5),
            mask_probability=0.5,
            dtype=torch.float32,
            n_tokens=4,
            mask_generator=MaskingGenerator((2, 2), max_num_patches=2),
        )
        local_values = result["collated_local_crops"][:, 0, 0, 0]
        torch.testing.assert_close(local_values, torch.tensor([0.0, 1.0, 100.0, 101.0]))
        torch.testing.assert_close(result["local_sample_ids"], torch.tensor([0, 1, 0, 1]))
        torch.testing.assert_close(result["local_view_ids"], torch.tensor([0, 0, 1, 1]))
        torch.testing.assert_close(result["global_sample_ids"], torch.tensor([0, 1, 0, 1]))
        torch.testing.assert_close(result["global_view_ids"], torch.tensor([0, 0, 1, 1]))

    def test_anchor_masks_never_select_padding(self):
        random.seed(1)
        valid = torch.tensor([[False, True], [False, True]])
        result = collate_data_and_cast_gcvd(
            [_sample(0, anchor_valid_mask=valid)],
            mask_ratio_tuple=(0.5, 0.5),
            mask_probability=1.0,
            dtype=torch.float32,
            n_tokens=4,
            mask_generator=MaskingGenerator((2, 2), max_num_patches=2),
        )
        anchor_mask = result["collated_masks"][0].reshape(2, 2)
        self.assertFalse((anchor_mask & ~valid).any())
        self.assertEqual(int(anchor_mask.sum()), 1)


class LossTests(unittest.TestCase):
    def test_identical_normalized_embeddings_have_zero_loss(self):
        patch = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
        region = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
        valid = torch.ones((2, 4), dtype=torch.bool)
        output = GeometryDistillationLoss()(
            student_patch_embeddings=patch,
            teacher_patch_embeddings=patch,
            student_region_embeddings=region,
            teacher_region_embeddings=region,
            valid_mask=valid,
        )
        self.assertLess(abs(output["dense"].item()), 1e-6)
        self.assertLess(abs(output["region"].item()), 1e-6)


if __name__ == "__main__":
    unittest.main()
