import unittest

import torch
from PIL import Image

from methods.geotopo_dino.data.augmentations import DataAugmentationGeoTopoDINO
from methods.geotopo_dino.geometry.transforms import (
    crop_resize_transform,
    horizontal_flip_transform,
    letterbox_transform,
    transform_points,
)
from methods.geotopo_dino.geometry.warp import warp_anchor_features_to_local


class TransformTests(unittest.TestCase):
    def test_crop_resize_edges(self):
        transform = crop_resize_transform(
            top=20,
            left=10,
            height=40,
            width=20,
            output_size=(80, 100),
        )
        source = torch.tensor([[10.0, 20.0], [30.0, 60.0]])
        expected = torch.tensor([[0.0, 0.0], [100.0, 80.0]])
        torch.testing.assert_close(transform_points(transform, source), expected)

    def test_horizontal_flip_edges(self):
        points = torch.tensor([[0.0, 2.0], [100.0, 2.0], [25.0, 2.0]])
        expected = torch.tensor([[100.0, 2.0], [0.0, 2.0], [75.0, 2.0]])
        torch.testing.assert_close(
            transform_points(horizontal_flip_transform(100), points),
            expected,
        )

    def test_letterbox_rounding_uses_separate_scales(self):
        transform = letterbox_transform(
            source_size=(101, 51),
            resized_size=(518, 261),
            padding_left=128,
            padding_top=0,
        )
        lower_right = transform_points(transform, torch.tensor([[51.0, 101.0]]))
        torch.testing.assert_close(lower_right, torch.tensor([[389.0, 518.0]]))


class WarpTests(unittest.TestCase):
    def test_identity_warp(self):
        height, width = 3, 4
        features = torch.arange(height * width * 2, dtype=torch.float32).reshape(1, height * width, 2)
        aligned, valid = warp_anchor_features_to_local(
            features,
            torch.eye(3).unsqueeze(0),
            torch.eye(3).unsqueeze(0),
            torch.tensor([0]),
            torch.ones((1, height, width), dtype=torch.bool),
            patch_size=1,
            anchor_grid_size=(height, width),
            local_grid_size=(height, width),
        )
        torch.testing.assert_close(aligned, features)
        self.assertTrue(valid.all())

    def test_local_flip_reverses_teacher_grid(self):
        features = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
        aligned, valid = warp_anchor_features_to_local(
            features,
            torch.eye(3).unsqueeze(0),
            horizontal_flip_transform(4).unsqueeze(0),
            torch.tensor([0]),
            torch.ones((1, 1, 4), dtype=torch.bool),
            patch_size=1,
            anchor_grid_size=(1, 4),
            local_grid_size=(1, 4),
        )
        torch.testing.assert_close(aligned.flatten(), torch.tensor([3.0, 2.0, 1.0, 0.0]))
        self.assertTrue(valid.all())

    def test_padding_is_invalid(self):
        features = torch.ones((1, 16, 1))
        valid_anchor = torch.zeros((1, 4, 4), dtype=torch.bool)
        valid_anchor[:, :, 1:3] = True
        _, valid = warp_anchor_features_to_local(
            features,
            torch.eye(3).unsqueeze(0),
            torch.eye(3).unsqueeze(0),
            torch.tensor([0]),
            valid_anchor,
            patch_size=1,
            anchor_grid_size=(4, 4),
            local_grid_size=(4, 4),
        )
        torch.testing.assert_close(valid.reshape(4, 4), valid_anchor[0])


class AugmentationTests(unittest.TestCase):
    def test_anchor_and_local_shapes_and_geometry(self):
        image = Image.new("RGB", (100, 300), color=(128, 128, 128))
        augmentation = DataAugmentationGeoTopoDINO(
            (0.5, 1.0),
            (0.2, 0.5),
            2,
            global_crops_size=518,
            local_crops_size=196,
            patch_size=14,
            horizontal_flip_probability=0.0,
        )
        # Geometry is the subject of this test.  Keeping photometric transforms
        # deterministic also avoids a torchvision/NumPy compatibility issue in
        # development environments that do not use the pinned project versions.
        augmentation.anchor_photometric = lambda value: value
        augmentation.random_global_photometric = lambda value: value
        augmentation.local_photometric = lambda value: value
        output = augmentation(image)
        self.assertEqual(output["global_crops"][0].shape, (3, 518, 518))
        self.assertEqual(output["global_crops"][1].shape, (3, 518, 518))
        self.assertEqual(output["local_crops"][0].shape, (3, 196, 196))
        self.assertEqual(output["anchor_geometry"]["valid_mask"].shape, (37, 37))
        self.assertEqual(len(output["local_geometry"]), 2)


if __name__ == "__main__":
    unittest.main()
