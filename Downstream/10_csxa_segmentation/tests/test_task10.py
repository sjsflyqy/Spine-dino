"""Protocol tests: geometry, matching, freezing, gradients and inference failures."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import tempfile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as TF

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))

from common import read_splits, restore_trainable, trainable_state
from dataset import crop_box, make_crop, restore_probability, classifier_input
from models.auxiliary import load_auxiliaries, split_fingerprints
from metrics import SpineFMEvaluator, match_scores
from models.segmenter import Segmenter, segmentation_loss
from backbone.lora import LoRALinear
from pipeline import SpineFMPipeline
from prepare_data import annotation_to_masks
from train import train_epoch
from evaluate import evaluate_split


class TinyBackbone(nn.Module):
    embedding_dim = 8
    patch_size = 14
    adapter_enabled = False

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 8, 14, stride=14).requires_grad_(False)

    def forward(self, x):
        return SimpleNamespace(feature_map=self.projection(x))


class TinyLoRABackbone(TinyBackbone):
    adapter_enabled = True

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.blocks = nn.ModuleList([LoRALinear(nn.Linear(8, 8), rank=2, alpha=2, dropout=0)])

    def forward(self, x):
        feature = self.projection(x)
        tokens = feature.permute(0, 2, 3, 1)
        for block in self.model.blocks:
            tokens = block(tokens)
        return SimpleNamespace(feature_map=tokens.permute(0, 3, 1, 2))


def annotation():
    shapes = []
    for j, level in enumerate(("C3", "C4", "C5", "C6", "C7")):
        for position, p in zip(("top left", "top right", "bottom right", "bottom left"),
                               ((5, 5+j*15), (15, 5+j*15), (15, 15+j*15), (5, 15+j*15))):
            shapes.append(dict(label=f"{level} {position}", shape_type="point", points=[list(p)]))
    return dict(imageWidth=40, imageHeight=100, shapes=shapes)


class GeometryTests(unittest.TestCase):
    def test_official_classifier_preprocessing_matches_reference(self):
        image = Image.fromarray(np.random.RandomState(9).randint(0, 256, (300, 310, 3), dtype=np.uint8))
        for center in [(155., 150.), (12.7, 15.3)]:
            patch = TF.crop(image, top=int(center[1]-128), left=int(center[0]-128), height=256, width=256)
            reference = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])(patch)
            reference = torch.nn.functional.interpolate(reference[None], (224, 224))[0]
            actual = classifier_input(image, center, "spinefm_official")
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            self.assertGreaterEqual(float(actual.min()), 0)
            self.assertLessEqual(float(actual.max()), 1)

    def test_official_loader_rejects_wrong_reference_split(self):
        store = SimpleNamespace(splits={"train": ["a"], "val": ["b"], "test": ["c"]})
        config = dict(source="spinefm_official", reference_split_hashes={})
        with self.assertRaisesRegex(ValueError, "published SpineFM split"):
            load_auxiliaries(config, store, torch.device("cpu"))
        self.assertEqual(split_fingerprints(store.splits), split_fingerprints(dict(reversed(list(store.splits.items())))))

    def test_crossed_corner_labels_preserve_region(self):
        original = annotation()
        expected, _ = annotation_to_masks(original, (40, 100))
        a = original["shapes"]
        a[2]["points"], a[3]["points"] = a[3]["points"], a[2]["points"]
        actual, audit = annotation_to_masks(original, (40, 100))
        np.testing.assert_array_equal(actual["masks"], expected["masks"])
        self.assertEqual([x["action"] for x in audit], ["reorder_vertices_only"])

    def test_multiclick_is_averaged_but_ambiguous_clicks_rejected(self):
        a = annotation()
        a["shapes"][0]["points"] = [[4.8, 5], [5.2, 5]]
        result, audit = annotation_to_masks(a, (40, 100))
        self.assertEqual(len(audit), 1)
        np.testing.assert_allclose(result["corners"][0, 0], [5, 5])
        a["shapes"][0]["points"] = [[1, 5], [9, 5]]
        with self.assertRaises(ValueError):
            annotation_to_masks(a, (40, 100))

    def test_crop_border_and_restore_have_same_origin(self):
        image = Image.new("RGB", (40, 100))
        mask = np.zeros((100, 40), np.uint8)
        mask[0:10, 0:10] = 1
        _, point, target, box = make_crop(image, (2, 3), 20, 20, mask, 20)
        restored = restore_probability(target[0], box, image.size)
        np.testing.assert_array_equal(restored.numpy(), mask)
        np.testing.assert_allclose(point.numpy(), [10, 10])
        self.assertEqual(box, (-8, -7, 12, 13))

    def test_off_image_crop_restores_zero(self):
        restored = restore_probability(torch.ones(5, 5), (200, 200, 205, 205), (40, 100))
        self.assertEqual(float(restored.sum()), 0)


class MetricsTests(unittest.TestCase):
    def test_missing_vertebra_and_missing_image_are_in_denominator(self):
        gt, _ = annotation_to_masks(annotation(), (40, 100))
        ev = SpineFMEvaluator()
        ev.add("found", gt["masks"][:4], gt["masks"])
        ev.add("empty", [], gt["masks"])
        result = ev.summary(["found", "empty"])["table"]
        self.assertEqual(result["Avg"]["identified_percent"], 40.)
        self.assertEqual(result["Avg"]["located_dsc"], 1.)
        self.assertEqual(result["Avg"]["overall_dsc"], .4)
        self.assertIsNone(result["C7"]["located_dsc"])
        with self.assertRaises(ValueError):
            ev.summary(["found"])

    def test_reference_last_match_and_one_to_one_are_distinct(self):
        matrix = np.array([[.9, .5], [.8, .2]])
        ref, assignment = match_scores(matrix)
        np.testing.assert_allclose(ref, [.5, .8])
        unique, idx = match_scores(matrix, "one_to_one")
        np.testing.assert_allclose(unique, [.5, .8])
        self.assertEqual(len(set(idx)), 2)
        ref, _ = match_scores(np.array([[.9], [.8]]))
        unique, _ = match_scores(np.array([[.9], [.8]]), "one_to_one")
        self.assertEqual(int((ref > .4).sum()), 2)
        self.assertEqual(int((unique > .4).sum()), 1)

    def test_localization_threshold_is_strict(self):
        score, match = match_scores(np.array([[.4]]))
        self.assertEqual(float(score[0]), 0)
        self.assertEqual(int(match[0]), -1)


class ModelTests(unittest.TestCase):
    def test_lora_linear_updates_only_adapter_and_linear_head(self):
        config = dict(mode="lora_linear", input_size=56, target_size=32, adapter=dict(enabled=True))
        model = Segmenter(config, TinyLoRABackbone())
        self.assertTrue(model.uses_lora)
        self.assertFalse(model.uses_decoder)
        self.assertFalse(hasattr(model, "mask_decoder"))
        frozen = {n: p.detach().clone() for n, p in model.backbone.named_parameters() if not p.requires_grad}
        before_head = model.head.weight.detach().clone()
        before_lora = model.backbone.model.blocks[0].lora_B.weight.detach().clone()
        samples = [dict(image=torch.rand(2, 3, 56, 56), point=torch.tensor([[28., 28.], [28., 28.]]),
                        mask=torch.ones(2, 1, 32, 32))]
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        train_epoch(model, samples, optimizer, torch.device("cpu"))
        self.assertFalse(torch.equal(model.head.weight, before_head))
        self.assertFalse(torch.equal(model.backbone.model.blocks[0].lora_B.weight, before_lora))
        for name, parameter in model.backbone.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(parameter, frozen[name]))
        model.eval()
        expected, quality = model(samples[0]["image"], samples[0]["point"])
        self.assertIsNone(quality)
        state = trainable_state(model)
        self.assertTrue(all(n.startswith("head.") or "lora_" in n for n in state))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lora_linear.pt"
            torch.save(state, path)
            state = torch.load(path, weights_only=True)
        with torch.no_grad():
            model.head.weight.zero_()
        restore_trainable(model, state)
        actual, _ = model(samples[0]["image"], samples[0]["point"])
        torch.testing.assert_close(actual, expected)

    def test_generic_sam_mlp_key_mapping_is_strict(self):
        model = Segmenter(dict(mode="decoder", input_size=56, target_size=32, embedding_grid=4), TinyBackbone())
        state = {}
        for prefix in ("prompt_encoder", "mask_decoder"):
            for key, value in getattr(model, prefix).state_dict().items():
                generic = key.replace(".mlp.layers.0.0.", ".mlp.lin1.").replace(".mlp.fc.", ".mlp.lin2.")
                state[f"{prefix}.{generic}"] = value.clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generic_sam.pth"
            torch.save(state, path)
            model.load_sam_initialization(path)
            state.pop(next(iter(state)))
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                model.load_sam_initialization(path)

    def test_training_step_handles_partial_accumulation(self):
        model = Segmenter(dict(mode="linear", input_size=56, target_size=32), TinyBackbone())
        samples = [dict(image=torch.rand(1, 3, 56, 56), point=torch.tensor([[28., 28.]]),
                        mask=torch.ones(1, 1, 32, 32)) for _ in range(3)]
        before = model.head.weight.detach().clone()
        frozen = model.backbone.projection.weight.detach().clone()
        optimizer = torch.optim.SGD(model.head.parameters(), lr=.1)
        loss = train_epoch(model, samples, optimizer, torch.device("cpu"), accumulation=2)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.head.weight))
        self.assertTrue(torch.equal(frozen, model.backbone.projection.weight))

    def test_linear_only_trains_affine_head_and_roundtrips(self):
        model = Segmenter(dict(mode="linear", input_size=56, target_size=32), TinyBackbone())
        model.train()
        self.assertFalse(model.backbone.training)
        image, points = torch.rand(2, 3, 56, 56), torch.tensor([[28., 28.], [28., 28.]])
        logits, quality = model(image, points)
        segmentation_loss(logits, torch.ones_like(logits), quality).backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertIsNotNone(model.head.weight.grad)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 9)
        state = trainable_state(model)
        self.assertEqual(set(state), {"head.weight", "head.bias"})
        restore_trainable(model, state)
        with self.assertRaises(ValueError):
            restore_trainable(model, {"head.weight": state["head.weight"]})

    def test_sam_decoder_batch_gradients_and_no_backbone_update(self):
        model = Segmenter(dict(mode="decoder", input_size=56, target_size=32, embedding_grid=4), TinyBackbone())
        model.train()
        image, points = torch.rand(2, 3, 56, 56), torch.tensor([[28., 28.], [25., 24.]])
        logits, quality = model(image, points)
        self.assertEqual(tuple(logits.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(quality.shape), (2, 1))
        segmentation_loss(logits, torch.ones_like(logits), quality).backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertIsNotNone(model.feature_adapter[0].weight.grad)
        self.assertTrue(any(p.grad is not None for p in model.mask_decoder.parameters()))


class PipelineTests(unittest.TestCase):
    def test_complete_evaluation_serializes_empty_predictions_and_matches(self):
        payload, _ = annotation_to_masks(annotation(), (40, 100))
        image = Image.new("RGB", (40, 100))
        class Store:
            splits = {"test": ["empty", "perfect"]}
            fingerprint = "test-fixture"
            def load(self, identifier):
                return image, payload
        class Pipeline:
            mask_threshold = .9
            calls = 0
            def predict(self, incoming_image):
                assert incoming_image is image
                self.calls += 1
                masks = [] if self.calls == 1 else list(payload["masks"])
                return masks, dict(candidates=0 if not masks else 5, stops=[], seeds=[])
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_split(Pipeline(), Store(), "test", directory, save_masks=True)
            self.assertEqual(result["table"]["Avg"]["overall_dsc"], .5)
            self.assertEqual(result["pipeline_diagnostics"]["zero_prediction_images"], 1)
            saved = json.loads((Path(directory) / "metrics.json").read_text())
            self.assertEqual(saved, result)
            with np.load(Path(directory) / "masks" / "empty.npz") as cache:
                self.assertEqual(cache["masks"].shape, (0, 100, 40))

    def test_no_detections_returns_empty_without_gt_or_exception(self):
        class Detector:
            def __call__(self, images):
                h, w = images[0].shape[-2:]
                return [dict(scores=torch.zeros(0), masks=torch.zeros(0, 1, h, w))]
        model = Segmenter(dict(mode="linear", input_size=56, target_size=32), TinyBackbone())
        pipeline = SpineFMPipeline(model, {"detector": Detector()}, torch.device("cpu"))
        masks, trace = pipeline.predict(Image.new("RGB", (40, 100)))
        self.assertEqual(masks, [])
        self.assertIn("fewer_than_three_valid_seeds", trace["stops"])

    def test_seeds_and_iteration_limit(self):
        class Detector:
            def __call__(self, images):
                masks = torch.zeros(3, 1, 200, 100)
                for j in range(3):
                    masks[j, 0, 40+j*20:45+j*20, 40:45] = 1
                return [dict(scores=torch.ones(3), masks=masks)]
        class Predictor:
            def __call__(self, x):
                points = x.reshape(-1, 3, 2)
                return 2*points[:, -1]-points[:, -2]
        class Classifier:
            def __call__(self, x):
                return torch.tensor([[0., 1.]])
        model = Segmenter(dict(mode="linear", input_size=56, target_size=32), TinyBackbone())
        pipe = SpineFMPipeline(model, dict(detector=Detector(), point_predictor=Predictor(), classifier=Classifier()),
                               torch.device("cpu"), dict(max_steps_per_direction=1))
        def segment(image, point):
            mask = np.zeros((image.height, image.width), bool)
            x, y = map(int, point)
            mask[y-1:y+2, x-1:x+2] = True
            return mask, tuple(point)
        pipe.segment = segment
        masks, trace = pipe.predict(Image.new("RGB", (100, 200)))
        self.assertEqual(len(masks), 5)
        self.assertEqual(trace["stops"], ["iteration_limit", "iteration_limit"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
