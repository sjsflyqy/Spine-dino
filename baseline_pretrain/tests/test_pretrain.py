from __future__ import annotations

import copy
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import yaml

from baseline_pretrain.checkpointing import (atomic_torch_save, initialize, load_tensor_checkpoint, public_state,
                                            sha256)
from baseline_pretrain.config import ROOT, load_config
from baseline_pretrain.data import IndexedSpineDataset, StreamBatchSampler
from baseline_pretrain.export_backbone import encoder_state
from baseline_pretrain.models import VisionTransformer, build_network
from baseline_pretrain.objectives import koleo, sinkhorn
from baseline_pretrain.train import initialize_fsdp_adam_state, main, output_path, parameter_groups


def tiny_config(method, root):
    c = load_config(f"baseline_pretrain/configs/{method}.yaml")
    c["model"].update(dim=24, depth=2, heads=3, image_size=32, patch_size=8,
                       gradient_checkpointing=True, drop_path=0.1)
    c["crops"].update(global_size=32, local_size=16, local_count=2)
    c["training"].update(expected_world_size=1, parallel="none", precision="fp32",
                           batch_size_per_gpu=2, accumulation_steps=2, total_updates=4,
                           # Save at update 2 while projection heads are still
                           # frozen; resume must also handle missing Adam states.
                           warmup_updates=1, freeze_last_layer_updates=3 if method != "medical_mae" else 0,
                           checkpoint_every=2, keep_last=3, workers=0, log_every=1)
    c["initialization"] = str(root / f"public_{method}.pth")
    c["data"] = {"root": str(root / "data"), "index": str(root / "data/extra/image_files-TRAIN.npy")}
    if method == "medical_mae":
        c["decoder"].update(dim=16, depth=1, heads=4)
    else:
        c["objective"].update(prototypes=17, head_hidden=32, dino_bottleneck=8,
                               temperature_warmup_updates=1)
        if method == "chexfound":
            c["objective"]["ibot_bottleneck"] = 8
    return c


def write_fixture(root, method):
    data = root / "data"
    (data / "train/spine").mkdir(parents=True, exist_ok=True)
    (data / "extra").mkdir(exist_ok=True)
    names = []
    for i in range(7):
        name = f"spine/{i}.png"
        image = np.random.default_rng(i).integers(0, 256, (43, 37, 3), dtype=np.uint8)
        Image.fromarray(image).save(data / "train" / name)
        names.append(name)
    np.save(data / "extra/image_files-TRAIN.npy", np.array(names))
    c = tiny_config(method, root)
    torch.manual_seed(12)
    state = build_network(c).state_dict()
    if method == "dax":
        state = {k.replace("dino_head.", "head.").replace("last_layer.weight_g", "last_layer.parametrizations.weight.original0").replace("last_layer.weight_v", "last_layer.parametrizations.weight.original1"): v for k, v in state.items()}
    if method == "chexfound":
        state = {k.replace("backbone.blocks.0.", "backbone.blocks.0.0.").replace("backbone.blocks.1.", "backbone.blocks.1.1."): v for k, v in state.items()}
    torch.save({"model" if method == "medical_mae" else "teacher": state, "epoch": 799}, c["initialization"])
    config_path = root / f"{method}.yaml"
    config_path.write_text(yaml.safe_dump(c))
    return c, config_path


class PretrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dax_reference_output_parity_including_rectangular_interpolation(self):
        source = ROOT / "reference_code/baseline/DAX/DAX-main/code/vision_transformer_dax.py"
        utils = types.ModuleType("utils")
        utils.trunc_normal_ = torch.nn.init.trunc_normal_
        spec = importlib.util.spec_from_file_location("_dax_reference_test", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"utils": utils}):
            spec.loader.exec_module(module)
        reference = module.VisionTransformer(img_size=[32], patch_size=8, embed_dim=24,
                                             depth=2, num_heads=3, qkv_bias=True,
                                             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)).eval()
        model = VisionTransformer(dim=24, depth=2, heads=3, image_size=32, patch_size=8, antialias=True).eval()
        model.load_state_dict(reference.state_dict(), strict=True)
        with torch.no_grad():
            for shape in ((2, 3, 32, 32), (2, 3, 32, 48)):
                x = torch.randn(shape)
                torch.testing.assert_close(model(x)[0], reference(x), atol=2e-6, rtol=2e-5)

    def test_index_and_augmentations_resume_independently_of_prefetch(self):
        with tempfile.TemporaryDirectory() as directory:
            c, _ = write_fixture(Path(directory), "dax")
            dataset = IndexedSpineDataset(c)
            uninterrupted = list(StreamBatchSampler(7, 2, 1, 2, 42, 0, 6))
            resumed = list(StreamBatchSampler(7, 2, 1, 2, 42, 3, 6))
            self.assertEqual(uninterrupted[3:], resumed)
            first, second = dataset[uninterrupted[3][0]], dataset[resumed[0][0]]
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            dataset.files = dataset.files.astype("S")
            for a, b in zip(first, dataset[resumed[0][0]]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            # Across ranks every global stream position is assigned exactly once.
            positions = [p for rank in range(2) for batch in StreamBatchSampler(7, 2, rank, 2, 42, 0, 6) for _, p in batch]
            self.assertEqual(sorted(positions), list(range(24)))

    def test_chexfound_reference_output_parity_with_registers_and_swiglu(self):
        package = types.ModuleType("chexfound")
        source_root = ROOT / "reference_code/baseline/CheXFound/CheXFound-main/chexfound"
        package.__path__ = [str(source_root)]
        spec = importlib.util.spec_from_file_location("_chexfound_reference_test", source_root / "models/vision_transformer.py")
        reference_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"chexfound": package}), patch.dict(os.environ, {"XFORMERS_DISABLED": "1"}):
            spec.loader.exec_module(reference_module)
            reference = reference_module.DinoVisionTransformer(img_size=32, patch_size=8, embed_dim=24,
                depth=2, num_heads=3, init_values=1e-5, ffn_layer="swiglufused", block_chunks=0, num_register_tokens=4).eval()
        model = VisionTransformer(dim=24, depth=2, heads=3, image_size=32, patch_size=8,
                                  registers=4, swiglu=True, layerscale=True, masked=True).eval()
        model.load_state_dict(reference.state_dict(), strict=True)
        with torch.no_grad():
            for shape in ((2, 3, 32, 32), (2, 3, 32, 48)):
                x = torch.randn(shape)
                expected = reference.forward_features(x)
                cls, patches = model(x)
                torch.testing.assert_close(cls, expected["x_norm_clstoken"], atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(patches, expected["x_norm_patchtokens"], atol=2e-6, rtol=2e-5)

    def test_public_initialization_is_strict_and_does_not_restore_old_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            for method in ("dax", "chexfound", "medical_mae"):
                c, _ = write_fixture(Path(directory), method)
                model = build_network(c, device="meta")
                audit = initialize(model, c)
                self.assertTrue(audit["all_parameters_loaded"])
                state = load_tensor_checkpoint(c["initialization"])
                component = "model" if method == "medical_mae" else "teacher"
                state[component].pop(next(iter(state[component])))
                atomic_torch_save(state, c["initialization"])
                with self.assertRaisesRegex(ValueError, "initialization mismatch"):
                    initialize(build_network(c, device="meta"), c)

    def test_all_methods_train_save_export_and_resume_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method in ("dax", "chexfound", "medical_mae"):
                with self.subTest(method=method):
                    c, config = write_fixture(root, method)
                    before = sha256(c["initialization"])
                    full, resumed = root / f"{method}_full", root / f"{method}_resume"
                    common = ["--config", str(config), "--device", "cpu"]
                    main(common + ["--output-dir", str(full)])
                    main(common + ["--output-dir", str(resumed), "--max-updates", "2"])
                    main(common + ["--output-dir", str(resumed), "--resume", str(resumed / "checkpoint_u00000002.pt")])
                    a = load_tensor_checkpoint(full / "checkpoint_u00000004.pt")
                    b = load_tensor_checkpoint(resumed / "checkpoint_u00000004.pt")
                    self.assertEqual(b["optimizer_updates"], 4)
                    self.assertEqual(b["microbatches"], 8)
                    self.assertEqual(b["images_seen"], 16)
                    for component in ("student", "teacher", "objective"):
                        if a[component] is not None:
                            for key in a[component]:
                                torch.testing.assert_close(a[component][key], b[component][key], rtol=0, atol=0)
                    exported = load_tensor_checkpoint(resumed / "backbone_u00000004.pth")
                    self.assertEqual(exported["metadata"]["method"], method)
                    self.assertFalse(any("decoder_" in k or "head" in k for k in exported["model"]))
                    for key, value in encoder_state(b).items():
                        torch.testing.assert_close(exported["model"][key], value, rtol=0, atol=0)
                    original = public_state(load_tensor_checkpoint(c["initialization"]), method)
                    self.assertTrue(any(not torch.equal(a["student"][k], original[k]) for k in original))
                    self.assertEqual(before, sha256(c["initialization"]))

    def test_safety_dry_run_and_resume_config_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c, config = write_fixture(root, "medical_mae")
            output = root / "run"
            main(["--config", str(config), "--output-dir", str(output), "--dry-run", "--check-initialization"])
            self.assertFalse(output.exists())
            main(["--config", str(config), "--output-dir", str(output), "--device", "cpu", "--max-updates", "2"])
            with self.assertRaises(FileExistsError):
                main(["--config", str(config), "--output-dir", str(output), "--device", "cpu"])
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                main(["--config", str(config), "--output-dir", str(output), "--device", "cpu",
                      "--resume", str(output / "checkpoint_u00000002.pt"), "training.learning_rate=0.001"])
            with self.assertRaises(ValueError):
                output_path("outputs/geotopo_dino/existing")
            with self.assertRaises(ValueError):
                load_config(config, ["training.base_lr=0.1"])
            self.assertEqual(load_config(config, ["training.learning_rate=4e-4"])["training"]["learning_rate"], 4e-4)

    def test_sinkhorn_balancing_and_koleo_gradient(self):
        logits = torch.randn(64, 7)
        targets = sinkhorn(logits, 0.2, iterations=100)
        torch.testing.assert_close(targets.sum(-1), torch.ones(64), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(targets.mean(0), torch.full((7,), 1 / 7), atol=2e-4, rtol=2e-4)
        features = torch.randn(4, 24, requires_grad=True)
        koleo(features, "local").backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0)

    def test_fsdp_adam_initialization_does_not_take_an_optimizer_step(self):
        p = torch.nn.Parameter(torch.randn(3, 4))
        empty = torch.nn.Parameter(torch.empty(0))
        before = p.detach().clone()
        optimizer = torch.optim.AdamW([p, empty], lr=0.01)
        initialize_fsdp_adam_state(optimizer)
        self.assertEqual(float(optimizer.state[p]["step"]), 0)
        self.assertFalse(optimizer.state[p]["exp_avg"].any())
        self.assertNotIn(empty, optimizer.state)
        torch.testing.assert_close(p, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
