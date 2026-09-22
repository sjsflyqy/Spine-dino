"""Cached instance GT and synchronized prompt-centered crops."""
from __future__ import annotations

from functools import lru_cache
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from common import LEVELS, read_splits, resolve, sha256

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class CSXAStore:
    def __init__(self, data_root="Downstream/data/task10", cache_root=None, verify=True):
        self.root = resolve(data_root)
        self.cache = resolve(cache_root) if cache_root else self.root / "masks_gt"
        self.manifest_path = self.cache / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        self.splits = read_splits()
        if self.manifest["signature"]["splits"] != self.splits:
            raise ValueError("GT cache split differs from the fixed SpineFM split")
        self.fingerprint = sha256(self.manifest_path)
        if verify:
            for i, row in self.manifest["samples"].items():
                if sha256(self.cache / row["file"]) != row["sha256"]:
                    raise ValueError(f"Corrupted GT: {i}")
                src = self.manifest["signature"]["sources"][i]
                if (sha256(self.root / "datasets" / f"{i}.png") != src["image"] or
                        sha256(self.root / "dataset_json" / f"{i}.json") != src["json"]):
                    raise ValueError(f"Sources changed for {i}; rebuild GT in a new directory")

    @lru_cache(maxsize=8)
    def load(self, identifier):
        with Image.open(self.root / "datasets" / f"{identifier}.png") as image:
            image = image.convert("RGB")
        with np.load(self.cache / self.manifest["samples"][identifier]["file"], allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        if payload["masks"].shape != (5, image.height, image.width) or tuple(payload["levels"]) != LEVELS:
            raise ValueError(f"Invalid GT cache shape/levels: {identifier}")
        return image, payload


def crop_box(center, patch_size):
    left, top = np.floor(np.asarray(center) - patch_size / 2).astype(int)
    return int(left), int(top), int(left + patch_size), int(top + patch_size)


def image_tensor(image, input_size, augment=False):
    image = image.resize((input_size, input_size), Image.Resampling.BICUBIC)
    if augment:
        image = TF.adjust_brightness(image, float(np.random.uniform(0.9, 1.1)))
        image = TF.adjust_contrast(image, float(np.random.uniform(0.9, 1.1)))
    return TF.normalize(TF.to_tensor(image), MEAN, STD)


def classifier_input(image, center, profile="local"):
    if profile == "spinefm_official":
        # Exact reference generate_patch(256) -> classify resize sequence:
        # int crop origin, PIL bilinear 1024, ToTensor, tensor nearest 224.
        # No ImageNet normalization is applied in the released classifier code.
        left, top = int(center[0] - 128), int(center[1] - 128)
        patch = TF.crop(image, top, left, 256, 256)
        patch = TF.resize(patch, [1024, 1024], interpolation=TF.InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(patch)
        return torch.nn.functional.interpolate(tensor[None], (224, 224), mode="nearest")[0]
    if profile != "local":
        raise ValueError(f"Unknown classifier preprocessing: {profile}")
    return image_tensor(image.crop(crop_box(center, 256)), 224)


def make_crop(image, center, patch_size, input_size, mask=None, target_size=256, augment=False):
    box = crop_box(center, patch_size)
    cropped = image.crop(box)  # PIL zero-pads out-of-image pixels.
    x = image_tensor(cropped, input_size, augment)
    # Pixel-center convention: same crop origin and pixel-center resize as masks.
    prompt = ((np.asarray(center) - np.asarray(box[:2]) + 0.5) * input_size / patch_size - 0.5)
    y = None
    if mask is not None:
        patch = Image.fromarray(mask).crop(box).resize((target_size, target_size), Image.Resampling.NEAREST)
        y = torch.from_numpy(np.asarray(patch).copy()).float().unsqueeze(0)
    return x, torch.as_tensor(prompt, dtype=torch.float32), y, box


def restore_probability(probability, box, size):
    """Paste a local probability map into (height,width); outside crop is zero."""
    width, height = size
    left, top, right, bottom = box
    local = torch.nn.functional.interpolate(probability.reshape(1, 1, *probability.shape[-2:]),
                                            size=(bottom-top, right-left), mode="bilinear", align_corners=False)[0, 0]
    result = torch.zeros((height, width), dtype=local.dtype, device=local.device)
    x0, y0, x1, y1 = max(0, left), max(0, top), min(width, right), min(height, bottom)
    if x1 > x0 and y1 > y0:
        result[y0:y1, x0:x1] = local[y0-top:y1-top, x0-left:x1-left]
    return result


class SegmentationDataset(Dataset):
    def __init__(self, store, split, input_size=896, patch_size=300, target_size=256, repeats=1):
        self.store, self.split = store, split
        self.input_size, self.patch_size, self.target_size = input_size, patch_size, target_size
        self.samples = [(i, j) for i in store.splits[split] for j in range(5)] * repeats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        identifier, j = self.samples[index]
        image, gt = self.store.load(identifier)
        center = gt["centroids"][j]
        if self.split == "train":
            y, x = np.nonzero(gt["masks"][j])
            k = np.random.randint(len(x))
            center = (x[k], y[k])
        x, point, target, _ = make_crop(image, center, self.patch_size, self.input_size,
                                       gt["masks"][j], self.target_size, self.split == "train")
        return dict(image=x, point=point, mask=target, id=identifier, level=j)


class DetectionDataset(Dataset):
    def __init__(self, store, split):
        self.store, self.ids = store, store.splits[split]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image, gt = self.store.load(self.ids[index])
        masks = torch.from_numpy(gt["masks"].copy())
        return TF.to_tensor(image), dict(boxes=torch.from_numpy(gt["boxes"].copy()),
                                        labels=torch.ones(5, dtype=torch.int64), masks=masks,
                                        image_id=torch.tensor([index]), area=masks.flatten(1).sum(1).float(),
                                        iscrowd=torch.zeros(5, dtype=torch.int64))


def detection_collate(batch):
    return tuple(zip(*batch))


class ClassificationDataset(Dataset):
    def __init__(self, store, split, seed=42):
        self.store, self.ids, self.split, self.seed = store, store.splits[split], split, seed

    def __len__(self):
        return len(self.ids) * 10

    def __getitem__(self, index):
        image, gt = self.store.load(self.ids[index // 10])
        j = index % 10
        rng = np.random if self.split == "train" else np.random.RandomState(self.seed + index)
        if j < 5:
            mask = gt["masks"][j]
            if self.split == "train":
                y, x = np.nonzero(mask)
                k = rng.randint(len(x))
                center = (x[k], y[k])
            else:
                center = gt["centroids"][j]
            label = 1
        else:
            # A negative prompt lies outside every annotated C3-C7 instance.
            y, x = np.nonzero(~gt["masks"].any(axis=0))
            k = rng.randint(len(x))
            center, label = (x[k], y[k]), 0
        patch = image.crop(crop_box(center, 256))
        return image_tensor(patch, 224, self.split == "train"), torch.tensor(label)


class PointDataset(Dataset):
    def __init__(self, store, split):
        self.samples = []
        for identifier in store.splits[split]:
            _, gt = store.load(identifier)
            points = gt["centroids"] / gt["size"]
            for chain in (points, points[::-1]):
                for j in (0, 1):
                    self.samples.append((chain[j:j+3].reshape(6).copy(), chain[j+3].copy()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y = self.samples[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
