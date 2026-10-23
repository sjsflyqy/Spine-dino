"""Read-only indexed spine data and reproducible, resumable sample/augmentation streams."""
from __future__ import annotations

import contextlib
import math
import random

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF, InterpolationMode

from .config import resolve


def relative_name(value):
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


class Blur:
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 2))) if random.random() < self.probability else image


class RandomSharpness:
    def __call__(self, image):
        return TF.adjust_sharpness(image, random.uniform(0.8, 1.2))


class MultiCrop:
    def __init__(self, c):
        cfg, method = c["crops"], c["method"]
        self.local_count = cfg["local_count"] if method != "medical_mae" else 0
        end = [T.ToTensor()]
        if cfg["mean"] is not None:
            end.append(T.Normalize(cfg["mean"], cfg["std"]))
        ratio = tuple(cfg["aspect_ratio"])

        def make(size, scale, view):
            ops = [T.RandomResizedCrop(size, scale=tuple(scale), ratio=ratio,
                                      interpolation=InterpolationMode.BICUBIC, antialias=True), T.RandomHorizontalFlip()]
            if method == "medical_mae":
                return T.Compose(ops + end)
            if method == "dax":
                # Checkpoint vA: square crops, vertical flips, grayscale jitter, sharpness.
                ops += [T.RandomVerticalFlip(), T.RandomApply([T.ColorJitter(0.4, 0.4, 0, 0)], p=0.8)]
                if view == 0:
                    ops.append(RandomSharpness())
            else:
                ops += [T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8), T.RandomGrayscale(p=0.2)]
            ops.append(Blur(1.0 if view == 0 else 0.1 if view == 1 else 0.5))
            if view == 1:
                ops.append(T.RandomSolarize(128, p=0.2))
            return T.Compose(ops + end)

        self.first = make(cfg["global_size"], cfg["global_scale"], 0)
        self.second = make(cfg["global_size"], cfg["global_scale"], 1)
        self.local = make(cfg["local_size"], cfg["local_scale"], 2)
        self.mae = method == "medical_mae"

    def __call__(self, image):
        if self.mae:
            return self.first(image)
        return [self.first(image), self.second(image)] + [self.local(image) for _ in range(self.local_count)]


@contextlib.contextmanager
def sample_randomness(seed):
    python_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            # Do not seed CUDA from loader workers / sample transforms.
            torch.random.default_generator.manual_seed(seed)
            random.seed(seed)
            yield
    finally:
        random.setstate(python_state)


class IndexedSpineDataset(Dataset):
    def __init__(self, c):
        self.root = resolve(c["data"]["root"])
        self.index_path = resolve(c["data"]["index"])
        self.files = np.load(self.index_path, allow_pickle=False, mmap_mode="r")
        if self.files.ndim != 1 or len(self.files) == 0 or self.files.dtype.kind not in {"U", "S"}:
            raise ValueError("Expected a nonempty one-dimensional image-path index")
        self.transform = MultiCrop(c)
        self.seed = c["seed"]
        # Check lexical relative paths without rejecting legitimate existing symlinks.
        from pathlib import PurePosixPath
        for item in self.files:
            path = PurePosixPath(relative_name(item))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Index contains an unsafe relative path: {item}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        index, position = item
        path = self.root / "train" / relative_name(self.files[index])
        with Image.open(path) as image:
            image = image.convert("RGB")
        seed = (self.seed + position * 6364136223846793005) % (2 ** 63 - 1)
        with sample_randomness(seed):
            return self.transform(image)


class StreamBatchSampler(Sampler):
    """Global shuffled stream, sliced by rank; start is a microbatch counter.

    Each item carries its absolute position, so augmentation is independent of
    worker count/prefetch. Resuming does not replay or skip prefetched samples.
    """
    def __init__(self, size, batch_size, rank, world_size, seed, start, stop):
        self.size, self.batch_size, self.rank, self.world_size = size, batch_size, rank, world_size
        self.seed, self.start, self.stop = seed, start, stop

    def __len__(self):
        return max(0, self.stop - self.start)

    def __iter__(self):
        epoch, permutation = -1, None
        for step in range(self.start, self.stop):
            batch = []
            base = (step * self.world_size + self.rank) * self.batch_size
            for position in range(base, base + self.batch_size):
                current_epoch, offset = divmod(position, self.size)
                if current_epoch != epoch:
                    permutation = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + current_epoch))
                    epoch = current_epoch
                batch.append((int(permutation[offset]), position))
            yield batch


def block_masks(batch, side, cfg, device):
    """DINO/iBOT rectangular block masking, independent of existing samplers."""
    masks = torch.zeros(batch, side, side, dtype=torch.bool)
    selected = torch.randperm(batch)[:max(1, int(batch * cfg["mask_sample_probability"]))]
    for i in selected.tolist():
        target = max(1, int(random.uniform(cfg["mask_ratio_min"], cfg["mask_ratio_max"]) * side * side))
        for _ in range(side * side):
            remaining = target - int(masks[i].sum())
            if remaining <= 0:
                break
            area = random.uniform(min(4, remaining), remaining)
            aspect = math.exp(random.uniform(math.log(0.3), math.log(1 / 0.3)))
            h, w = max(1, round(math.sqrt(area * aspect))), max(1, round(math.sqrt(area / aspect)))
            if h > side or w > side:
                continue
            top, left = random.randrange(side - h + 1), random.randrange(side - w + 1)
            box = masks[i, top:top + h, left:left + w]
            added = h * w - int(box.sum())
            if 0 < added <= remaining:
                box.fill_(True)
        if not masks[i].any():
            masks[i, random.randrange(side), random.randrange(side)] = True
    return masks.flatten(1).to(device)
