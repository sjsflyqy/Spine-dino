"""Fully automatic SpineFM-style prediction. No ground truth input is accepted."""
from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import functional as TF

from dataset import classifier_input, make_crop, restore_probability


class SpineFMPipeline:
    def __init__(self, segmenter, auxiliaries, device, config=None):
        self.segmenter, self.aux, self.device = segmenter, auxiliaries, device
        self.config = config or {}
        self.mask_threshold = float(self.config.get("mask_threshold", 0.9))
        self.patch_size = int(self.config.get("patch_size", 300))
        required_threshold = getattr(self.aux.get("detector"), "csxa_mask_threshold", None)
        if required_threshold is not None and float(self.config.get("detector_mask_threshold", required_threshold)) != required_threshold:
            raise ValueError("Official SpineFM detector requires detector_mask_threshold=0.9")
        self.detector_mask_threshold = float(self.config.get("detector_mask_threshold", required_threshold or 0.5))

    @torch.no_grad()
    def segment(self, image, point):
        x, prompt, _, box = make_crop(image, point, self.patch_size, self.segmenter.input_size)
        amp = self.device.type == "cuda" and self.config.get("mixed_precision", True)
        dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=amp):
            logits, _ = self.segmenter(x[None].to(self.device), prompt[None].to(self.device))
        probability = restore_probability(logits[0, 0].float().sigmoid(), box, image.size).cpu().numpy()
        mask = probability > self.mask_threshold
        if not mask.any():
            return None
        # Weighted centroid of the target foreground; avoids low-probability background drift.
        weights = probability * mask
        ys, xs = np.nonzero(mask)
        center = (float((xs*weights[ys, xs]).sum()/weights.sum()),
                  float((ys*weights[ys, xs]).sum()/weights.sum()))
        return mask, center

    @torch.no_grad()
    def predict(self, image):
        self.segmenter.eval()
        output = self.aux["detector"]([TF.to_tensor(image).to(self.device)])[0]
        candidates = []
        for score, probability in zip(output["scores"].cpu().numpy(), output["masks"][:, 0].cpu().numpy()):
            if score <= float(self.config.get("detector_score_threshold", 0.6)):
                continue
            y, x = np.nonzero(probability > self.detector_mask_threshold)
            if len(x):
                candidates.append((float(score), (float(x.mean()), float(y.mean()))))
        candidates.sort(key=lambda x: x[1][1])
        trace = {"candidates": len(candidates), "stops": [], "seeds": []}
        if len(candidates) >= 3:
            start = max(range(len(candidates)-2), key=lambda j: sum(x[0] for x in candidates[j:j+3]))
            candidates = candidates[start:start+3]
        masks, centers = [], []
        for _, point in candidates:
            prediction = self.segment(image, point)
            trace["seeds"].append(list(point))
            if prediction is not None:
                mask, center = prediction
                masks.append(mask)
                centers.append(center)
        if len(centers) < 3:
            trace["stops"].append("fewer_than_three_valid_seeds")
            return masks, trace
        # The same three refined centers initialize both directions.
        for initial in (centers[:3][::-1], centers[:3]):
            chain = list(initial)
            for _ in range(int(self.config.get("max_steps_per_direction", 8))):
                normalized = np.asarray(chain[-3:], np.float32) / np.asarray(image.size, np.float32)
                next_point = self.aux["point_predictor"](torch.from_numpy(normalized.reshape(1, 6)).to(self.device))[0]
                point = next_point.cpu().numpy() * np.asarray(image.size)
                if not np.isfinite(point).all() or not (0 <= point[0] < image.width and 0 <= point[1] < image.height):
                    trace["stops"].append("outside_image")
                    break
                prediction = self.segment(image, point)
                if prediction is None:
                    trace["stops"].append("empty_mask")
                    break
                mask, center = prediction
                overlap = max(np.count_nonzero(mask & previous) / max(np.count_nonzero(mask | previous), 1)
                              for previous in masks)
                if overlap > float(self.config.get("stop_iou", 0.1)):
                    trace["stops"].append("duplicate_instance")
                    break
                profile = getattr(self.aux["classifier"], "csxa_preprocessing", "local")
                patch = classifier_input(image, center, profile)
                logits = self.aux["classifier"](patch[None].to(self.device))
                if int(logits.argmax(1)) == 0:
                    trace["stops"].append("background")
                    break
                masks.append(mask)
                chain.append(center)
            else:
                trace["stops"].append("iteration_limit")
        return masks, trace
