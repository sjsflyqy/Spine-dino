from __future__ import annotations

import torch
from torch import nn
from torchvision.ops import roi_align

from backbone import build_backbone


class VertebraLinearProbe(nn.Module):
    """Frozen visual backbone plus one shared linear T1--S1 classifier.

    Task 09 follows the VertFound ``BOX_TYPE=GT`` identification protocol:
    ground-truth vertebral boxes define regions, and only their anatomical
    labels are predicted. ROIAlign and spatial averaging have no trainable
    parameters; ``classifier`` is the only trainable module.
    """

    def __init__(
        self,
        backbone_name: str,
        backbone_weights: str | None = None,
        *,
        num_classes: int = 18,
        roi_size: int = 3,
        adapter: dict | None = None,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(
            backbone_name,
            backbone_weights,
            freeze=True,
            adapter=adapter,
        )
        self.roi_size = roi_size
        self.classifier = nn.Linear(self.backbone.embedding_dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.backbone.adapter_enabled:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor, boxes: list[torch.Tensor]):
        # A frozen backbone should neither retain activation graphs nor update
        # stochastic/training state during linear probing.
        context = (
            torch.enable_grad()
            if self.training and self.backbone.adapter_enabled
            else torch.no_grad()
        )
        with context:
            features = self.backbone(pixel_values).feature_map

        feature_h, feature_w = features.shape[-2:]
        rois = []
        counts = []
        for batch_index, image_boxes in enumerate(boxes):
            counts.append(len(image_boxes))
            if len(image_boxes) == 0:
                continue
            scaled = image_boxes.to(features.device, features.dtype).clone()
            scaled[:, (0, 2)] *= feature_w
            scaled[:, (1, 3)] *= feature_h
            indices = torch.full(
                (len(scaled), 1), batch_index, device=features.device,
                dtype=features.dtype,
            )
            rois.append(torch.cat((indices, scaled), dim=1))

        if not rois:
            empty = features.new_empty((0, self.classifier.out_features))
            return {"logits": empty, "counts": counts}

        pooled = roi_align(
            features,
            torch.cat(rois, dim=0),
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        ).mean(dim=(-2, -1))
        return {"logits": self.classifier(pooled), "counts": counts}
