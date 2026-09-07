"""
Spine-DINO 关键点检测模型

使用 DINOv2 ViT-Base/14 作为 backbone（加载 spine-dino 预训练权重），
HeatmapHead 与 zpj_fine_tune 中的 RadDINOHeatmapModel 完全一致，
确保评估指标可直接对比。
"""

import math
import sys
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import maximum_filter
from torchvision import transforms

# 将 dinov2-main 加入 sys.path 以便导入 dinov2 模块
_DINOV2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dinov2-main"))
if _DINOV2_ROOT not in sys.path:
    sys.path.insert(0, _DINOV2_ROOT)

from dinov2.models.vision_transformer import vit_base

INPUT_SIZE = 224        # DINOv2 标准输入（必须是 14 的倍数）
HEATMAP_SIZE = 64
NUM_KEYPOINTS = 8
KEYPOINT_ORDER = ("1", "2", "3", "4", "21", "22", "23", "24")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class HeatmapHead(nn.Module):
    """热图预测头——与 zpj_fine_tune/models/rad_dino_model.py 完全一致"""

    def __init__(self, hidden_size=768, out_ch=NUM_KEYPOINTS, upsample_target=HEATMAP_SIZE):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.final = nn.Conv2d(128, out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.ups = upsample_target

    def forward(self, token_feat: torch.Tensor) -> torch.Tensor:
        bsz, tokens, dim = token_feat.shape
        side = int(math.sqrt(tokens))
        if side * side != tokens:
            token_feat = token_feat[:, 1:, :]
            tokens = token_feat.shape[1]
            side = int(math.sqrt(tokens))
        feat = token_feat.permute(0, 2, 1).reshape(bsz, dim, side, side)
        feat = self.relu(self.bn1(self.conv1(feat)))
        feat = nn.functional.interpolate(feat, size=(self.ups // 2, self.ups // 2),
                                         mode="bilinear", align_corners=False)
        feat = self.relu(self.bn2(self.conv2(feat)))
        feat = nn.functional.interpolate(feat, size=(self.ups, self.ups),
                                         mode="bilinear", align_corners=False)
        return self.final(feat)


class SpineDINOHeatmapModel(nn.Module):
    """
    Spine-DINO 关键点检测模型

    Args:
        backbone_weights: spine-dino backbone 权重路径（extract_backbone.py 的输出）
        num_out: 关键点通道数
        heatmap_size: 输出热图尺寸
        freeze_backbone: 是否冻结 backbone（默认 True，即 linear probing）
    """

    def __init__(
        self,
        backbone_weights: str = "",
        num_out: int = NUM_KEYPOINTS,
        heatmap_size: int = HEATMAP_SIZE,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = vit_base(
            patch_size=14,
            img_size=INPUT_SIZE,
            init_values=1e-5,
            block_chunks=0,
            num_register_tokens=0,
        )

        if backbone_weights and os.path.isfile(backbone_weights):
            sd = torch.load(backbone_weights, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"Backbone 权重加载: 缺失 {len(missing)} 个, 未预期 {len(unexpected)} 个")
        else:
            if backbone_weights:
                print(f"[警告] 权重文件不存在: {backbone_weights}，使用随机初始化")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("Backbone 已冻结（仅训练 HeatmapHead）")

        hidden = self.backbone.embed_dim   # 768
        self.head = HeatmapHead(hidden_size=hidden, out_ch=num_out, upsample_target=heatmap_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone.get_intermediate_layers(pixel_values, n=1, return_class_token=True)
        # features 是一个 list: [(patch_tokens, cls_token)]
        patch_tokens, cls_token = features[0]
        # 拼接 CLS + patch tokens → (B, 1+N, D)
        tokens = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        return self.head(tokens)


def create_spine_dino_transforms(input_size: int = INPUT_SIZE):
    """创建与 DINOv2 匹配的图像预处理 transform"""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def heatmaps_to_points(heatmaps: np.ndarray, threshold: float = 0.1) -> List[Tuple[float, float]]:
    """热图转关键点坐标（与 zpj_fine_tune 完全一致）"""
    num_channels, height, width = heatmaps.shape
    points: List[Tuple[float, float]] = []
    for idx in range(num_channels):
        hm = heatmaps[idx]
        max_val = float(np.max(hm))
        if max_val < threshold:
            points.append((0.0, 0.0))
            continue
        mask = hm == maximum_filter(hm, size=3)
        coords = np.argwhere(mask & (hm == max_val))
        if coords.size > 0:
            y, x = coords[0]
        else:
            flat = int(np.argmax(hm))
            y, x = divmod(flat, width)
        points.append((float(x), float(y)))
    return points
