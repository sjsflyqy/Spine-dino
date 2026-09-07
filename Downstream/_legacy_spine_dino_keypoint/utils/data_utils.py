"""
数据工具函数（与 zpj_fine_tune/utils/data_utils.py 保持一致）

主要内容：
- make_gaussian_heatmap：生成高斯热图（用于训练目标）
- KeypointDataset：关键点检测数据集（支持 images/labels 分离或扁平目录）
- visualize_keypoints_on_patch：在单张 patch 图像上绘制关键点
- visualize_with_gt：绘制 Pred（绿）+ GT（蓝）+ 误差线（黄），与 zpj_fine_tune 完全一致
- save_results_json：保存 JSON 结果
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


KEYPOINT_ORDER = ("1", "2", "3", "4", "21", "22", "23", "24")


# ─── 热图 ────────────────────────────────────────────────────────────────────

def make_gaussian_heatmap(h: int, w: int, x: float, y: float, sigma: float = 1.5) -> np.ndarray:
    """生成归一化高斯热图，与 zpj_fine_tune 完全一致"""
    if x < 0 or y < 0 or x >= w or y >= h:
        return np.zeros((h, w), dtype=np.float32)
    xv = np.arange(w, dtype=np.float32)
    yv = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xv, yv)
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    return np.exp(-d2 / (2 * sigma * sigma)).astype(np.float32)


# ─── 数据集 ──────────────────────────────────────────────────────────────────

class KeypointDataset(Dataset):
    """
    关键点检测数据集（与 zpj_fine_tune 中的 KeypointDataset 兼容）

    目录结构支持两种形式：
      1. 扁平：图像与 json 在同一目录（*.png / *.jpg + *.json）
      2. 分离：images/ 和 labels/ 子目录

    json 格式：{"points": {"1": [x, y], "2": [x, y], ...}}
    关键点坐标为 patch 像素坐标（相对于裁剪后的 patch）。
    """

    def __init__(
        self,
        data_root: str,
        transform=None,
        input_size: int = 224,
        heatmap_size: int = 64,
        sigma: float = 1.5,
    ):
        self.root = Path(data_root)
        self.transform = transform
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.samples: List[Tuple[Path, dict]] = []

        images_dir = self.root / "images"
        labels_dir = self.root / "labels"

        if images_dir.exists() and labels_dir.exists():
            for img_p in sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg")):
                json_p = labels_dir / img_p.with_suffix(".json").name
                if json_p.exists():
                    with open(json_p, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if "points" in meta and len(meta["points"]) > 0:
                        self.samples.append((img_p, meta))
        else:
            for img_p in sorted(self.root.glob("*.png")) + sorted(self.root.glob("*.jpg")):
                json_p = img_p.with_suffix(".json")
                if json_p.exists():
                    with open(json_p, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if "points" in meta and len(meta["points"]) > 0:
                        self.samples.append((img_p, meta))

        print(f"加载了 {len(self.samples)} 个样本（来自 {data_root}）")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, meta = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        if self.transform is not None:
            pixel_values = self.transform(img)     # (3, H, W)
            _, H, W = pixel_values.shape
        else:
            from torchvision import transforms
            t = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
            ])
            pixel_values = t(img)
            H = W = self.input_size

        scale_x = W / float(orig_w)
        scale_y = H / float(orig_h)

        points = meta.get("points", {})
        heatmaps = np.zeros((8, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        # gt_coords 中用 (-1.0, -1.0) 表示"该关键点缺失"
        # 不使用 None，避免 DataLoader default_collate 无法处理 None 值
        gt_coords: List[Tuple[float, float]] = []

        for i, k in enumerate(KEYPOINT_ORDER):
            if k not in points:
                gt_coords.append((-1.0, -1.0))
                continue
            x_patch, y_patch = points[k]
            x_proc = x_patch * scale_x
            y_proc = y_patch * scale_y
            x_h = x_proc * (self.heatmap_size / W)
            y_h = y_proc * (self.heatmap_size / H)
            gt_coords.append((x_h, y_h))
            if 0 <= x_h < self.heatmap_size and 0 <= y_h < self.heatmap_size:
                heatmaps[i] = make_gaussian_heatmap(self.heatmap_size, self.heatmap_size, x_h, y_h, self.sigma)

        return pixel_values, torch.from_numpy(heatmaps), gt_coords, str(img_path)


# ─── 可视化 ──────────────────────────────────────────────────────────────────

def visualize_keypoints_on_patch(
    patch: np.ndarray,
    keypoints: Dict[str, Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 4,
    draw_label: bool = True,
) -> np.ndarray:
    """在 patch 图像（BGR）上绘制关键点"""
    vis = patch.copy()
    h, w = vis.shape[:2]
    for kid, (x, y) in keypoints.items():
        x, y = float(x), float(y)
        if not (0 <= x < w and 0 <= y < h):
            continue
        cv2.circle(vis, (int(round(x)), int(round(y))), radius, color, -1)
        if draw_label:
            cv2.putText(vis, str(kid), (int(round(x)) + 5, int(round(y)) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return vis


def visualize_with_gt(
    image: np.ndarray,
    pred_kp: Dict[str, Tuple[float, float]],
    gt_kp: Optional[Dict[str, Tuple[float, float]]] = None,
    pred_color: Tuple[int, int, int] = (0, 255, 0),      # 绿：Pred
    gt_color: Tuple[int, int, int] = (255, 0, 0),        # 蓝：GT
    line_color: Tuple[int, int, int] = (0, 255, 255),    # 黄：误差线
    draw_label: bool = False,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    绘制预测点（绿）+ GT（蓝）+ 误差线（黄），与 zpj_fine_tune 完全一致
    """
    vis = image.copy()
    h, w = vis.shape[:2]
    has_gt = gt_kp is not None and len(gt_kp) > 0

    # 1) 误差线（先画，不遮挡点）
    if has_gt:
        for kid in set(pred_kp.keys()) & set(gt_kp.keys()):
            px, py = float(pred_kp[kid][0]), float(pred_kp[kid][1])
            gx, gy = float(gt_kp[kid][0]), float(gt_kp[kid][1])
            if (0 <= px < w and 0 <= py < h and 0 <= gx < w and 0 <= gy < h):
                cv2.line(vis, (int(round(px)), int(round(py))),
                         (int(round(gx)), int(round(gy))), line_color, 2, cv2.LINE_AA)

    # 2) GT 点（蓝）
    if has_gt:
        for kid, (x, y) in gt_kp.items():
            x, y = float(x), float(y)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(vis, (int(round(x)), int(round(y))), 4, gt_color, -1)
                if draw_label:
                    cv2.putText(vis, str(kid), (int(round(x)) + 5, int(round(y)) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, gt_color, 1, cv2.LINE_AA)

    # 3) Pred 点（绿）
    for kid, (x, y) in pred_kp.items():
        x, y = float(x), float(y)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(vis, (int(round(x)), int(round(y))), 4, pred_color, -1)
            if draw_label:
                cv2.putText(vis, str(kid), (int(round(x)) + 5, int(round(y)) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, pred_color, 1, cv2.LINE_AA)

    # 4) 图例
    cv2.putText(vis, "Pred", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, pred_color, 2)
    if has_gt:
        cv2.putText(vis, "GT", (70, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, gt_color, 2)
        cv2.putText(vis, "Error", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 2)

    if output_path:
        cv2.imwrite(output_path, vis)

    return vis


# ─── 保存 ────────────────────────────────────────────────────────────────────

def save_results_json(results: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {output_path}")
