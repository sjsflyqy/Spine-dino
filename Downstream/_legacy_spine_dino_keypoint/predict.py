"""
Spine-DINO 关键点检测纯推理脚本

只需要图片，不需要任何标注文件（JSON）。
关键点会叠加到原始图像上，并保存可视化结果。

用法示例：
    # 对单张图片推理
    python spine_dino_keypoint/predict.py \
        --model-path spine_dino_output/spine_dino_kp_best.pth \
        --input test_image/full_AP_1.jpg \
        --vis-dir vis_predict/

    # 对整个目录推理
    python spine_dino_keypoint/predict.py \
        --model-path spine_dino_output/spine_dino_kp_best.pth \
        --input test_image/ \
        --vis-dir vis_predict/ \
        --draw-label
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SpineDINOHeatmapModel, heatmaps_to_points, create_spine_dino_transforms, KEYPOINT_ORDER

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# 8个关键点的颜色（BGR），方便区分
KP_COLORS = [
    (0, 255, 0),    # 1  绿
    (0, 200, 50),   # 2  绿偏深
    (0, 255, 128),  # 3  青绿
    (0, 230, 80),   # 4  草绿
    (255, 100, 0),  # 21 橙
    (255, 60, 0),   # 22 橙红
    (255, 140, 0),  # 23 橘
    (255, 180, 0),  # 24 金橙
]


def collect_images(input_path: str):
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        imgs = []
        for ext in IMAGE_EXTS:
            imgs.extend(sorted(p.glob(f"*{ext}")))
            imgs.extend(sorted(p.glob(f"*{ext.upper()}")))
        return sorted(set(imgs))
    raise FileNotFoundError(f"路径不存在: {input_path}")


def draw_keypoints_on_image(
    image_bgr: np.ndarray,
    keypoints: dict,
    draw_label: bool = True,
    radius: int = 6,
) -> np.ndarray:
    """将关键点叠加到原始图像上（坐标已映射到原图尺寸）"""
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    for i, kid in enumerate(KEYPOINT_ORDER):
        if kid not in keypoints:
            continue
        x, y = float(keypoints[kid][0]), float(keypoints[kid][1])
        if not (0 <= x < w and 0 <= y < h):
            continue
        color = KP_COLORS[i % len(KP_COLORS)]
        cv2.circle(vis, (int(round(x)), int(round(y))), radius, color, -1)
        cv2.circle(vis, (int(round(x)), int(round(y))), radius + 1, (0, 0, 0), 1)  # 黑色描边
        if draw_label:
            cv2.putText(
                vis, str(kid),
                (int(round(x)) + radius + 2, int(round(y)) - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                vis, str(kid),
                (int(round(x)) + radius + 2, int(round(y)) - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
            )
    # 图例
    legend_y = 30
    for i, kid in enumerate(KEYPOINT_ORDER):
        color = KP_COLORS[i % len(KP_COLORS)]
        label = f"KP {kid}"
        cv2.putText(vis, label, (10, legend_y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (10, legend_y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def predict_single(model, img_path: Path, transform, device, heatmap_size: int, threshold: float):
    """对单张图片推理，返回关键点坐标（原图坐标系）"""
    img_pil = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img_pil.size

    pixel_values = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_hm = model(pixel_values)  # (1, 8, H, W)

    pred_np = pred_hm[0].cpu().numpy()  # (8, heatmap_size, heatmap_size)
    pts_hm = heatmaps_to_points(pred_np, threshold=threshold)

    # 热图坐标 → 原图坐标
    scale_x = orig_w / heatmap_size
    scale_y = orig_h / heatmap_size
    keypoints = {}
    for i, kid in enumerate(KEYPOINT_ORDER):
        px_hm, py_hm = pts_hm[i]
        if px_hm > 0 or py_hm > 0:
            keypoints[kid] = (px_hm * scale_x, py_hm * scale_y)

    return keypoints, pred_np, orig_w, orig_h


def predict(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Spine-DINO 关键点检测推理")
    print("=" * 60)
    print(f"设备       : {device}")
    print(f"输入       : {args.input}")
    print(f"模型路径   : {args.model_path}")
    print(f"输出目录   : {args.vis_dir}")
    print(f"热图阈值   : {args.threshold}")
    print("=" * 60)

    # ── 收集图片 ──
    img_paths = collect_images(args.input)
    if not img_paths:
        print(f"[错误] 未找到图片: {args.input}")
        return
    print(f"找到 {len(img_paths)} 张图片")

    # ── 加载模型 ──
    model = SpineDINOHeatmapModel(
        backbone_weights=args.backbone_weights,
        num_out=8,
        heatmap_size=args.heatmap_size,
        freeze_backbone=True,
    ).to(device)

    ckpt = torch.load(args.model_path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[警告] 缺失键（前5个）: {missing[:5]}")
    saved_epoch = ckpt.get("epoch", "?")
    print(f"加载检查点（epoch {saved_epoch}）\n")

    model.eval()
    transform = create_spine_dino_transforms(args.input_size)

    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    # ── 推理循环 ──
    all_results = {}
    for img_path in img_paths:
        print(f"推理: {img_path.name}")
        keypoints, pred_hm_np, orig_w, orig_h = predict_single(
            model, img_path, transform, device, args.heatmap_size, args.threshold
        )

        print(f"  原图尺寸: {orig_w} × {orig_h}")
        print(f"  检测到关键点: {list(keypoints.keys())}")
        for kid, (x, y) in keypoints.items():
            print(f"    KP {kid:>3}: ({x:.1f}, {y:.1f})")

        all_results[str(img_path)] = keypoints

        if args.vis_dir:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                # PIL 读取再转 BGR（处理部分 DICOM/特殊格式）
                img_np = np.array(Image.open(img_path).convert("RGB"))
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            vis = draw_keypoints_on_image(img_bgr, keypoints, draw_label=args.draw_label)
            out_path = os.path.join(args.vis_dir, f"{img_path.stem}_pred.jpg")
            cv2.imwrite(out_path, vis)
            print(f"  保存可视化: {out_path}")

        print()

    print("=" * 60)
    print(f"推理完成，共处理 {len(img_paths)} 张图片")
    if args.vis_dir:
        print(f"可视化结果保存在: {args.vis_dir}")
    print("=" * 60)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Spine-DINO 关键点检测推理（无需标注）")
    parser.add_argument("--model-path", type=str,
                        default="spine_dino_output/spine_dino_kp_best.pth",
                        help="训练好的模型权重路径")
    parser.add_argument("--backbone-weights", type=str, default="",
                        help="Spine-DINO backbone 权重（若已嵌入 model-path 可不填）")
    parser.add_argument("--input", type=str, required=True,
                        help="输入：单张图片路径 或 图片目录")
    parser.add_argument("--vis-dir", type=str, default="vis_predict",
                        help="可视化输出目录（默认 vis_predict/）")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="热图峰值阈值，低于此值认为该关键点不存在（默认 0.05）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--draw-label", action="store_true",
                        help="在可视化图上标注关键点 ID")
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
