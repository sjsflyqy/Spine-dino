"""
Spine-DINO 关键点检测评估脚本

评估指标（与 zpj_fine_tune 对齐，可直接对比）：
  - MSE   : 热图级别均方误差（与 zpj_fine_tune/evaluate_model.py 完全一致）
  - MAE   : 热图级别平均绝对误差（与 zpj_fine_tune/train_raddino.py 训练监控一致）
  - Mean Distance : 关键点坐标欧氏距离（像素，在 heatmap_size×heatmap_size 上度量）
  - Normalized Distance : Mean Distance / heatmap_size × 100%

可视化：
  - 绿色点：预测点 Pred
  - 蓝色点：真实点 GT
  - 黄色线：误差连线（与 zpj_fine_tune/utils/data_utils.py 完全一致）
  - 每个样本保存一张可视化图像到 --vis-dir

用法示例：
    python evaluate.py \
        --model-path checkpoints/spine_dino_kp_best.pth \
        --backbone-weights /path/to/spine_dino_backbone.pth \
        --test-dir  /path/to/test_data \
        --vis-dir   vis_results/
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SpineDINOHeatmapModel, heatmaps_to_points, create_spine_dino_transforms, KEYPOINT_ORDER
from utils.data_utils import KeypointDataset, visualize_with_gt, save_results_json


# ─────────────────────────────────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_batch_metrics(
    pred_hm: torch.Tensor,
    gt_hm: torch.Tensor,
    gt_coords_list,
    heatmap_size: int,
) -> dict:
    """
    计算一个 batch 的 MSE / MAE / Mean Distance。

    Args:
        pred_hm: (B, 8, H, W) 预测热图
        gt_hm  : (B, 8, H, W) GT 热图
        gt_coords_list: DataLoader 转置后的结构，索引为 [k][b]，即第 k 个关键点的第 b 个样本
                        值为 tensor([x, y])，(-1,-1) 表示缺失
        heatmap_size: 热图边长
    """
    mse = nn.MSELoss()(pred_hm, gt_hm).item()
    mae = nn.L1Loss()(pred_hm, gt_hm).item()

    distances = []
    B = pred_hm.shape[0]
    for k in range(8):
        for b in range(B):
            gx, gy = float(gt_coords_list[k][b][0]), float(gt_coords_list[k][b][1])
            if gx < 0:   # (-1,-1) 表示缺失
                continue
            pred_np = pred_hm[b, k].cpu().numpy()
            py, px = np.unravel_index(pred_np.argmax(), pred_np.shape)
            dist = float(np.sqrt((px - gx) ** 2 + (py - gy) ** 2))
            distances.append(dist)

    return {"mse": mse, "mae": mae, "distances": distances}


# ─────────────────────────────────────────────────────────────────────────────
# 主评估逻辑
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Spine-DINO 关键点检测评估")
    print("=" * 60)
    print(f"设备       : {device}")
    print(f"测试数据   : {args.test_dir}")
    print(f"模型路径   : {args.model_path}")
    print(f"可视化目录 : {args.vis_dir or '不保存'}")
    print("=" * 60)

    # ── 数据集 ──
    tf = create_spine_dino_transforms(args.input_size)
    test_ds = KeypointDataset(args.test_dir, transform=tf,
                              input_size=args.input_size,
                              heatmap_size=args.heatmap_size,
                              sigma=args.sigma)
    if len(test_ds) == 0:
        print(f"[错误] 测试集为空: {args.test_dir}")
        return

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ── 模型 ──
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
        print(f"[警告] 缺失键: {missing[:5]}...")
    saved_epoch = ckpt.get("epoch", "?")
    print(f"加载检查点（epoch {saved_epoch}）")

    # ── 评估循环 ──
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    model.eval()
    all_mse, all_mae, all_distances = [], [], []
    per_sample_results = []

    with torch.no_grad():
        for pixel_values, target_hm, gt_coords_list, img_paths in tqdm(test_loader, desc="评估"):
            pixel_values = pixel_values.to(device)
            target_hm = target_hm.to(device)
            pred_hm = model(pixel_values)

            metrics = compute_batch_metrics(pred_hm, target_hm, gt_coords_list, args.heatmap_size)
            all_mse.append(metrics["mse"])
            all_mae.append(metrics["mae"])
            all_distances.extend(metrics["distances"])

            # ── 可视化 ──
            if args.vis_dir:
                B = pixel_values.shape[0]
                for b in range(B):
                    pred_np = pred_hm[b].cpu().numpy()  # (8, H, W)
                    pred_pts = heatmaps_to_points(pred_np, threshold=0.05)

                    gt_kp = {}
                    pred_kp = {}
                    # vis_size = heatmap_size * vis_scale，坐标也 ×vis_scale，确保点落在图内
                    vis_size = args.heatmap_size * args.vis_scale
                    for i, kid in enumerate(KEYPOINT_ORDER):
                        px, py = pred_pts[i]
                        if px > 0 or py > 0:
                            pred_kp[kid] = (px * args.vis_scale, py * args.vis_scale)
                        gx = float(gt_coords_list[i][b][0])
                        gy = float(gt_coords_list[i][b][1])
                        if gx >= 0:   # (-1,-1) 表示缺失
                            gt_kp[kid] = (gx * args.vis_scale, gy * args.vis_scale)

                    # 把 tensor 还原为 BGR 图像，再缩放到 vis_size
                    inp = pixel_values[b].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
                    mean = np.array([0.485, 0.456, 0.406])
                    std  = np.array([0.229, 0.224, 0.225])
                    inp = (inp * std + mean).clip(0, 1)
                    patch_bgr = (inp[:, :, ::-1] * 255).astype(np.uint8)
                    patch_bgr_vis = cv2.resize(patch_bgr, (vis_size, vis_size))

                    vis = visualize_with_gt(
                        patch_bgr_vis, pred_kp, gt_kp if gt_kp else None,
                        draw_label=args.draw_label,
                    )

                    stem = Path(img_paths[b]).stem
                    out_path = os.path.join(args.vis_dir, f"{stem}_vis.jpg")
                    cv2.imwrite(out_path, vis)

                    # 记录单样本指标（坐标仍在热图空间，不乘 vis_scale）
                    sample_distances = []
                    for i, kid in enumerate(KEYPOINT_ORDER):
                        gx = float(gt_coords_list[i][b][0])
                        if gx < 0:
                            continue
                        gy = float(gt_coords_list[i][b][1])
                        if kid in pred_kp:
                            px, py = pred_pts[i]
                            sample_distances.append(float(np.sqrt((px - gx) ** 2 + (py - gy) ** 2)))

                    per_sample_results.append({
                        "image": img_paths[b],
                        "mean_dist_px": float(np.mean(sample_distances)) if sample_distances else None,
                        "pred_keypoints": {k: list(v) for k, v in pred_kp.items()},
                        "gt_keypoints": {k: list(v) for k, v in gt_kp.items()},
                    })

    # ── 汇总 ──
    avg_mse = float(np.mean(all_mse))
    avg_mae = float(np.mean(all_mae))
    avg_dist = float(np.mean(all_distances)) if all_distances else 0.0
    norm_dist = avg_dist / args.heatmap_size * 100

    print("\n" + "=" * 60)
    print("评估结果（与 zpj_fine_tune 对齐，可直接对比）")
    print("=" * 60)
    print(f"测试样本数       : {len(test_ds)}")
    print(f"有效关键点总数   : {len(all_distances)}")
    print(f"MSE  (热图级别)  : {avg_mse:.6f}")
    print(f"MAE  (热图级别)  : {avg_mae:.6f}")
    print(f"Mean Distance    : {avg_dist:.2f} px  "
          f"（热图 {args.heatmap_size}×{args.heatmap_size}，不是原图像素）")
    print(f"Normalized Dist  : {norm_dist:.2f}%")
    print("=" * 60)

    if args.output_json:
        save_results_json({
            "summary": {
                "mse": avg_mse,
                "mae": avg_mae,
                "mean_distance_px": avg_dist,
                "normalized_distance_pct": norm_dist,
                "num_samples": len(test_ds),
                "num_keypoints": len(all_distances),
                "heatmap_size": args.heatmap_size,
            },
            "per_sample": per_sample_results,
        }, args.output_json)


def main():
    parser = argparse.ArgumentParser(description="Spine-DINO 关键点检测评估")
    parser.add_argument("--model-path", type=str, required=True, help="训练好的模型权重路径")
    parser.add_argument("--backbone-weights", type=str, default="",
                        help="spine-dino backbone 权重（若已嵌入 model-path 则可不填）")
    parser.add_argument("--test-dir", type=str, required=True, help="测试数据目录")
    parser.add_argument("--vis-dir", type=str, default="", help="可视化输出目录（留空则不保存）")
    parser.add_argument("--output-json", type=str, default="", help="保存结果到 JSON 文件")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--vis-scale", type=int, default=4,
                        help="可视化时将热图坐标放大到原图的倍数（默认 4，即 64×4=256px 显示）")
    parser.add_argument("--draw-label", action="store_true", help="在可视化图上标注关键点 ID")
    args = parser.parse_args()

    if not args.vis_dir:
        args.vis_dir = None
    if not args.output_json:
        args.output_json = None

    evaluate(args)


if __name__ == "__main__":
    main()
