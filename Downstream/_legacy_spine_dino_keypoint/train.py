"""
Spine-DINO 关键点检测训练脚本

训练策略：冻结 backbone，只训练 HeatmapHead（Linear Probing）
损失函数：MSE（与 zpj_fine_tune 一致，方便对比）
同时监控 MAE（热图级别），保存 Val MSE 最优模型

用法示例：
    python train.py \
        --backbone-weights /path/to/spine_dino_backbone.pth \
        --train-dir /path/to/train_data \
        --val-dir   /path/to/val_data \
        --output    checkpoints/spine_dino_kp_best.pth \
        --epochs    50 \
        --batch-size 16

对于服务器（nohup 后台运行）：
    nohup python train.py ... > train.log 2>&1 &
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 让当前目录下的模块可以被导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SpineDINOHeatmapModel, create_spine_dino_transforms
from utils.data_utils import KeypointDataset


def build_loaders(args):
    tf = create_spine_dino_transforms(args.input_size)

    train_ds = KeypointDataset(args.train_dir, transform=tf,
                               input_size=args.input_size,
                               heatmap_size=args.heatmap_size,
                               sigma=args.sigma)
    if len(train_ds) == 0:
        print(f"[错误] 训练集为空: {args.train_dir}")
        sys.exit(1)

    if args.val_dir:
        val_ds = KeypointDataset(args.val_dir, transform=tf,
                                 input_size=args.input_size,
                                 heatmap_size=args.heatmap_size,
                                 sigma=args.sigma)
        if len(val_ds) == 0:
            print("警告: 验证集为空，从训练集划分 10%")
            val_ds = None
    else:
        val_ds = None

    if val_ds is None:
        val_n = max(1, int(len(train_ds) * 0.1))
        train_n = len(train_ds) - val_n
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_n, val_n])

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    print(f"训练集: {len(train_ds)} 样本 | 验证集: {len(val_ds)} 样本")
    return train_loader, val_loader


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Spine-DINO 关键点检测训练")
    print("=" * 60)
    print(f"设备         : {device}")
    print(f"Backbone     : {args.backbone_weights or '随机初始化'}")
    print(f"冻结 Backbone: {not args.unfreeze_backbone}")
    print(f"训练数据     : {args.train_dir}")
    print(f"验证数据     : {args.val_dir or '（从训练集划分）'}")
    print(f"批次大小     : {args.batch_size}")
    print(f"训练轮数     : {args.epochs}")
    print(f"学习率       : {args.lr}")
    print(f"输出路径     : {args.output}")
    print("=" * 60)

    train_loader, val_loader = build_loaders(args)

    model = SpineDINOHeatmapModel(
        backbone_weights=args.backbone_weights,
        num_out=8,
        heatmap_size=args.heatmap_size,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)

    # 只优化需要梯度的参数
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n可训练参数量: {sum(p.numel() for p in params):,}")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()

    best_val_mse = float("inf")
    best_epoch = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("\n开始训练...\n")
    for epoch in range(args.epochs):
        # ── 训练 ──
        model.train()
        train_mse_sum = train_mae_sum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for pixel_values, target_heatmaps, _, _ in pbar:
            pixel_values = pixel_values.to(device)
            target_heatmaps = target_heatmaps.to(device)

            optimizer.zero_grad()
            pred = model(pixel_values)
            mse = mse_fn(pred, target_heatmaps)
            mse.backward()
            optimizer.step()

            with torch.no_grad():
                mae = mae_fn(pred, target_heatmaps)

            n = pixel_values.size(0)
            train_mse_sum += mse.item() * n
            train_mae_sum += mae.item() * n
            pbar.set_postfix(mse=f"{mse.item():.6f}", mae=f"{mae.item():.6f}")

        n_train = len(train_loader.dataset)
        avg_train_mse = train_mse_sum / n_train
        avg_train_mae = train_mae_sum / n_train

        # ── 验证 ──
        model.eval()
        val_mse_sum = val_mae_sum = 0.0
        with torch.no_grad():
            for pixel_values, target_heatmaps, _, _ in val_loader:
                pixel_values = pixel_values.to(device)
                target_heatmaps = target_heatmaps.to(device)
                pred = model(pixel_values)
                n = pixel_values.size(0)
                val_mse_sum += mse_fn(pred, target_heatmaps).item() * n
                val_mae_sum += mae_fn(pred, target_heatmaps).item() * n

        n_val = len(val_loader.dataset)
        avg_val_mse = val_mse_sum / n_val
        avg_val_mae = val_mae_sum / n_val

        scheduler.step(avg_val_mse)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train  MSE: {avg_train_mse:.6f} | MAE: {avg_train_mae:.6f}")
        print(f"  Val    MSE: {avg_val_mse:.6f} | MAE: {avg_val_mae:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            best_epoch = epoch + 1
            torch.save({
                "epoch": best_epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_mse": avg_train_mse,
                "train_mae": avg_train_mae,
                "val_mse": avg_val_mse,
                "val_mae": avg_val_mae,
                "args": vars(args),
            }, args.output)
            print(f"  ✓ 保存最佳模型 (Val MSE: {best_val_mse:.6f})")

        print("-" * 60)

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最佳: Epoch {best_epoch}  Val MSE: {best_val_mse:.6f}")
    print(f"模型保存: {args.output}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Spine-DINO 关键点检测训练")
    parser.add_argument("--backbone-weights", type=str, default="",
                        help="spine-dino backbone 权重路径（extract_backbone.py 的输出）")
    parser.add_argument("--train-dir", type=str, required=True, help="训练数据目录")
    parser.add_argument("--val-dir", type=str, default=None, help="验证数据目录（可选）")
    parser.add_argument("--output", type=str, default="checkpoints/spine_dino_kp_best.pth",
                        help="最佳模型保存路径")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=224, help="输入图像大小（224 对应 DINOv2/14）")
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.5, help="高斯热图标准差")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--unfreeze-backbone", action="store_true",
                        help="解冻 backbone 进行全参微调（默认冻结）")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
