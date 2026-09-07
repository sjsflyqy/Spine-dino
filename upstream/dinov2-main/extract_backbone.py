"""
从 DINOv2 训练检查点中提取 teacher backbone 权重。

DINOv2 训练保存的检查点包含完整的 SSLMetaArch（student + teacher + heads），
此脚本提取 teacher backbone 并保存为干净的 state_dict，方便下游任务使用。

用法：
    python extract_backbone.py \
        --checkpoint spine_dino_output/model_final.rank_0.pth \
        --output spine_dino_backbone.pth
"""

import argparse
from pathlib import Path

import torch


def extract(checkpoint_path: str, output_path: str, source: str = "teacher"):
    print(f"加载检查点: {checkpoint_path}")

    # weights_only=False 兼容包含 numpy scalar 的旧格式检查点
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if source in ckpt and isinstance(ckpt[source], dict):
        # Evaluation checkpoints are saved as {"teacher": {"backbone.*": ...}}.
        state = ckpt[source]
        prefixes = (
            "backbone._fsdp_wrapped_module.",
            "backbone.",
        )
    elif "model" in ckpt:
        state = ckpt["model"]
        prefixes = (
            f"{source}.backbone._fsdp_wrapped_module.",
            f"{source}.backbone.",
        )
    else:
        state = ckpt
        prefixes = (
            f"{source}.backbone._fsdp_wrapped_module.",
            f"{source}.backbone.",
            "backbone._fsdp_wrapped_module.",
            "backbone.",
        )

    all_keys = list(state.keys())
    print(f"检查点中共 {len(all_keys)} 个键，前20个：")
    for k in all_keys[:20]:
        print(f"  {k}")

    # 尝试两种键名格式（有无 FSDP 包装）
    for prefix in prefixes:
        backbone_sd = {
            k[len(prefix):]: v
            for k, v in state.items()
            if k.startswith(prefix)
        }
        if backbone_sd:
            break

    if not backbone_sd:
        raise RuntimeError(
            f"未找到 {source} backbone 权重，请检查上方键名。"
        )

    print(f"\n提取了 {len(backbone_sd)} 个 {source} backbone 权重")
    print("示例键名：")
    for k in list(backbone_sd.keys())[:8]:
        print(f"  {k}: {backbone_sd[k].shape}")

    # 只保存 backbone，大幅节省磁盘空间
    Path(output_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone_sd, output_path)
    print(f"\n已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="从 DINOv2 检查点提取 backbone 权重")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练检查点路径")
    parser.add_argument("--output", type=str, default="spine_dino_backbone.pth", help="输出路径")
    parser.add_argument("--source", type=str, default="teacher",
                        choices=["teacher", "student"],
                        help="提取 teacher 还是 student（teacher 通常更好）")
    args = parser.parse_args()
    extract(args.checkpoint, args.output, args.source)


if __name__ == "__main__":
    main()
