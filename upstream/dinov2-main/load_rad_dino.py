"""
RAD-DINO权重转换脚本
将 microsoft/rad-dino 的 HuggingFace 格式权重转换为 DINOv2 格式，
并保存为可被 DINOv2 训练流程直接加载的 .pth 文件。

关键差异：
  HuggingFace ViT：Q/K/V 是三个独立的线性层
  DINOv2 ViT：    Q/K/V 合并为一个 attn.qkv 层（dim 0 顺序为 Q->K->V）
"""

import torch
from safetensors.torch import load_file
from dinov2.models.vision_transformer import vit_base
import argparse


def convert_rad_dino_to_dinov2(state_dict: dict) -> dict:
    """
    将 RAD-DINO（HuggingFace transformers）格式的 state_dict 转换为 DINOv2 格式。

    RAD-DINO 实际键名结构（混合风格）：
      外层: HuggingFace 的 encoder.layer.{i} 结构
      内层: 已使用 DINOv2 风格的子键名（norm1/norm2/mlp.fc1/layer_scale1）

    完整映射规则：
      embeddings.cls_token                              → cls_token
      embeddings.mask_token                             → mask_token
      embeddings.position_embeddings                    → pos_embed
      embeddings.patch_embeddings.projection.*          → patch_embed.proj.*
      layernorm.*                                       → norm.*
      encoder.layer.{i}.attention.attention.query  ┐
      encoder.layer.{i}.attention.attention.key    ├── 合并 → blocks.{i}.attn.qkv
      encoder.layer.{i}.attention.attention.value  ┘
      encoder.layer.{i}.attention.output.dense.*        → blocks.{i}.attn.proj.*
      encoder.layer.{i}.norm1.*                         → blocks.{i}.norm1.*
      encoder.layer.{i}.norm2.*                         → blocks.{i}.norm2.*
      encoder.layer.{i}.mlp.*                           → blocks.{i}.mlp.*
      encoder.layer.{i}.layer_scale1.lambda1            → blocks.{i}.ls1.gamma
      encoder.layer.{i}.layer_scale2.lambda1            → blocks.{i}.ls2.gamma
      (兼容) encoder.layer.{i}.layernorm_before.*       → blocks.{i}.norm1.*
      (兼容) encoder.layer.{i}.layernorm_after.*        → blocks.{i}.norm2.*
      (兼容) encoder.layer.{i}.intermediate.dense.*     → blocks.{i}.mlp.fc1.*
      (兼容) encoder.layer.{i}.output.dense.*           → blocks.{i}.mlp.fc2.*
    """
    new_state_dict = {}
    # 按层收集 Q/K/V 权重，最后再合并
    qkv_parts: dict = {}   # { layer_num: {'q_w', 'k_w', 'v_w', 'q_b', 'k_b', 'v_b'} }

    for key, value in state_dict.items():

        # ── embeddings ──────────────────────────────────────────────────────
        if key == "embeddings.cls_token":
            new_state_dict["cls_token"] = value
        elif key == "embeddings.mask_token":
            new_state_dict["mask_token"] = value
        elif key == "embeddings.position_embeddings":
            new_state_dict["pos_embed"] = value
        elif key == "embeddings.patch_embeddings.projection.weight":
            new_state_dict["patch_embed.proj.weight"] = value
        elif key == "embeddings.patch_embeddings.projection.bias":
            new_state_dict["patch_embed.proj.bias"] = value

        # ── 最终 LayerNorm ────────────────────────────────────────────────
        elif key == "layernorm.weight":
            new_state_dict["norm.weight"] = value
        elif key == "layernorm.bias":
            new_state_dict["norm.bias"] = value

        # ── encoder 各层 ─────────────────────────────────────────────────
        elif key.startswith("encoder.layer."):
            parts = key.split(".")
            layer_num = int(parts[2])
            rest = ".".join(parts[3:])

            if layer_num not in qkv_parts:
                qkv_parts[layer_num] = {}

            # ── Attention Q/K/V（收集后统一合并）────────────────────────
            if rest == "attention.attention.query.weight":
                qkv_parts[layer_num]["q_w"] = value
            elif rest == "attention.attention.query.bias":
                qkv_parts[layer_num]["q_b"] = value
            elif rest == "attention.attention.key.weight":
                qkv_parts[layer_num]["k_w"] = value
            elif rest == "attention.attention.key.bias":
                qkv_parts[layer_num]["k_b"] = value
            elif rest == "attention.attention.value.weight":
                qkv_parts[layer_num]["v_w"] = value
            elif rest == "attention.attention.value.bias":
                qkv_parts[layer_num]["v_b"] = value

            # ── Attention 输出投影 ────────────────────────────────────────
            elif rest == "attention.output.dense.weight":
                new_state_dict[f"blocks.{layer_num}.attn.proj.weight"] = value
            elif rest == "attention.output.dense.bias":
                new_state_dict[f"blocks.{layer_num}.attn.proj.bias"] = value

            # ── LayerScale（RAD-DINO 实际使用的键名）──────────────────────
            elif rest == "layer_scale1.lambda1":
                new_state_dict[f"blocks.{layer_num}.ls1.gamma"] = value
            elif rest == "layer_scale2.lambda1":
                new_state_dict[f"blocks.{layer_num}.ls2.gamma"] = value

            # ── LayerNorm（RAD-DINO 已用 DINOv2 风格：norm1/norm2）────────
            elif rest.startswith("norm1.") or rest.startswith("norm2."):
                new_state_dict[f"blocks.{layer_num}.{rest}"] = value

            # ── MLP（RAD-DINO 已用 DINOv2 风格：mlp.fc1/mlp.fc2）─────────
            elif rest.startswith("mlp."):
                new_state_dict[f"blocks.{layer_num}.{rest}"] = value

            # ── 兼容标准 HuggingFace 风格（layernorm_before/intermediate）──
            elif rest == "layernorm_before.weight":
                new_state_dict[f"blocks.{layer_num}.norm1.weight"] = value
            elif rest == "layernorm_before.bias":
                new_state_dict[f"blocks.{layer_num}.norm1.bias"] = value
            elif rest == "layernorm_after.weight":
                new_state_dict[f"blocks.{layer_num}.norm2.weight"] = value
            elif rest == "layernorm_after.bias":
                new_state_dict[f"blocks.{layer_num}.norm2.bias"] = value
            elif rest == "intermediate.dense.weight":
                new_state_dict[f"blocks.{layer_num}.mlp.fc1.weight"] = value
            elif rest == "intermediate.dense.bias":
                new_state_dict[f"blocks.{layer_num}.mlp.fc1.bias"] = value
            elif rest == "output.dense.weight":
                new_state_dict[f"blocks.{layer_num}.mlp.fc2.weight"] = value
            elif rest == "output.dense.bias":
                new_state_dict[f"blocks.{layer_num}.mlp.fc2.bias"] = value
            else:
                print(f"  [警告] 未处理的键: {key}")

        else:
            print(f"  [警告] 未处理的顶层键: {key}")

    # ── 合并 Q/K/V → 单个 attn.qkv ────────────────────────────────────────
    for layer_num in sorted(qkv_parts.keys()):
        p = qkv_parts[layer_num]

        # weight: [3*d, d]，顺序 Q->K->V
        if all(k in p for k in ("q_w", "k_w", "v_w")):
            new_state_dict[f"blocks.{layer_num}.attn.qkv.weight"] = torch.cat(
                [p["q_w"], p["k_w"], p["v_w"]], dim=0
            )
        else:
            print(f"  [警告] layer {layer_num} 缺少 Q/K/V weight，跳过 qkv.weight")

        # bias: [3*d]，顺序 Q->K->V
        if all(k in p for k in ("q_b", "k_b", "v_b")):
            new_state_dict[f"blocks.{layer_num}.attn.qkv.bias"] = torch.cat(
                [p["q_b"], p["k_b"], p["v_b"]], dim=0
            )
        else:
            print(f"  [警告] layer {layer_num} 缺少 Q/K/V bias，跳过 qkv.bias")

    return new_state_dict


def load_rad_dino_weights(model, safetensors_path: str):
    """
    从 safetensors 文件加载 RAD-DINO 权重到 DINOv2 模型。

    Args:
        model: dinov2 的 DinoVisionTransformer（vit_base）实例
        safetensors_path: RAD-DINO 的 model.safetensors 文件路径

    Returns:
        加载了权重的模型
    """
    print(f"\n[1/3] 读取 safetensors 文件: {safetensors_path}")
    state_dict = load_file(safetensors_path)
    print(f"      原始权重: {len(state_dict)} 个键")
    print("      示例键名 (前5个):")
    for k in list(state_dict.keys())[:5]:
        print(f"        {k}: {state_dict[k].shape}")

    print("\n[2/3] 转换权重格式（Q/K/V 合并为 QKV）...")
    converted = convert_rad_dino_to_dinov2(state_dict)
    print(f"      转换后: {len(converted)} 个键")
    print("      示例键名 (前5个):")
    for k in list(converted.keys())[:5]:
        print(f"        {k}: {converted[k].shape}")

    print("\n[3/3] 加载权重到模型...")
    missing_keys, unexpected_keys = model.load_state_dict(converted, strict=False)
    print(f"      缺失的键: {len(missing_keys)} 个")
    if missing_keys:
        print("        示例:", missing_keys[:5])
    print(f"      未预期的键: {len(unexpected_keys)} 个")
    if unexpected_keys:
        print("        示例:", unexpected_keys[:5])

    loaded = len(converted) - len(unexpected_keys)
    print(f"\n      成功加载: {loaded}/{len(converted)} 个权重")
    return model


def create_rad_dino_model(safetensors_path: str, img_size: int = 518, patch_size: int = 14):
    """
    创建 DINOv2 ViT-Base 模型并载入 RAD-DINO 权重。

    RAD-DINO 规格：ViT-Base/14，img_size=518，无 register tokens。
    """
    print("创建 DINOv2 ViT-Base 模型（patch_size=14，block_chunks=0，layerscale=1e-5）...")
    model = vit_base(
        patch_size=patch_size,
        img_size=img_size,
        num_register_tokens=0,
        block_chunks=0,      # 确保扁平的 blocks.{i} 结构，不使用嵌套 blocks.{i}.{j}
        init_values=1e-5,    # 启用 LayerScale（匹配 RAD-DINO 的 layer_scale1/2）
    )
    model = load_rad_dino_weights(model, safetensors_path)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="将 RAD-DINO safetensors 权重转换并保存为 DINOv2 .pth 格式"
    )
    parser.add_argument(
        "--weights-path", "--weights",
        type=str,
        default="models--microsoft--rad-dino/snapshots/"
                "2ec9ca0e7a73c23aded999b844acd2f07c7e46b9/model.safetensors",
        dest="weights_path",
        help="RAD-DINO 的 model.safetensors 文件路径",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="rad_dino_dinov2_format.pth",
        help="输出的 PyTorch 权重文件路径（供 DINOv2 训练使用）",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=518,
        help="图像大小（RAD-DINO 默认为 518）",
    )
    args = parser.parse_args()

    model = create_rad_dino_model(args.weights_path, img_size=args.img_size)

    print(f"\n保存权重到: {args.output_path}")
    # 以 {"model": state_dict} 格式保存，与 DINOv2 pretrained_weights 加载接口兼容
    torch.save(
        {
            "model": model.state_dict(),
            "arch": "vit_base",
            "patch_size": 14,
            "img_size": args.img_size,
        },
        args.output_path,
    )

    print(f"\n完成！权重已保存到: {args.output_path}")
    print(f"\n下一步：在训练配置中设置")
    print(f"  student.pretrained_weights: '{args.output_path}'")


if __name__ == "__main__":
    main()
