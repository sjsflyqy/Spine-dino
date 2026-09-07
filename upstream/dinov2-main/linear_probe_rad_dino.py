"""
使用RAD-DINO权重进行线性探测的示例脚本
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from dinov2.models.vision_transformer import vit_base
from dinov2.eval.linear import run_eval_linear
from dinov2.eval.setup import get_args_parser as get_setup_args_parser


def load_rad_dino_model(safetensors_path, img_size=518, patch_size=14):
    """
    加载RAD-DINO模型
    
    Args:
        safetensors_path: RAD-DINO的model.safetensors文件路径
        img_size: 图像大小
        patch_size: patch大小
    
    Returns:
        加载了权重的模型
    """
    print(f"正在加载RAD-DINO模型从: {safetensors_path}")
    
    # 创建模型
    model = vit_base(
        patch_size=patch_size,
        img_size=img_size,
        num_register_tokens=0,
    )
    
    # 加载权重
    state_dict = load_file(safetensors_path)
    
    # 简化的权重转换（根据实际情况可能需要调整）
    new_state_dict = {}
    for key, value in state_dict.items():
        # 这里需要根据实际的键名进行转换
        # 如果键名不匹配，可以使用load_rad_dino.py中的详细转换逻辑
        new_state_dict[key] = value
    
    # 尝试加载权重
    try:
        model.load_state_dict(new_state_dict, strict=False)
        print("权重加载成功!")
    except Exception as e:
        print(f"警告: 权重加载时出现问题: {e}")
        print("建议先运行 load_rad_dino.py 转换权重格式")
    
    return model


def get_args_parser(description=None):
    """获取参数解析器"""
    parents = []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    
    parser = argparse.ArgumentParser(
        description=description or "RAD-DINO线性探测",
        parents=parents,
        add_help=True,
    )
    
    # RAD-DINO特定参数
    parser.add_argument(
        "--rad-dino-weights",
        type=str,
        default="models--microsoft--rad-dino/snapshots/2ec9ca0e7a73c23aded999b844acd2f07c7e46b9/model.safetensors",
        help="RAD-DINO权重文件路径",
    )
    
    # 数据集参数
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        default="ImageNet:split=TRAIN",
        help="训练数据集",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        default="ImageNet:split=VAL",
        help="验证数据集",
    )
    
    # 训练参数
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="批次大小（每个GPU）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="数据加载的worker数量",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
        help="学习率网格搜索",
    )
    
    # 输出参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/rad_dino_linear_probe",
        help="输出目录",
    )
    
    return parser


def main():
    """主函数"""
    parser = get_args_parser()
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("RAD-DINO线性探测")
    print("=" * 80)
    print(f"权重文件: {args.rad_dino_weights}")
    print(f"训练数据集: {args.train_dataset_str}")
    print(f"验证数据集: {args.val_dataset_str}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)
    
    # 加载RAD-DINO模型
    model = load_rad_dino_model(args.rad_dino_weights)
    model = model.cuda()
    model.eval()
    
    # 冻结backbone参数
    for param in model.parameters():
        param.requires_grad = False
    
    print("\n开始线性探测训练...")
    
    # 运行线性探测
    try:
        results = run_eval_linear(
            model=model,
            output_dir=args.output_dir,
            train_dataset_str=args.train_dataset_str,
            val_dataset_str=args.val_dataset_str,
            batch_size=args.batch_size,
            epochs=args.epochs,
            epoch_length=1250,  # 默认值
            num_workers=args.num_workers,
            save_checkpoint_frequency=20,
            eval_period_iterations=1250,
            learning_rates=args.learning_rates,
            autocast_dtype=torch.float16,
            resume=True,
        )
        
        print("\n" + "=" * 80)
        print("线性探测完成!")
        print("=" * 80)
        print("结果:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())