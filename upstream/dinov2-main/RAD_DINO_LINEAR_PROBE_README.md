# RAD-DINO 线性探测使用指南

本指南说明如何将RAD-DINO的权重加载到dinov2框架中，并进行线性探测评估。

## 📋 目录

- [背景](#背景)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [详细步骤](#详细步骤)
- [自定义数据集](#自定义数据集)
- [常见问题](#常见问题)

## 🎯 背景

**RAD-DINO** 是Microsoft开发的医学影像自监督学习模型，基于DINOv2架构。本项目提供了将RAD-DINO权重加载到dinov2框架的工具，使你能够：

1. 使用RAD-DINO的预训练权重
2. 在自己的下游任务上进行线性探测
3. 评估模型在特定任务上的性能

**模型配置：**
- 架构：ViT-Base
- 嵌入维度：768
- 层数：12
- 注意力头数：12
- Patch大小：14×14
- 输入图像大小：518×518

## 🔧 环境要求

### 必需的依赖

```bash
# 安装dinov2的依赖
pip install -r requirements.txt

# 安装额外的依赖
pip install safetensors transformers
```

### 验证环境

```bash
python -c "import torch; import safetensors; print('环境配置正确!')"
```

## 🚀 快速开始

### 方法1：直接使用（推荐用于测试）

```bash
# 使用默认的RAD-DINO权重进行线性探测
python linear_probe_rad_dino.py \
    --train-dataset "YourDataset:split=TRAIN" \
    --val-dataset "YourDataset:split=VAL" \
    --output-dir ./output/rad_dino_results \
    --batch-size 64 \
    --epochs 10
```

### 方法2：先转换权重（推荐用于生产环境）

```bash
# 步骤1: 转换权重格式
python load_rad_dino.py \
    --weights-path models--microsoft--rad-dino/snapshots/2ec9ca0e7a73c23aded999b844acd2f07c7e46b9/model.safetensors \
    --output-path rad_dino_converted.pth

# 步骤2: 使用dinov2的标准线性探测脚本
python dinov2/eval/linear.py \
    --config-file dinov2/configs/eval/vitb14_pretrain.yaml \
    --pretrained-weights rad_dino_converted.pth \
    --train-dataset "YourDataset:split=TRAIN" \
    --val-dataset "YourDataset:split=VAL" \
    --output-dir ./output/rad_dino_linear
```

## 📖 详细步骤

### 步骤1：准备RAD-DINO权重

你的权重文件已经在：
```
models--microsoft--rad-dino/snapshots/2ec9ca0e7a73c23aded999b844acd2f07c7e46b9/model.safetensors
```

### 步骤2：了解权重转换

RAD-DINO使用Hugging Face的transformers格式，需要转换为dinov2格式：

**主要转换规则：**
- `embeddings.patch_embeddings.projection` → `patch_embed.proj`
- `embeddings.cls_token` → `cls_token`
- `embeddings.position_embeddings` → `pos_embed`
- `encoder.layer.{N}.attention.attention.{qkv}` → `blocks.{N}.attn.qkv`
- `encoder.layer.{N}.layernorm_before` → `blocks.{N}.norm1`
- `encoder.layer.{N}.intermediate.dense` → `blocks.{N}.mlp.fc1`

### 步骤3：准备数据集

#### 使用ImageNet格式的数据集

```python
# 数据集应该组织为：
your_dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img1.jpg
│       └── img2.jpg
└── val/
    ├── class1/
    └── class2/
```

#### 在代码中指定数据集

```bash
python linear_probe_rad_dino.py \
    --train-dataset "ImageFolder:root=/path/to/your_dataset/train:split=TRAIN" \
    --val-dataset "ImageFolder:root=/path/to/your_dataset/val:split=VAL"
```

### 步骤4：运行线性探测

#### 基础用法

```bash
python linear_probe_rad_dino.py \
    --rad-dino-weights models--microsoft--rad-dino/snapshots/2ec9ca0e7a73c23aded999b844acd2f07c7e46b9/model.safetensors \
    --train-dataset "YourDataset:split=TRAIN" \
    --val-dataset "YourDataset:split=VAL" \
    --output-dir ./output/my_experiment \
    --batch-size 64 \
    --epochs 20 \
    --num-workers 4
```

#### 高级用法：学习率网格搜索

```bash
python linear_probe_rad_dino.py \
    --train-dataset "YourDataset:split=TRAIN" \
    --val-dataset "YourDataset:split=VAL" \
    --learning-rates 1e-4 5e-4 1e-3 5e-3 1e-2 \
    --epochs 10 \
    --batch-size 128
```

### 步骤5：查看结果

结果将保存在输出目录中：

```
output/my_experiment/
├── results_eval_linear.json    # 详细的评估结果
├── model_final.pth             # 最终的分类器权重
└── checkpoints/                # 训练过程中的检查点
```

## 🎨 自定义数据集

### 创建自定义数据集类

如果你的数据集格式特殊，可以创建自定义数据集：

```python
# custom_dataset.py
from torch.utils.data import Dataset
from PIL import Image
import os

class MyCustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # 加载你的数据
        self.samples = self._load_samples()
    
    def _load_samples(self):
        # 实现你的数据加载逻辑
        samples = []
        # ... 加载图像路径和标签
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_targets(self):
        return [label for _, label in self.samples]
```

### 注册自定义数据集

在 [`dinov2/data/datasets/__init__.py`](dinov2/data/datasets/__init__.py) 中注册：

```python
from .custom_dataset import MyCustomDataset

# 在数据集注册表中添加
_DATASET_REGISTRY["MyCustomDataset"] = MyCustomDataset
```

## 🔍 常见问题

### Q1: 权重加载失败怎么办？

**A:** 如果直接加载失败，使用两步法：

```bash
# 1. 先转换权重
python load_rad_dino.py --weights-path <path> --output-path converted.pth

# 2. 检查转换后的权重
python -c "import torch; d=torch.load('converted.pth'); print(d.keys())"

# 3. 使用转换后的权重
python dinov2/eval/linear.py --pretrained-weights converted.pth ...
```

### Q2: 内存不足怎么办？

**A:** 减小批次大小和worker数量：

```bash
python linear_probe_rad_dino.py \
    --batch-size 32 \
    --num-workers 2
```

### Q3: 如何使用多GPU训练？

**A:** 使用torchrun：

```bash
torchrun --nproc_per_node=4 linear_probe_rad_dino.py \
    --train-dataset "YourDataset:split=TRAIN" \
    --val-dataset "YourDataset:split=VAL"
```

### Q4: 图像大小不是518×518怎么办？

**A:** RAD-DINO在518×518上训练，但可以处理其他尺寸。模型会自动插值位置编码。建议：
- 保持宽高比
- 使用接近518的尺寸以获得最佳性能

### Q5: 如何调整学习率？

**A:** 线性探测会自动进行学习率网格搜索。你可以自定义搜索范围：

```bash
python linear_probe_rad_dino.py \
    --learning-rates 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2
```

### Q6: 如何在医学影像上使用？

**A:** RAD-DINO专为医学影像设计，建议：

1. **保持原始分辨率**：医学影像的细节很重要
2. **适当的预处理**：
   ```python
   from torchvision import transforms
   
   transform = transforms.Compose([
       transforms.Resize(518),
       transforms.CenterCrop(518),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
   ])
   ```
3. **考虑数据增强**：根据任务选择合适的增强策略

## 📊 性能优化建议

### 1. 批次大小选择

- **小数据集** (<10k样本): batch_size=32-64
- **中等数据集** (10k-100k): batch_size=64-128
- **大数据集** (>100k): batch_size=128-256

### 2. 学习率调整

线性探测的学习率通常在 `1e-4` 到 `1e-2` 之间。建议：
- 从较小的学习率开始
- 使用网格搜索找到最佳值
- 考虑使用余弦退火调度器

### 3. 训练轮数

- **快速实验**: 5-10 epochs
- **正式评估**: 20-50 epochs
- **最佳性能**: 50-100 epochs

## 📝 示例：完整的工作流程

```bash
# 1. 验证环境
python -c "import torch, safetensors; print('OK')"

# 2. 检查权重文件
ls -lh models--microsoft--rad-dino/snapshots/*/model.safetensors

# 3. 准备数据集（假设已经准备好）
# your_data/train/ 和 your_data/val/

# 4. 运行线性探测
python linear_probe_rad_dino.py \
    --train-dataset "ImageFolder:root=your_data/train" \
    --val-dataset "ImageFolder:root=your_data/val" \
    --output-dir ./results/experiment_001 \
    --batch-size 64 \
    --epochs 20 \
    --learning-rates 1e-4 5e-4 1e-3 5e-3 1e-2

# 5. 查看结果
cat ./results/experiment_001/results_eval_linear.json
```

## 🎓 进阶使用

### 使用dinov2原生接口

如果你想更灵活地控制训练过程：

```python
from dinov2.models.vision_transformer import vit_base
from safetensors.torch import load_file
import torch

# 加载模型
model = vit_base(patch_size=14, img_size=518)
weights = load_file('path/to/model.safetensors')
model.load_state_dict(weights, strict=False)

# 冻结backbone
for param in model.parameters():
    param.requires_grad = False

# 添加分类头
classifier = torch.nn.Linear(768, num_classes)

# 训练分类器
# ... 你的训练代码
```

## 📚 参考资料

- [DINOv2 论文](https://arxiv.org/abs/2304.07193)
- [RAD-DINO 论文](https://arxiv.org/abs/2401.10815)
- [DINOv2 GitHub](https://github.com/facebookresearch/dinov2)
- [RAD-DINO Hugging Face](https://huggingface.co/microsoft/rad-dino)

## 💡 提示

1. **首次运行**：建议先用小数据集测试，确保流程正确
2. **监控训练**：使用tensorboard或wandb记录训练过程
3. **保存检查点**：定期保存模型以防训练中断
4. **验证结果**：在测试集上验证最终性能

---

如有问题，请查看日志文件或提issue！