# 当前预训练数据集

当前训练入口只有这一套：

```text
SpinePretrain-v1/spine_dino_no_known_cervical_with_downstream_v2/
```

共 **47,071 张独立图像**：排除已知颈椎，加入下游训练图像，并隔离已识别的验证/测试重叠及病例关联。
混合来源的部位标签不完整，不能保证完全没有颈椎。

## 训练配置

GeoTopo-DINO 在原命令中指定：

```bash
"train.dataset_path=SpineDINO:split=TRAIN:root=/home/think/mnt/zpj2025/spine-dino/SpinePretrain-v1/spine_dino_no_known_cervical_with_downstream_v2:extra=/home/think/mnt/zpj2025/spine-dino/SpinePretrain-v1/spine_dino_no_known_cervical_with_downstream_v2/extra"
```

DINOv2/RAD-DINO/DINOv3 启动脚本使用：

```bash
--dataset-root /home/think/mnt/zpj2025/spine-dino/SpinePretrain-v1/spine_dino_no_known_cervical_with_downstream_v2 \
--dataset-extra /home/think/mnt/zpj2025/spine-dino/SpinePretrain-v1/spine_dino_no_known_cervical_with_downstream_v2/extra
```

迁移服务器时替换仓库绝对路径。训练命令其余参数保留，使用新的实验输出目录。

## 必须保留的数据依赖

- `SpinePretrain-v1/spine_dino_dataset/train/spine/`：原始预训练图像，v2 的符号链接指向这里。
- `SpinePretrain-v1/spine_dino_dataset/extra/`、`SpinePretrain-v1/metadata/`：原始索引及审计元数据，用于重建。
- `Downstream/data/`：下游原图与标注，v2 的另一条符号链接指向这里。
- `Downstream/splits/`：固定划分与重复证据，必须与下游评估保持一致。

上传时保留相对目录结构和符号链接，或在服务器上重新构建。仅复制 v2 目录不足以包含实际图像。
原始图像池里存在被排除的图像不影响训练：Dataset 只读取 v2 索引列出的路径。
本方案是直接图片模式，未更新原始 WebDataset tar。

## 重建

构建脚本直接从原始索引完成颈椎过滤和下游合并，不再需要中间过滤目录：

```bash
python tools/build_pretrain_with_downstream.py --output /path/to/new_combined_dataset
```

输出目录必须不存在。省略 `--output` 时默认使用当前 v2 路径，已有目录会拒绝覆盖。
默认读取 `Downstream/splits/task02_04_05_06_seed42/splits.csv`、07/08 的固定划分与 09 的官方划分。
`tools/split_downstream_02_04_05_06.py` 可重建 02/04/05/06 划分，但正常训练直接使用现有清单。

详细组成及排除依据见 [当前数据集审计](docs/下游划分与重复审计_v2.md)，
下游划分读取方式见 [固定划分说明](Downstream/splits/README.md)。
