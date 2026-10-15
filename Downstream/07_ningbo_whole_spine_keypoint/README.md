# Task 07: paired whole-spine keypoint linear probe

This benchmark follows the paired-view task and landmark metrics in Dandan
Zhou et al., *A Dual-View Fusion Network for Automatic Spinal Keypoint
Detection in Biplane X-ray Images* (BIBM 2023).

## Protocol

- One sample is the AP/LAT pair sharing the same `DR_<id>`.
- The visual backbone is frozen and forced to evaluation mode.
- AP and LAT frozen feature maps are concatenated in both directions.
- The only trainable modules are two affine `1x1 Conv2d` heads, one per view.
- Each head predicts 68 semantic heatmaps: 17 vertebrae times four corners.
- Coordinates are decoded with parameter-free spatial soft-argmax.
- All models receive the same `896 x 448` input. Both dimensions are divisible
  by patch sizes 14 and 16.

The annotation audit excludes a pair if either view does not contain exactly
four points for each label 1--17. The remaining `DR_<id>` values are split as
paired groups: 20% test, then 10% of the remainder for validation. The seed,
IDs, and exclusions are saved to `split_and_audit.json` in every output folder.

## Metrics

AP and LAT are reported separately using the paper metrics:

- `epsilon_dec`: mean Euclidean error after mapping coordinates to the paper's
  `1024 x 512` coordinate system.
- `epsilon_MAE`: mean absolute error of normalized coordinates.

The validation checkpoint is selected by the lowest mean AP/LAT
`epsilon_dec`.

## Run

Run from the repository root after activating the dedicated downstream environment:

```bash
conda activate spine_dino_downstream
cd /home/think/mnt/zpj2025/spine-dino
python Downstream/07_ningbo_whole_spine_keypoint/train.py --config Downstream/07_ningbo_whole_spine_keypoint/configs/rad_dino.yaml
python Downstream/07_ningbo_whole_spine_keypoint/train.py --config Downstream/07_ningbo_whole_spine_keypoint/configs/rad_dino_maira2.yaml
python Downstream/07_ningbo_whole_spine_keypoint/train.py --config Downstream/07_ningbo_whole_spine_keypoint/configs/dinov2.yaml
python Downstream/07_ningbo_whole_spine_keypoint/train.py --config Downstream/07_ningbo_whole_spine_keypoint/configs/dinov3.yaml
```

Results are written under `outputs/downstream/task07_keypoint_linear/<model>/`:

- `metrics.jsonl`: validation metrics for every epoch
- `best_linear_probe.pt`: linear heads only; backbone weights are not duplicated
- `test_metrics.json`: final paper metrics
- `test_predictions.json`: predicted and ground-truth normalized coordinates
- `split_and_audit.json`: exact data split and excluded annotations

## 可选探测头：保留原始线性基线

原 `model.py`、`train.py` 和旧 YAML 配置保持不变。新增入口为
`train_probe.py`。未启用 LoRA 时，只训练头，骨干始终冻结并保持 eval 模式。
配置中启用 `model.adapter.enabled: true` 时，同时训练 LoRA 与所选头，
骨干原始参数仍冻结。原来的 LoRA 入口继续可用。

| `model.probe.type` | 结构 | 分类 |
| --- | --- | --- |
| `legacy_linear` | 原始两个 `1×1 Conv(1536, 68)`，直接使用原模型类 | 最后一层线性探测 |
| `multilayer_linear` | 多个冻结 block 特征通道拼接，再使用两个单层 `1×1 Conv` | 多层特征线性探测 |
| `spatial_linear` | 两个单层 `3×3 Conv`，无激活函数 | 邻域线性探测 |
| `mlp_probe` | 每个视角 `1×1 Conv → GELU → 1×1 Conv` | 冻结特征非线性探测 |
| `cnn_probe` | 每个视角 `1×1 Conv → GroupNorm → GELU → 3×3 Conv → GroupNorm → GELU → 1×1 Conv` | 冻结特征非线性探测 |

表中分类针对未启用 LoRA 的情况；启用 LoRA 后，五种头均属于 LoRA 微调实验，
因为特征随适配器训练而变化。

所有新头保留 AP/LAT 的双向拼接、每视角 68 通道热图、原始 token 网格、
高斯热图交叉熵、soft-argmax 及原评价指标。新入口复用原训练脚本中的
数据加载器构造与评估函数；数据审计和按 DR 编号划分也使用相同实现。
MLP/CNN 默认隐藏宽度为 256，两个视角的头参数独立。

多层线性头默认取零起始 block 索引 `[2, 5, 8, 11]`，即第 3、6、9、12 层。
各层保持原 token 网格，不构建特征金字塔。原生 DINO 使用官方中间层接口的
冻结最终 LayerNorm；HF DINO 对所选 hidden states 应用其冻结 `layernorm`。
这里的“线性”指从这些固定特征到热图 logits 的读出，坐标解码仍使用 softmax。
这三种线性读出使用不同的特征范围，应分别标注；MLP/CNN 不应报告为严格线性探测。

### 执行命令

从仓库根目录执行。只在尚未初始化 Conda 的 shell 中需要先执行 `source`。

```bash
source /home/think/mnt/zpj2025/miniconda3/etc/profile.d/conda.sh
conda activate spine_dino_downstream
cd /home/think/mnt/zpj2025/spine-dino
```

原始线性头的执行方式保持不变：

```bash
python Downstream/07_ningbo_whole_spine_keypoint/train.py \
  --config Downstream/07_ningbo_whole_spine_keypoint/configs/dinov2.yaml
```

多层线性头：

```bash
python Downstream/07_ningbo_whole_spine_keypoint/train_probe.py \
  --config Downstream/07_ningbo_whole_spine_keypoint/configs/probes/multilayer_linear/dinov2.yaml
```

轻量 CNN 头：

```bash
python Downstream/07_ningbo_whole_spine_keypoint/train_probe.py \
  --config Downstream/07_ningbo_whole_spine_keypoint/configs/probes/cnn_probe/dinov2.yaml
```

这两种配置目录中均提供以下七个预设，训练参数从对应旧配置复制，仅增加头配置、
改变输出目录：

- `dinov2.yaml`
- `dinov3.yaml`
- `rad_dino.yaml`
- `rad_dino_maira2.yaml`
- `spine_rad_dino_teacher_20k.yaml`
- `spine_rad_dino_maira2_teacher_20k.yaml`
- `spine_dinov3_teacher_20k.yaml`

例如预训练后的 DINOv3 使用 CNN 头：

```bash
python Downstream/07_ningbo_whole_spine_keypoint/train_probe.py \
  --config Downstream/07_ningbo_whole_spine_keypoint/configs/probes/cnn_probe/spine_dinov3_teacher_20k.yaml
```

可以在旧配置或新配置基础上，通过参数选择其他头，无需编辑 YAML：

```bash
# 在新入口使用原始模型类，结果写入新实验目录。
python Downstream/07_ningbo_whole_spine_keypoint/train_probe.py \
  --config Downstream/07_ningbo_whole_spine_keypoint/configs/dinov2.yaml \
  --probe-type legacy_linear

python Downstream/07_ningbo_whole_spine_keypoint/train_probe.py \
  --config Downstream/07_ningbo_whole_spine_keypoint/configs/dinov2.yaml \
  --probe-type spatial_linear

python Downstream/07_ningbo_whole_spine_keypoint/train_probe.py \
  --config Downstream/07_ningbo_whole_spine_keypoint/configs/dinov2.yaml \
  --probe-type mlp_probe --hidden-channels 256
```

### LoRA + MLP / CNN

直接使用原来的 `lora_*.yaml`，将入口替换为 `train_probe.py` 并增加 `--probe-type`。
是否启用 LoRA 由配置文件决定，`--probe-type` 仅选择任务头。
以下两条命令可分别执行，默认隐藏宽度均为 256：

```bash
CUDA_VISIBLE_DEVICES=6 python Downstream/07_ningbo_whole_spine_keypoint/train_probe.py \
  --config Downstream/07_ningbo_whole_spine_keypoint/configs/lora_dinov2.yaml \
  --weights weights/pretrained/maira2_G0125_2gpu_50passes/teacher_training_14709.pth \
  --output-dir outputs/downstream/task07_keypoint_lora_probe/mlp_probe/maira2_G0125_2gpu_50passes/training_14709 \
  --probe-type mlp_probe

CUDA_VISIBLE_DEVICES=6 python Downstream/07_ningbo_whole_spine_keypoint/train_probe.py \
  --config Downstream/07_ningbo_whole_spine_keypoint/configs/lora_dinov2.yaml \
  --weights weights/pretrained/maira2_G0125_2gpu_50passes/teacher_training_14709.pth \
  --output-dir outputs/downstream/task07_keypoint_lora_probe/cnn_probe/maira2_G0125_2gpu_50passes/training_14709 \
  --probe-type cnn_probe
```

新入口保留原 LoRA 配置中的 rank、alpha、dropout、target_modules，
并复用原来的适配器/头分组学习率、warmup、余弦调度、梯度累积和裁剪。
训练时仅 LoRA 分支和头接收梯度；评估时进入 eval 模式并关闭梯度。
也支持 LoRA 与其余三种头的组合，但均应报告为微调实验。

`--probe-type` 会将特征层和隐藏宽度重置为所选类型的默认值；随后可通过
`--feature-layers last` 或 `--feature-layers 2,5,8,11`、`--hidden-channels` 显式覆盖。
`legacy_linear` 只允许 `last`，`multilayer_linear` 至少需要两个 block。
其余头也可显式选择多个特征层。

自定义骨干权重时，配置中的 `model.backbone` 必须与权重的架构和格式匹配。
例如本项目导出的 RAD-DINO teacher `.pth` 使用 `dinov2` 原生加载器，
HF RAD-DINO 则使用包含配置和 safetensors 的目录：

```bash
python Downstream/07_ningbo_whole_spine_keypoint/train_probe.py \
  --config Downstream/07_ningbo_whole_spine_keypoint/configs/probes/cnn_probe/dinov2.yaml \
  --weights /absolute/path/to/extracted_teacher.pth \
  --output-dir outputs/downstream/task07_keypoint_probe/cnn_probe/my_teacher
```

### 输出、兼容性与验证

新预设的默认输出目录为
`outputs/downstream/task07_keypoint_probe/<probe_type>/<model>/`。
启用 LoRA 时，自动生成的默认目录为
`outputs/downstream/task07_keypoint_lora_probe/<probe_type>/<model>/`。
使用旧配置进入新入口，或使用 `--probe-type` 选择头时，也会自动采用这个目录结构。
显式 `--output-dir` 优先；目录必须为空，防止覆盖已有实验。重跑时请指定新目录。

输出包括 `resolved_config.json`、`split_and_audit.json`、`metrics.jsonl`、
`test_metrics.json`、`test_predictions.json` 和 `training_summary.json`。
未启用 LoRA 时，checkpoint 仅保存探测头，不复制冻结骨干；其中记录完整
`probe_spec`、`adapter_spec` 和实验配置。
原头的文件名为 `best_linear_probe.pt`，其他头为 `best_<probe_type>.pt`。
启用 LoRA 时，checkpoint 的 `trainable_state` 同时保存全部适配器和头参数，
不复制冻结基础权重。MLP/CNN 的文件名分别为 `best_lora_mlp_probe.pt`、
`best_lora_cnn_probe.pt`，legacy 头则为 `best_lora_finetune.pt`。
恢复时 `restore_probe` 校验头类型、特征层、隐藏宽度和 LoRA 配置，
并检查适配器与头参数是否完整，避免只恢复头而遗漏 LoRA。
旧版不含 `probe_spec` 的线性 checkpoint 仍可恢复到 `legacy_linear`；
新入口的 legacy checkpoint 也保留原来的 `probe.ap_head/lat_head` 格式。
原 `train.py` 保存的 LoRA checkpoint 也可恢复到相同配置的 `legacy_linear`。
不支持断点续训；保存的最佳权重用于最终测试。

可通过 `--epochs 1 --workers 0 --output-dir <新的空目录>` 做一轮完整数据流程检查，
也支持 `--batch-size` 和 `--device cpu|cuda|auto`。
训练 epoch、batch、workers 的命令行覆盖不会修改 YAML。

运行兼容性及微型端到端测试：

```bash
python -m unittest discover \
  -s Downstream/07_ningbo_whole_spine_keypoint/tests -v
```

测试覆盖原头初始化与输出一致性、旧 checkpoint 兼容性、线性性质、
HF/原生中间层规范化、五种头的梯度和冻结状态，以及训练、验证、保存、加载与测试输出。
还覆盖 LoRA 梯度、基础权重不变、分组学习率、调度器执行、旧 LoRA checkpoint
兼容性，以及 LoRA + MLP/CNN 的完整训练与保存加载流程。
完整结构化解码器继续使用独立的 `train_decoder.py` 和 `configs/decoder/`；
它使用不同预测形式及损失，应单独比较。
