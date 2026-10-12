# DINO spine-map visualization

This module visualizes spatial signals from every Transformer block of the
repository's native DINOv2 ViT-B/14 and DINOv3 ViT-B/16 checkpoints. It is an
offline analysis tool: it does not alter pretraining masks, the dataloader, or
teacher/student training.

## Maps

- **CLS attention** is the CLS-query row of each self-attention layer, sliced
  to patch keys after the full softmax. DINOv3 RoPE and storage tokens are
  handled explicitly. The raw attention retained for a layer has shape
  `[12, grid_height, grid_width]`.
- **CLS-to-patch cosine similarity** is computed from normalized CLS and patch
  tokens after each Transformer block. It is available for all layers, not
  only the final layer.

Both maps are heuristics for visual analysis. Neither should be interpreted as
a calibrated spine-segmentation probability.

## 脊柱 patch 与所有 patch 的余弦相似度

新增独立入口 `patch_similarity.py`，保留原有 `visualize.py` 的 CLS 功能。
在原图的脊柱内部指定一个点，映射到模型输入中的 patch；取该 patch 的
backbone 特征，与同一幅图的所有 patch 特征计算余弦相似度：

```python
# patches: [1, grid_h * grid_w, embedding_dim]
# 使用该层 block 输出，经 model.norm 后的 patch token；不包含 CLS/register/storage token。
patches = torch.nn.functional.normalize(patches.float(), dim=-1)
query_index = row * grid_w + col
similarity = (patches * patches[:, query_index:query_index + 1]).sum(-1)
similarity = similarity[0].reshape(grid_h, grid_w)
```

选择的是整图前向传播得到的、带上下文的 patch token，并非裁剪小图后重新编码。
查询 patch 与自身的相似度约为 1，所有原始值均保留，不做逐图 min-max 归一化。

单个 GCVD checkpoint：

```bash
python visualization/dino_spine_maps/patch_similarity.py \
  --image image_data/images_for_mask_maps/ningbo_00000356.jpg \
  --weights weights/pretrained/maira2_F025_2gpu_29420steps/teacher_training_29419.pth \
  --query-xy 505 1740 \
  --layers last \
  --output-dir outputs/patch_similarity/ningbo_gcvd
```

这里的 `505 1740` 是该示例图的脊柱选点，换图片时需要重新选择。
坐标为 EXIF 方向修正后的原图像素 `(x,y)`，从左上角开始，先水平再垂直。
代码按实际 resize 尺寸和 padding 自动转换坐标。

比较 baseline、GCVD 或不同训练迭代的 checkpoint（顺序由 `--weights` 决定）：

```bash
python visualization/dino_spine_maps/patch_similarity.py \
  --image image_data/images_for_mask_maps/ningbo_00000356.jpg \
  --weights \
    weights/pretrained/spine_rad_dino_maira2_vitb14_teacher_20000.pth \
    weights/pretrained/maira2_F025_2gpu_29420steps/teacher_training_29419.pth \
  --labels MAIRA2_spine GCVD_F025 \
  --query-xy 505 1740 \
  --layers last --long-side 896 \
  --output-dir outputs/patch_similarity/ningbo_comparison
```

上面的两个现有权重用于演示命令；判断 GCVD 的增益，应替换成同一初始权重、
训练数据和训练步数下的有/无 GCVD 对照。图中每个模型都使用相同原图选点。
也可以把 `--weights` 换成多个训练阶段的导出权重，`--labels` 换成对应迭代数，
得到类似示例论文的演化图。

参数说明：

- `--query-xy X Y`：原图像素选点。
- `--query-relative U V`：原图宽、高的比例坐标，范围 `[0,1)`；例如 `0.45 0.68`。
- `--query-patch ROW COL`：补边后的模型 patch 网格中从零开始的行、列；
  多 checkpoint 比较时要求 patch size 和网格一致。
- 三种选点方式必须且只能提供一种。程序不会自动假设图像中心就是脊柱。
- `--layers last` 默认取最后一层；可选 `all`、`5-12`、`3,6,9,12`。
- `--vmin -1 --vmax 1` 默认固定所有图的色标；可统一改为 `--vmin 0 --vmax 1`
  来突出正相似度，负值此时仅在显示中被截到最低颜色，NPZ 数值不变。
- `--device cpu` 可用于无 GPU 的环境；`--long-side 448` 可用于快速检查。
- 目前复用原有加载器，支持 native DINOv2/3 ViT-B，包括 RAD-DINO/MAIRA2
  和基于这些 backbone 训练后导出的 GCVD 权重。

每个 checkpoint 输出 `layer_12_grid.png`（原图 + 无平滑的 patch 热图）、
`layer_12_overlay.png`（原图 + 热图叠加）、`raw_maps.npz` 和 `metadata.json`。
原图、热图和叠加图均用红框标出查询 patch。
根目录的 `comparison_layer_12_grid.png` 将原图和所有 checkpoint 横向排列，
`comparison_layer_12_overlay.png` 提供对应叠加图。其他层按层号命名。
NPZ 中 `layer_12_patch_cosine` 为 patch–patch 原始相似度，
`query_patch_rc` 为查询 patch 的行列，`valid_patch_mask` 排除无效补边区域。
metadata 保存原图选点、patch 索引、原图上的 patch 边界和完整几何信息。
比较不同 patch size 的模型时，同一选点对应的 patch 覆盖范围会不同，原图中会
显示这些边界；严谨消融优先比较同架构、同分辨率的模型。

完整 SSL checkpoint 先导出 teacher backbone：

```bash
python tools/extract_teacher_backbone.py \
  --architecture dinov2 \
  --checkpoint /path/to/model_final.rank_0.pth \
  --output /path/to/gcvd_teacher.pth
```

可视化使用 backbone 特征，没有经过 GCVD projection head。它可观察特征的空间
局部性和结构一致性，但仅凭热图不能证明 GCVD 改进：查询自身的亮点是必然的，
还需固定脊柱/背景选点、病例和训练配置比较，配合下游指标。

运行数值与几何测试：

```bash
python -m unittest discover -s visualization/dino_spine_maps/tests -v
```

## Installation

Install the repository's downstream or pretraining requirements first, then:

```bash
python -m pip install -r visualization/dino_spine_maps/requirements.txt
```

## Basic usage

Run from the repository root:

```bash
python visualization/dino_spine_maps/visualize.py \
  --image /path/to/spine_xray.png \
  --weights weights/pretrained/dinov3_vitb16_lvd1689m_100pass_plan/teacher_training_378562.pth
```

Process every supported image directly inside one folder and choose the output
folder:

```bash
python visualization/dino_spine_maps/visualize.py \
  --image-dir /path/to/xray_folder \
  --weights weights/pretrained/spine_dinov3_vitb16_teacher_20000.pth \
  --output-dir /path/to/visualization_results
```

Add `--recursive` to scan nested folders. Their relative directory structure
is preserved in the output, and files with the same stem but different
extensions remain separate. The model is loaded once for the entire batch.
Unreadable images are reported and skipped by default; `--fail-fast` stops at
the first failure. A checkpoint-specific `batch_summary__*.json` records all
successful and failed inputs.

Select layers or save the individual attention heads:

```bash
python visualization/dino_spine_maps/visualize.py \
  --image /path/to/spine_xray.png \
  --weights weights/pretrained/dinov2_vitb14_lvd142m_100pass_plan/teacher_training_378562.pth \
  --layers 5-12 \
  --maps attention cosine \
  --save-heads
```

Useful options:

- `--architecture auto|dinov2|dinov3`: automatic checkpoint inspection is the
  default.
- `--layers all|12|5-12|3,6,9,12`: user-facing layer numbers are one-based.
- `--long-side 896`: aspect-preserving inference resolution. The other side is
  inferred and minimally padded to a patch-size multiple.
- `--device auto|cpu|cuda|cuda:N`: `auto` uses CUDA when available.
- `--output-dir`: defaults to `outputs/visualization`.

The first version accepts raster images supported by Pillow (PNG, JPEG, TIFF,
BMP, and similar formats). It intentionally does not perform DICOM windowing.

## Supported checkpoints

The loader targets extracted/native backbone `.pth` files, including:

- training initialization files with a tensor state dict under `teacher` and
  metadata under `meta`, such as
  `weights/initialization/dinov3_vitb16_lvd1689m_training_init.pth`;
- official DINOv2 ViT-B/14 and DINOv3 ViT-B/16 weights under `weights/base`;
- `teacher_training_*.pth` files under `weights/pretrained`;
- `spine_dinov3_vitb16_teacher_20000.pth`;
- `spine_rad_dino_vitb14_teacher_20000.pth`;
- `spine_rad_dino_maira2_vitb14_teacher_20000.pth`.

A full SSL checkpoint must first be converted with
`tools/extract_teacher_backbone.py`. The historical
`weights/pretrained/spine-dino-v1/model_final.rank_0.pth` is not treated as a
portable backbone.

Automatic architecture detection inspects parameter names rather than the
checkpoint filename. A `pos_embed` parameter identifies the repository's
DINOv2 family; `storage_tokens` or `rope_embed.*` identifies DINOv3. The loader
then constructs the corresponding ViT-B/14 or ViT-B/16 and calls strict state
dict loading. Therefore an incorrect detection, incompatible model size, or
malformed checkpoint fails instead of silently accepting unmatched weights.

## Output

For each image/checkpoint pair, the CLI writes:

```text
outputs/visualization/<relative-image-path>/<checkpoint>/
├── input/
│   ├── original.png
│   ├── model_input.png
│   └── valid_region.png
├── attention/
│   ├── all_layers.png
│   └── layer_01_mean.png ... layer_12_mean.png
├── cosine/
│   ├── all_layers.png
│   └── layer_01.png ... layer_12.png
├── raw_maps.npz
└── metadata.json
```

`raw_maps.npz` contains unmodified patch-grid values and the exact
`valid_patch_mask` for later layer selection, smoothing, dynamic-programming
centerline extraction, and band construction.
PNG overlays use a shared robust color scale across the selected layers.

`metadata.json` records the original/model geometry, padding, patch grid,
architecture, selected layers, and per-head attention mass assigned to image
patches. These values make every map invertible to the original image space.

## Validation

Before interpreting maps from a new environment or checkpoint, run:

```bash
python visualization/dino_spine_maps/validate.py \
  --image /path/to/spine_xray.png \
  --weights /path/to/extracted_backbone.pth
```

The validator checks layer/head counts, finite values, cosine bounds, map
geometry, and agreement between manual block traversal and the model's normal
`forward_features()` output.
