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
