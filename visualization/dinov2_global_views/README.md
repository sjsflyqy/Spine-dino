# DINOv2 native global-view visualization

This module is separate from `visualization/dino_spine_maps`. The existing
module keeps the entire radiograph and preserves its aspect ratio; this module
instead reproduces the repository's upstream DINOv2 training augmentation and
visualizes the resulting global crops.

The original full-image visualizer is not modified.

## Exact view pipeline

For every source image, the upstream `DataAugmentationDINO` produces two
independently sampled global views:

```text
source image
  -> RandomResizedCrop(scale=(0.5, 1.0), output=518x518)
  -> RandomHorizontalFlip(p=0.5)
  -> DINO color jitter / random grayscale
  -> view-specific blur (and solarization for global view 2)
  -> ImageNet normalization
  -> DINOv2 ViT-B/14
```

The attention and cosine overlays therefore belong to each **518x518 cropped
global view**, not to the complete source-image coordinate system. Both the
complete source image and the two actual model inputs are saved for inspection.

## Run on one image

```bash
python visualization/dinov2_global_views/visualize.py \
  --image /path/to/spine.png \
  --weights weights/pretrained/spine_rad_dino_vitb14_teacher_20000.pth \
  --output-dir outputs/visualization/dinov2_global_views
```

## Run on an image folder

```bash
python visualization/dinov2_global_views/visualize.py \
  --image-dir /path/to/spine_images \
  --weights weights/pretrained/dinov2_vitb14_lvd142m_100pass_plan/teacher_training_378562.pth \
  --output-dir /path/to/results \
  --layers all \
  --maps attention cosine
```

Use `--recursive` for nested input folders. Use `--save-heads` to save all 12
attention heads for every selected layer. `--seed` controls reproducible crop,
flip, and appearance augmentation sampling. Image `i` uses `seed + i - 1`.

## Checkpoint scope

This module intentionally accepts only native DINOv2 ViT-B/14 backbone
checkpoints. That includes the repository's continued-pretraining weights
initialized from public DINOv2, RAD-DINO, or RAD-DINO-MAIRA-2, because all use
the same native ViT-B/14 architecture after conversion. DINOv3 weights are
detected and rejected rather than being passed a 518x518 DINOv2 crop.

## Output

```text
<output-dir>/<relative-source-image>/<checkpoint>/
├── source_image.png
├── global_view_1/
│   ├── global_view.png
│   ├── attention/
│   ├── cosine/
│   ├── raw_maps.npz
│   └── metadata.json
└── global_view_2/
    └── ...
```

The output root also receives a checkpoint-specific `batch_summary__*.json`.

## Dependency

After installing the repository's pretraining requirements:

```bash
python -m pip install -r visualization/dinov2_global_views/requirements.txt
```

