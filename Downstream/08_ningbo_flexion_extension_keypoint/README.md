# Task 08: lumbar landmark linear probe

This benchmark follows the landmark task and metrics in Tingting Hu et al.,
*A Deep-Learning-Based Lumbosacral Localization and Landmark Detection Network
for Automatic Lumbar Stability and Spondylolisthesis Grading* (BIBM 2024).

Only landmark detection is trained here. Clinical quantities can be calculated
later from the saved landmark predictions.

## Protocol

- The model receives one lateral, hyperflexion, or hyperextension image.
- No YOLO is used because the portable images already center the lumbar spine;
  adding a detector would confound the frozen-backbone comparison.
- The visual backbone is frozen and forced to evaluation mode.
- The only trainable module is one affine `1x1 Conv2d` head.
- The head predicts 22 semantic heatmaps: four points on each L1--L5 vertebra
  and two superior S1 points.
- Coordinates are decoded with parameter-free spatial soft-argmax.
- All models receive the same `896 x 448` input.

Patient grouping is recovered from the filenames: `N` is the neutral lateral
view, `gqN` is hyperflexion, and `gsN` is hyperextension. All available views
for `N` stay in one split. Following the paper, 87 cases are held out for test;
10% of the remaining cases form validation. The exact split and annotation
exclusions are saved in every output directory.

## Metrics

The paper landmark metrics are reported using its `0.143 mm/pixel` spacing:

- `MAE_mm`: mean L1 coordinate error
- `MRE_mm`: mean Euclidean radial error
- `SDR_2mm_percent` through `SDR_5mm_percent`

The validation checkpoint is selected by the lowest `MRE_mm`.

## Run

Run from the repository root after activating the dedicated downstream environment:

```bash
conda activate spine_dino_downstream
cd /home/think/mnt/zpj2025/spine-dino
python Downstream/08_ningbo_flexion_extension_keypoint/train.py --config Downstream/08_ningbo_flexion_extension_keypoint/configs/rad_dino.yaml
python Downstream/08_ningbo_flexion_extension_keypoint/train.py --config Downstream/08_ningbo_flexion_extension_keypoint/configs/rad_dino_maira2.yaml
python Downstream/08_ningbo_flexion_extension_keypoint/train.py --config Downstream/08_ningbo_flexion_extension_keypoint/configs/dinov2.yaml
python Downstream/08_ningbo_flexion_extension_keypoint/train.py --config Downstream/08_ningbo_flexion_extension_keypoint/configs/dinov3.yaml
```

Results are written under `outputs/downstream/task08_keypoint_linear/<model>/`:

- `metrics.jsonl`: validation metrics for every epoch
- `best_linear_probe.pt`: linear head only
- `test_metrics.json`: final paper landmark metrics
- `test_predictions.json`: coordinates and per-landmark radial errors
- `split_and_audit.json`: case-grouped split and excluded annotations
