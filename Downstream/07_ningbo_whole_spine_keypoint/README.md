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
