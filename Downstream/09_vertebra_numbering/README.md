# Task 09: vertebra identification linear probe

This task follows the VertFound `MODEL.BOX_TYPE=GT` identification setting.
The official COCO train/validation/test split is preserved (919/253/138
images), with 18 classes from T1 through S1.

For every image, the frozen visual backbone produces a patch feature map.
Ground-truth vertebral boxes are mapped onto that feature map with ROIAlign,
and spatially averaged region features are classified by one shared
`Linear(768, 18)` layer. The backbone has zero trainable parameters; the only
trainable parameters are the 13,842 weights and biases of this classifier.

Reported metrics match the paper definitions:

- IDR: correctly classified GT vertebral regions / all GT vertebral regions.
- IRA: images for which every visible GT vertebral region is classified
  correctly / all images.

Run from the repository root:

```bash
python Downstream/09_vertebra_numbering/train.py \
  --config Downstream/09_vertebra_numbering/configs/rad_dino.yaml

python Downstream/09_vertebra_numbering/train.py \
  --config Downstream/09_vertebra_numbering/configs/rad_dino_maira2.yaml

python Downstream/09_vertebra_numbering/train.py \
  --config Downstream/09_vertebra_numbering/configs/dinov2.yaml

python Downstream/09_vertebra_numbering/train.py \
  --config Downstream/09_vertebra_numbering/configs/dinov3.yaml
```

Each run selects `best_linear_probe.pt` by validation IRA, breaking ties with
validation IDR, and then writes final held-out results to `test_metrics.json`.
The current configs train for 100 epochs. Outputs are stored under
`outputs/downstream/task09_linear/<backbone>_100ep/`; the original 30-epoch
directories are retained for comparison.
