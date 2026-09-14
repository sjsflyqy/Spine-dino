# GeoTopo-DINO: GCVD MVP

This implementation replaces one of DINOv2's two random global crops with a
full-FOV 518x518 anchor.  The unmasked teacher anchor supplies geometry-aligned
dense and pooled-region targets to the student's eight local crops.  Original
global/local DINO, iBOT block masking, and KoLeo remain enabled in the MVP.

The package imports the baseline from `upstream/dinov2-main`; it does not copy
or modify that source tree.

## Environment

From the repository root:

```bash
conda activate /home/think/mnt/zpj2025/miniconda3/envs/spine_dino_pretrain
export PYTHONPATH="$PWD/upstream/dinov2-main:$PWD"
```

For a new environment, install `requirements-pretrain.txt` instead.

## Tests

```bash
python -m unittest discover -s methods/geotopo_dino/tests -v
```

## Geometry visualization

```bash
python -m methods.geotopo_dino.tools.visualize_correspondence \
  --image SpinePretrain-v1/spine_dino_dataset/train/spine/nhanes2_00000001.png \
  --output-dir outputs/geotopo_dino/geometry_check \
  --local-index 0
```

## Two-GPU training

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 --module \
  methods.geotopo_dino.train.train \
  --config-file methods/geotopo_dino/configs/gcvd_mvp.yaml \
  --output-dir outputs/geotopo_dino/gcvd_mvp \
  student.pretrained_weights=weights/initialization/dinov2_vitb14_dinov2_format.pth
```

The default config is a roughly ten-data-pass screening schedule when the
global batch size is eight (75,800 optimizer updates for the current 60,570
image training set).  Training resumes from the output directory by default.

Before the full run, a one-GPU, 20-update smoke test can verify the complete
forward/backward/checkpoint path:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc-per-node=1 --module \
  methods.geotopo_dino.train.train \
  --config-file methods/geotopo_dino/configs/gcvd_mvp.yaml \
  --output-dir outputs/geotopo_dino/gcvd_mvp_smoke \
  --no-resume \
  student.pretrained_weights=weights/initialization/dinov2_vitb14_dinov2_format.pth \
  train.batch_size_per_gpu=1 train.num_workers=2 train.stop_after_iterations=20 \
  train.checkpoint_period=20 evaluation.eval_period_iterations=0 \
  gcvd.warmup_type=no_warmup
```

Always compare the main experiment with an equal-budget baseline.

Useful ablations can be set from the command line:

```bash
# Anchor-only control (same views and original SSL losses, GCVD disabled)
gcvd.enabled=false

# Dense only
gcvd.region_weight=0

# Region only
gcvd.dense_weight=0 gcvd.region_weight=1

# Replace standard local-to-global DINO after the additive MVP is stable
gcvd.use_standard_local_dino=false
```

`local_order.enabled` and `tgsr.enabled` are intentionally false.  Local crop
source geometry is already collated for the former, while the teacher-first
mask-policy hook is the integration point for future topology masking.

## GCVD warmup and dense-loss switches

The canonical warmup values are `no_warmup`, `fraction`, and `fixed_iter`.
Warmup always uses the restored global optimizer iteration; it is not reset by
`stop_after_iterations` or by staged 10-pass launches.

```bash
# Immediate full weight
gcvd.warmup_type=no_warmup

# Fraction of the complete optim.epochs * OFFICIAL_EPOCH_LENGTH schedule
gcvd.warmup_type=fraction gcvd.warmup_fraction=0.05

# Fixed global optimizer iterations, independent of schedule length
gcvd.warmup_type=fixed_iter gcvd.warmup_iterations=1000
```

Dense GCVD supports the original normalized-feature cosine loss and an
independent centered prototype head. Region GCVD remains cosine in both cases.

```bash
gcvd.loss_type=cosine

gcvd.loss_type=prototype_ce \
  gcvd.head_n_prototypes=16384 \
  gcvd.teacher_temp=0.04 \
  gcvd.student_temp=0.1
```

GCVD logs now distinguish raw objectives from the contributions added to the
optimization loss:

- `gcvd_dense_raw_loss`, `gcvd_region_raw_loss`: unweighted objectives;
- `gcvd_dense_weighted_loss`, `gcvd_region_weighted_loss`: after GCVD, branch,
  and warmup weights;
- `gcvd_warmup_scale`, `gcvd_valid_ratio`: schedule and geometry diagnostics;
- prototype mode additionally logs teacher/student entropy, maximum
  probability, and active-prototype ratios;
- `optimization_loss` and `total_loss` are the actual scalar sent to backward.

The former ambiguous `gcvd_dense_loss` and `gcvd_region_loss` keys are no
longer emitted. Prototype experiments must use a new output directory rather
than resume a cosine checkpoint.
