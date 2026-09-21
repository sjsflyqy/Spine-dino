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

# Change only the GCVD prototype teacher to globally balanced assignments.
# train.centering remains unchanged for the standard DINO/iBOT objectives.
gcvd.loss_type=prototype_ce \
  gcvd.teacher_centering=sinkhorn_knopp \
  gcvd.sinkhorn_iterations=3
```

GCVD logs now distinguish raw objectives from the contributions added to the
optimization loss:

- `gcvd_dense_raw_loss`, `gcvd_region_raw_loss`: unweighted objectives;
- `gcvd_dense_weighted_loss`, `gcvd_region_weighted_loss`: after GCVD, branch,
  and warmup weights;
- `gcvd_warmup_scale`, `gcvd_valid_ratio`: schedule and geometry diagnostics;
- prototype mode additionally logs teacher/student entropy, maximum
  probability, hard active-prototype ratios, marginal entropy, and soft
  effective-prototype ratios; `gcvd_dense_raw_kl` subtracts teacher entropy
  from the dense cross-entropy for a comparable student-teacher mismatch;
- `optimization_loss` and `total_loss` are the actual scalar sent to backward.

The former ambiguous `gcvd_dense_loss` and `gcvd_region_loss` keys are no
longer emitted. Prototype experiments must use a new output directory rather
than resume a cosine checkpoint.

## Optional pixel reconstruction using the iBOT mask

`pixel_reconstruction.enabled` defaults to `false`. A configuration without this
section also uses the original training path: no decoder parameters, forward,
pixel losses, optimizer groups, or pixel logs are created. The augmentation,
iBOT mask policy and teacher EMA module list are unchanged.

When enabled, the student-only `student_aux.pixel_decoder` consumes the complete
`x_norm_patchtokens` sequence from the existing masked global forward. It uses
a projection, fixed 2-D position encoding, two 256-wide Transformer blocks by
default, and a linear pixel predictor. It does not delete/reorder tokens or add
new mask tokens. Both global views are reconstructed in their original order;
local crops have no reconstruction branch. There is no second backbone forward.

The target is `collated_global_crops`, exactly the **augmented, normalized input
received by the teacher**. This includes existing blur/intensity augmentations;
it is not the original pre-augmentation radiograph. MSE is computed in FP32 only
at final iBOT-masked, valid positions. Anchor padding is excluded. Each selected
patch has equal weight across the distributed batch (patches with more pixels
are first averaged internally). Ranks with zero selected patches still execute
the decoder and participate in synchronization with zero local contribution.

The loss is `existing_loss + loss_weight * warmup_scale * pixel_loss`. With
`warmup_iterations: 1000`, the scale is zero at iteration 0 and reaches one at
global iteration 1000. Set it to zero for no warmup. The default loss weight of
0.1 is an experimental starting value, not a calibrated balance.

`norm_pix_loss: false` predicts normalized input intensities directly; the
optional `true` setting additionally standardizes each target patch as in MAE.
The decoder uses linear outputs, without sigmoid. FSDP precision defaults to the
student DINO head settings (FP16 parameters / FP32 gradient reduction in the
default config); an explicit `compute_precision.student_aux.pixel_decoder`
section can override this with the same schema as other FSDP modules.

Start a pixel experiment (same schedule/views as the MVP):

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 --module \
  methods.geotopo_dino.train.train \
  --config-file methods/geotopo_dino/configs/gcvd_pixel.yaml \
  --output-dir outputs/geotopo_dino/gcvd_pixel --no-resume \
  student.pretrained_weights=weights/initialization/dinov2_vitb14_dinov2_format.pth
```

You can also use the existing MVP config and pass
`pixel_reconstruction.enabled=true`. Conversely, pass
`pixel_reconstruction.enabled=false` to disable it. The GCVD switch is independent:
`gcvd.enabled=false` gives the same anchor/global view setup with DINO, iBOT,
KoLeo and optional pixel reconstruction. It does not restore the baseline's
two-random-global augmentation policy.

Resume a pixel run using the same configuration/output directory and omit
`--no-resume`. Full training checkpoints include decoder and optimizer state;
the teacher export used downstream remains decoder-free. A checkpoint signature
rejects changed decoder settings, enabled flags or `norm_pix_loss` before state
is loaded. Old decoder-free checkpoints remain compatible with the disabled
branch. To switch configurations, start a new output directory with `--no-resume`
and use a backbone-only initialization file in the existing
`student.pretrained_weights` format (`{"model": backbone_state_dict}`), rather
than an incompatible recovery checkpoint in `MODEL.WEIGHTS`.

Additional logs:

- `pixel_raw_loss`: before weighting; its distributed average is the global masked-patch MSE.
- `pixel_weighted_loss`: the contribution added to the total optimization loss.
- `pixel_warmup_scale`: the current scale, computed from the restored global iteration.
- `pixel_masked_patch_count`: average selected patch count per rank, after the existing logger reduction.

Set `pixel_reconstruction.visualization_period=1000` to save up to two masked
views from rank zero under `pixel_reconstruction/step_XXXXXXX.png`. Columns show
the target, mask illustration, decoder-only prediction, a visible-target/masked-
prediction composite, and relative masked MSE. Visible-position decoder outputs
are unsupervised. With `norm_pix_loss=true`, converting predictions back to image
intensities uses **ground-truth patch statistics**, explicitly labeled in the
figure; it is not an independent recovery of absolute brightness. The error
panel is normalized independently per image, so use scalar logs for comparisons.
Visualization defaults to off and only runs on selected logging steps.

CPU checks (the meta-architecture integration tests replace only CUDA-only
encoder/head-packing operations with small CPU stand-ins):

```bash
python -m unittest discover -s methods/geotopo_dino/tests -v
```

Real CUDA/FSDP synthetic smoke test, requiring no images or pretrained weights:

```bash
torchrun --standalone --nproc-per-node=1 --module \
  methods.geotopo_dino.tests.smoke_pixel_fsdp \
  --output-dir /tmp/geotopo_pixel_fsdp_1gpu

torchrun --standalone --nproc-per-node=2 --module \
  methods.geotopo_dino.tests.smoke_pixel_fsdp \
  --output-dir /tmp/geotopo_pixel_fsdp_2gpu
```

This exercises the actual ViT-small, xFormers, mixed precision, auxiliary FSDP,
gradient clipping, teacher EMA, rank synchronization, and checkpoint restoration.
Before a full run, also use the existing 20-update data-pipeline smoke command
with `gcvd_pixel.yaml` and a fresh output directory.
