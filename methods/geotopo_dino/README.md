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

## Gradient accumulation

Pretraining accepts `train.gradient_accumulation_steps` (a positive integer,
default `1`). For example, append this to the existing two-GPU command:

```bash
train.batch_size_per_gpu=8 \
train.gradient_accumulation_steps=4
```

The global microbatch remains `2 * 8 = 16` images; a complete optimizer update
averages four microbatches, giving an effective batch of `64`. Gradients are
cleared before each group, and clipping, AMP unscaling/scaler updates and the
optimizer step occur once at its end. Teacher parameter EMA runs once per
successful optimizer step, and is skipped on AMP overflow. A shortened final
group is averaged over its actual number of microbatches.

All existing iteration counts **remain microbatch counts**: `optim.total_iterations`,
`train.stop_after_iterations`, LR/teacher-temperature/weight-decay/momentum
schedules, last-layer freezing, mask curricula and reconstruction warmups.
Thus `29420` iterations at the same physical batch still process the same number
of images (the same passes). With accumulation `4`, they make `7355` optimizer
update attempts. Each update uses the LR, weight decay and EMA momentum at the
last microbatch of its group. `optim.base_lr` still uses the existing physical
global-batch scaling rule; accumulation does **not** automatically raise the LR.

Evaluation/checkpoint periods also count microbatches, but a period reached
inside a group is deferred to the end of that group. Checkpoints contain the
microbatch iteration, successful `optimizer_updates`, accumulation factor,
optimizer state and AMP scaler state. Resume requires the same accumulation
factor; legacy checkpoints use `1`. Evaluation-only loading is independent of
this factor. Resume preserves the original sampler behavior; it does not promise
an identical augmentation/data sequence to an uninterrupted run. Splitting a run
at a partial group also creates an extra optimizer update compared with keeping
that group intact, so choose stage boundaries divisible by the accumulation factor.

Logs retain the original, unscaled losses and add `gradient_accumulation_steps`,
`effective_batch_size` (smaller for a partial final group), and `optimizer_updates`.
FSDP gradients continue to synchronize on every microbatch; `no_sync()` is not
used, avoiding its extra full-gradient memory cost. Accumulation does not combine
teacher logits/features across microbatches: GCVD Sinkhorn, DINO/iBOT centering
and KoLeo still operate on each forward's inputs. It therefore does not reproduce
every effect of a physically larger batch.

An optional integration check uses synthetic 28x28 images and a small ViT with
GCVD and pixel reconstruction. It exercises the actual trainer, AMP, two-rank
FSDP, a shortened final group and exact parameter/scaler restoration on resume:

```bash
torchrun --standalone --nproc-per-node=2 --module \
  methods.geotopo_dino.tests.smoke_accumulation_fsdp \
  --output-dir /tmp/accumulation-fsdp
```

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

## GCVD warmup and loss switches

The canonical warmup values are `no_warmup`, `fraction`, and `fixed_iter`.
Warmup always uses the restored global microbatch iteration; it is not reset by
`stop_after_iterations` or by staged 10-pass launches.

```bash
# Immediate full weight
gcvd.warmup_type=no_warmup

# Fraction of the complete optim.epochs * OFFICIAL_EPOCH_LENGTH schedule
gcvd.warmup_type=fraction gcvd.warmup_fraction=0.05

# Fixed global microbatch iterations, independent of schedule length
gcvd.warmup_type=fixed_iter gcvd.warmup_iterations=1000
```

`gcvd.loss_type` selects the loss for both dense and region GCVD. `cosine`
uses the original normalized-feature cosine distances. `prototype_ce` uses
independent dense and region prototype heads with the same prototype count,
temperatures, teacher centering mode, and Sinkhorn iteration count. Region
features are mean-pooled over valid aligned patches before projection; crops
with no valid correspondence are excluded. Dense and region teacher assignments
are balanced separately, and centering mode keeps a separate center per branch.
The region head uses the existing `geom_head` precision/FSDP configuration.

```bash
gcvd.loss_type=cosine

gcvd.loss_type=prototype_ce \
  gcvd.head_n_prototypes=16384 \
  gcvd.teacher_temp=0.04 \
  gcvd.student_temp=0.1

# Use globally balanced assignments for both GCVD prototype teachers.
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
- region prototype diagnostics use `gcvd_region_teacher_*` and
  `gcvd_region_student_*`; `gcvd_region_raw_kl` subtracts region teacher entropy;
- `optimization_loss` and `total_loss` are the optimization objective before
  gradient-accumulation normalization.

The former ambiguous `gcvd_dense_loss` and `gcvd_region_loss` keys are no
longer emitted. Prototype experiments must use a new output directory with
`--no-resume` rather than resume a cosine checkpoint or an older prototype
checkpoint whose region head used cosine. The region head architecture has
changed; initialize the backbone using `student.pretrained_weights` as usual.

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
