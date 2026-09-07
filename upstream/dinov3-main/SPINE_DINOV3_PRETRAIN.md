# Spine-DINOv3 ViT-B/16 continued pretraining

This setup uses the official DINOv3 self-supervised implementation and keeps
the released ViT-B/16 architecture: patch size 16, RoPE, four storage tokens,
MLP FFN, masked K bias, and layernorm-bf16.

## Initialization conversion

```bash
python convert_dinov3_vitb16.py \
  --weights-path ../../weights/base/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --output-path ../../weights/initialization/dinov3_vitb16_lvd1689m_training_init.pth
```

The converter performs a strict 188-tensor architecture check before writing
the training checkpoint.

## Training

```bash
CUDA_VISIBLE_DEVICES=5 python train_spine_dinov3.py \
  --pretrained-weights ../../weights/initialization/dinov3_vitb16_lvd1689m_training_init.pth \
  --output-dir ../../outputs/upstream/pretrain/dinov3_vitb16_lvd1689m_spine \
  --ngpu 1 \
  --batch-size 4 \
  --epochs 100 \
  --epoch-length 200 \
  --global-crop-size 512 \
  --local-crop-size 192 \
  --global-crop-scale 0.50 1.00 \
  --local-crop-scale 0.20 0.50 \
  --local-crops-number 8
```

The command runs 20,000 optimization steps. It resumes automatically from the
latest recovery checkpoint in the same output directory. Add `--no-resume`
only when deliberately starting a new run.

Recovery checkpoints are written every 1,000 steps, the latest three are kept,
and permanent linked copies are retained every 5,000 steps. Consolidated EMA
teacher checkpoints are written every 2,000 steps under `eval/`.

## Storage-controlled pass training and staged resume

The launcher can calculate exact equivalent passes from the dataset size and
global batch. To train to 10 passes now while keeping one continuous 50-pass
LR/WD/EMA schedule for later 20- and 50-pass resumes, use the same output
directory and the same `--schedule-passes 50` in every stage:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_spine_dinov3.py \
  --pretrained-weights ../../weights/initialization/dinov3_vitb16_lvd1689m_training_init.pth \
  --output-dir ../../outputs/upstream/pretrain/dinov3_lvd1689m_schedule50p \
  --ngpu 2 \
  --batch-size 4 \
  --num-workers 4 \
  --prefetch-factor 1 \
  --target-passes 10 \
  --schedule-passes 50 \
  --checkpoint-period 10000 \
  --checkpoint-max-to-keep 2 \
  --checkpoint-keep-every 0 \
  --teacher-period 0 \
  --global-crop-size 512 \
  --local-crop-size 192 \
  --global-crop-scale 0.50 1.00 \
  --local-crop-scale 0.20 0.50 \
  --local-crops-number 8
```

`--teacher-period 0` still writes a consolidated EMA teacher at the end of
each requested stage. `--checkpoint-keep-every 0` disables permanent `_keep`
copies, while `--checkpoint-max-to-keep 2` retains only the latest two numeric
recovery checkpoints. To continue, rerun the same command without
`--no-resume`, changing only `--target-passes` to 20 and later 50. The launcher
stores `spine_run_plan.json` and rejects changes to the schedule horizon or
global batch that would make a resume inconsistent.

Do not first train with `--target-passes 10 --schedule-passes 10` and later
change the schedule to 50: that would rebuild the cosine schedules and cause a
learning-rate discontinuity.

## Extracting the final teacher for downstream tasks

```bash
python extract_dinov3_teacher.py \
  --checkpoint ../../outputs/upstream/pretrain/dinov3_vitb16_lvd1689m_spine/eval/training_19999/teacher_checkpoint.pth \
  --output ../../weights/pretrained/spine_dinov3_vitb16_teacher_20000.pth
```

The extracted file is a raw, strictly validated DINOv3 ViT-B/16 backbone and
can be passed to the existing downstream DINOv3 backbone adapter.
