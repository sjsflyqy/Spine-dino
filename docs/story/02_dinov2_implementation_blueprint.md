# 从现有 DINOv2 实现 GeoTopo-DINO：代码补充蓝图

> 本文只描述应怎样修改，不修改当前代码。依据的是 `dinov2-main` 中现有 `augmentations.py`、`collate.py`、`masking.py`、`train.py` 和 `ssl_meta_arch.py`。

## 1. 现有代码中必须改变的依赖

当前流程：

```text
augmentation
  → collate 中提前生成 block masks
  → teacher unmasked forward
  → student masked forward
  → DINO + iBOT
```

目标流程：

```text
augmentation 返回图像 + 几何参数
  → collate 只整理图像和坐标
  → teacher unmasked forward
  → 从 teacher dense tokens 在线生成 topology mask
  → student masked forward
  → global DINO + GCVD + TGSR
```

最重要的改动是将 topology mask generation 从 CPU collate 移到 `SSLMetaArch.forward_backward()` 中 teacher backbone 之后、student backbone 之前。

---

## 2. 建议新增的文件

```text
dinov2/
├── data/
│   ├── geometry_augmentations.py     # 返回 crop/resize/flip 的 3×3 变换
│   └── topology_masking.py           # teacher response → centerline → span mask
├── layers/
│   ├── geometry_warp.py              # teacher anchor feature → local grid
│   └── ordered_part_pooling.py       # centerline parts 与 segment pooling
├── loss/
│   ├── geometry_distillation_loss.py # dense/region correspondence
│   ├── ordered_relation_loss.py      # local-local direction/ranking
│   └── topology_relation_loss.py     # masked-visible relation reconstruction
└── train/
    └── ssl_meta_arch.py              # 组装完整前向与 loss
```

首版也可以先把逻辑集中写在两三个文件中；上述拆分是稳定后建议的最终结构。

---

## 3. `augmentations.py`：返回图像之外的几何信息

### 3.1 自定义 transform

使用 `RandomResizedCrop.get_params()` 获取真实参数，再调用 functional API：

```python
class RandomResizedCropWithGeometry:
    def __call__(self, image):
        top, left, height, width = RandomResizedCrop.get_params(...)
        crop = TF.resized_crop(image, top, left, height, width, out_size)
        crop, flipped = maybe_horizontal_flip(crop)
        T = compose_crop_resize_pad_flip_matrix(...)
        return crop, {
            "box_xyxy": ...,
            "transform": T,          # source pixel -> view pixel
            "inverse_transform": inv(T),
            "flipped": flipped,
            "valid_pixel_mask": ...,
        }
```

不要只保存 `(top,left,height,width)`：当 anchor 使用 padding、local 使用 resize/flip 时，完整 $3\times3$ 仿射矩阵更不容易出错。

### 3.2 Anchor view

新增 `FullFOVAnchorTransform`：

1. 等比例缩放完整原图；
2. padding 到 $518\times518$；
3. 返回 padding-aware valid mask；
4. 几何变换保持可逆；
5. photometric augmentation 可与 student/teacher 原流程一致，但 geometry 必须固定且可记录。

推荐输出字典：

```python
output = {
    "global_crops": [anchor, random_global],
    "global_crops_teacher": [anchor, random_global],
    "local_crops": local_crops,
    "global_geometry": [anchor_meta, random_global_meta],
    "local_geometry": local_meta,
}
```

### 3.3 Local crop 策略

首轮实验保持你现有的 crop number/scale，避免采样策略和新 loss 同时改变。待核心方法有效后，再比较：

- 8 个相同尺度 local；
- 4 个 micro + 4 个 meso local；
- local 必须完全位于 anchor valid region；
- local 覆盖/重叠率受控采样。

---

## 4. `collate.py`：只整理视图与映射，不在线推断拓扑

新增 collated 字段：

```python
{
    "collated_global_crops": Tensor[2B, 3, 518, 518],
    "collated_local_crops": Tensor[KB, 3, 196, 196],
    "global_transforms": Tensor[2B, 3, 3],
    "local_transforms": Tensor[KB, 3, 3],
    "global_valid_masks": Tensor[2B, 37, 37],
    "sample_ids_global": Tensor[2B],
    "sample_ids_local": Tensor[KB],
    "view_ids_local": Tensor[KB],
}
```

保留 DINOv2 当前 view-major 顺序：先堆叠所有样本的 local-0，再 local-1，以便 `.chunk(n_local_crops)` 仍能工作。但必须显式保存 `sample_ids`，避免后续 local pair 配错样本。

当 `topology_mask.enabled=true` 时：

- collate 不再生成最终 `collated_masks`；
- 可保留标准 `MaskingGenerator` 作为 warm-up/fallback；
- mask ratio 等参数传给模型内的 topology generator。

---

## 5. `geometry_warp.py`：Local–Global patch 对齐

### 5.1 核心接口

```python
def warp_anchor_features_to_local(
    anchor_features,          # [B, D, 37, 37]
    anchor_transform,         # [B, 3, 3]
    local_transform,          # [KB, 3, 3]
    local_sample_ids,         # [KB]
    out_hw=(14, 14),
    anchor_valid_mask=None,
):
    ...
    return aligned_teacher, valid_local
```

### 5.2 坐标计算

对于 local grid center $p_l$：

```text
p_source = inverse(T_local) @ p_local
p_anchor = T_anchor @ p_source
p_anchor_patch = p_anchor / patch_size
```

随后将坐标归一化到 `grid_sample` 所需的 $[-1,1]$。需统一 `align_corners` 约定，并在单元测试中用人工棋盘图验证，不能仅凭可视化判断。

### 5.3 必须处理的边界

- anchor padding；
- crop 超出 valid region；
- independent horizontal flip；
- patch center 而非 patch corner；
- `H/W` 与代码中变量命名顺序；
- view-major batch indexing；
- 518/196 均可被 patch size 14 整除。

---

## 6. `topology_masking.py`：在线生成脊柱连续掩码

### 6.1 推荐接口

```python
class SpineTopologyMaskGenerator(nn.Module):
    @torch.no_grad()
    def forward(
        self,
        teacher_patch_tokens,   # [B, 37*37, D]
        teacher_cls_tokens,     # [B, D]
        valid_masks,            # [B, 37, 37]
        target_ratios,          # [B]
        progress,               # 0~1
    ):
        return {
            "masks": masks,                 # [B, 1369]
            "part_masks": part_masks,       # [B, Kp, 1369]
            "part_valid": part_valid,
            "centerline": centerline,
            "confidence": confidence,
            "used_fallback": used_fallback,
        }
```

该模块无可学习参数，输入必须 `.detach()`，不允许梯度通过离散 mask 回到 teacher。

### 6.2 具体算法

1. reshape patch tokens 为 `[B,37,37,D]`；
2. 计算 patch–CLS cosine response；
3. 对 response 做 $3\times3$ 或 $5\times5$ 平滑；
4. 通过动态规划寻找纵向连续高响应路径；
5. 对 centerline 做一维平滑；
6. 在路径左右扩展固定/自适应 band；
7. 沿弧长分为 6–8 个 ordered parts；
8. 采样相邻的 1–3 个 parts 形成 span；
9. 按 target mask ratio 调整 band 宽度或补充 block mask；
10. 低 confidence 样本回退到原 `MaskingGenerator`。

### 6.3 置信度建议

可综合：

- centerline 上的平均 response；
- centerline response 与全图平均 response 的差；
- 路径跳变大小；
- spine band 中有效 patch 比例。

训练日志必须记录 `topology_mask_usage_rate`、`fallback_rate`、平均 span 长度和实际 mask ratio。否则模型失败时无法判断是 loss 问题还是 mask 质量问题。

---

## 7. `ssl_meta_arch.py`：重排完整前向

### 7.1 新增子网络

在 student/teacher `ModuleDict` 中增加相同键：

```python
student_model_dict["geom_head"] = ProjectionHead(768, 256)
teacher_model_dict["geom_head"] = ProjectionHead(768, 256)

student_model_dict["part_head"] = ProjectionHead(768, 256)
teacher_model_dict["part_head"] = ProjectionHead(768, 256)

student_model_dict["relation_head"] = PairRelationHead(...)
teacher_model_dict["relation_head"] = PairRelationHead(...)  # 可不参与 forward
```

原因是当前 `prepare_for_distributed_training()` 和 `update_teacher()` 假定 student/teacher 具有完全相同的键。若只给 student 添加 head，会破坏 FSDP 包装和 EMA 循环。

同时在 `compute_precision.student/teacher` 下增加三个 head 的配置，否则 FSDP 初始化会查不到键。

### 7.2 推荐前向伪代码

```python
def forward_backward(self, batch, teacher_temp, progress):
    anchor, random_global = split_global_views(batch)
    local_crops = batch["collated_local_crops"]

    # 1. Teacher first: all unmasked
    with torch.no_grad():
        teacher_out = self.teacher.backbone(
            cat(anchor, random_global),
            masks=None,
            is_training=True,
        )
        t_anchor_cls, t_anchor_patch = select_anchor_tokens(teacher_out)

        # 2. Teacher-conditioned mask generated inside model
        topo = self.topology_mask_generator(
            t_anchor_patch.detach(),
            t_anchor_cls.detach(),
            batch["anchor_valid_masks"],
            target_ratios=sample_mask_ratios(...),
            progress=progress,
        )
        global_masks = combine_anchor_and_random_global_masks(topo, fallback_generator)
        mask_indices, mask_weights = flatten_mask_metadata(global_masks)

        # 3. Prepare DINO/iBOT/GCVD teacher targets
        teacher_targets = build_teacher_targets(
            teacher_out, mask_indices, batch, topo
        )

    # 4. Student sees masked globals and unmasked locals
    s_global, s_local = self.student.backbone(
        [global_crops, local_crops],
        masks=[global_masks, None],
        is_training=True,
    )

    # 5. Existing global DINO
    loss_global = global_dino_loss(s_global.cls, teacher_targets.global_cls)

    # 6. GCVD local-global alignment
    aligned_t, valid = warp_anchor_features_to_local(
        t_anchor_patch, batch.geometry, out_hw=(14, 14)
    )
    loss_dense = geometry_distill(s_local.patch, aligned_t, valid)
    loss_region = region_distill(s_local, aligned_t, valid)
    loss_order = ordered_local_relation_loss(s_local, batch.local_geometry)

    # 7. TGSR masked prediction + relation reconstruction
    loss_span_ibot = ibot_loss(s_global.patch, teacher_targets.masked_patch)
    loss_topo_rel = topology_relation_loss(
        s_global.patch, t_anchor_patch, topo.part_masks, global_masks
    )

    loss = (
        loss_global
        + w_dense * loss_dense
        + w_region * loss_region
        + w_order * loss_order
        + w_ibot * loss_span_ibot
        + w_topo * loss_topo_rel
    )
    self.backprop_loss(loss)
    return metrics
```

### 7.3 iBOT teacher target顺序

当前代码在 teacher backbone 输出后，根据 `mask_indices_list` 选择 teacher patch tokens。因此在线 topology mask 是可行的，但要将以下变量的构造移动到 teacher backbone 之后：

- `masks`；
- `mask_indices_list`；
- `n_masked_patches_tensor`；
- `upperbound`；
- `masks_weight`。

teacher backbone 仍只运行一次，不需要为了 mask 再运行第二次 teacher。

---

## 8. Loss 文件建议

### 8.1 `GeometryDistillationLoss`

输入 student local tokens、aligned teacher tokens 和 valid mask，输出：

- patch cosine loss；
- pooled region cosine loss；
- valid correspondence 数量。

初版不要使用大 prototype head。先用 L2-normalized 256-d projection + cosine，稳定后再比较 prototype CE。

### 8.2 `OrderedRelationLoss`

pair 构造必须限制在同一原图。建议对每张图随机采样若干 local pairs，而不是使用全部 $K(K-1)$ pairs，控制显存和类别不平衡。

标签：

```text
vertical_interval_i entirely above j  -> superior
vertical intervals overlap           -> overlap
vertical_interval_i entirely below j  -> inferior
```

距离排序只在局部窗口中构造，避免要求整条脊柱的 feature similarity 全局单调。

### 8.3 `TopologyRelationLoss`

使用 masked-part 到 visible-part 的 relation rows。必须排除：

- 无有效 teacher part 的样本；
- padding parts；
- 全 mask 或无 visible neighbor 的异常情况。

首版使用 cosine relation + Smooth-L1；若稳定，再尝试 temperature-softmax 后 KL。

---

## 9. 配置草案

```yaml
geotopo:
  enabled: true

  anchor:
    mode: full_fov_letterbox
    size: 518
    patch_size: 14

  gcvd:
    enabled: true
    projection_dim: 256
    dense_weight: 1.0
    region_weight: 0.5
    order_weight: 0.1
    use_standard_local_dino: false

  local_relation:
    direction_classes: 3
    use_distance_ranking: true
    max_pairs_per_image: 12

  topology_mask:
    enabled: true
    num_parts: 8
    span_parts_min: 1
    span_parts_max: 3
    band_half_width: 3
    warmup_epochs: 5
    ramp_epochs: 10
    max_topology_probability: 0.7
    fallback_to_block: true

  tgsr:
    ibot_weight: 1.0
    topology_relation_weight: 0.2
```

这些数值仅是代码起跑点，不是论文最终超参数。正式训练前应在短 schedule 上分别观察各 loss 数值和梯度规模。

---

## 10. 推荐实施顺序

### 阶段 A：只实现 geometry metadata

- transform 返回矩阵；
- collate 正确配对 sample/view；
- 棋盘图 warp 测试完全通过。

### 阶段 B：只实现 GCVD

- full-FOV anchor；
- teacher ROI/dense warp；
- local-global cosine loss；
- 不改 masking。

### 阶段 C：实现最简单 vertical span mask

- 不使用 teacher centerline；
- 沿 valid region 的纵向连续遮挡；
- 验证在线 mask→student→iBOT 数据流。

### 阶段 D：teacher-derived topology

- response、centerline、ordered parts；
- confidence 与 fallback；
- 可视化并人工检查 500 张。

### 阶段 E：关系学习

- local-local direction/ranking；
- masked-visible topology relation；
- 分别消融，未带来独立增益的项删除。

---

## 11. 必须提前写的测试

1. **Identity transform test**：local 等于 anchor 时，warp 后 feature grid 应逐点一致。
2. **Crop-resize test**：人工棋盘区域映射误差不超过一个 feature-cell 的可接受阈值。
3. **Flip test**：local flip 后 teacher target 同步反转。
4. **Padding test**：padding 区域全部被 valid mask 排除。
5. **Batch pairing test**：view-major stack 后所有 local 仍对应正确原图。
6. **Mask ratio test**：实际比例落在配置范围内。
7. **Contiguity test**：topology span 在 part graph 上必须连通。
8. **Fallback test**：低置信 teacher map 不产生空 mask 或异常索引。
9. **No-teacher-gradient test**：GCVD/TGSR target 不向 teacher 传播梯度。
10. **Distributed shape test**：两卡/八卡下 mask indices 和 center update 正常。

---

## 12. 预计风险

| 风险 | 处理方式 |
|---|---|
| Full-FOV anchor 改变了 baseline view distribution | 增加“相同 anchor、关闭新 loss”控制组 |
| Teacher response 不等于 spine segmentation | 只称 self-discovered topology proxy；量化 coverage 和 fallback |
| GCVD 只是已有 region matching | 必须加入 local-local ordered relation，并证明 partial-FOV/解剖身份收益 |
| Topology mask 只是 vertical block mask | 比较 vertical span 与 curved teacher span；加入 relation reconstruction |
| 新增 loss 相互冲突 | 记录梯度 cosine 与单模块/组合消融 |
| 518 + 8 locals 显存过大 | 开发阶段先 ViT-S/较少 locals；最终只跑关键 ViT-B 模型 |

