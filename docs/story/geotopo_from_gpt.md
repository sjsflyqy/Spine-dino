是。看你现在 GitHub `main` 上的代码后，我建议**从 DINOv2 baseline 正式分叉创新代码，但不要直接改 `upstream/dinov2-main`**。你现在的 baseline 很干净，正适合作为固定参照：`ssl_meta_arch.py` 已经能直接拿到 teacher 的 `x_norm_patchtokens`，student local backbone 也会产生 patch tokens，所以 **GCVD 不需要改 ViT 主干本身**。

但第一步我建议比你说的“把模块一全部写完”再谨慎一点：

> **先完成 4.1–4.3，4.4 OLRL 暂时不接。**
>
> 而且 4.1–4.3 内部也分两步：
> **先只把 geometry correspondence 做到绝对正确 → 再接 GCVD loss。**

你的 attention map / cosine similarity map **现在不是实现 GCVD 的前置条件**。它们更适合后面验证 anatomical over-invariance，以及以后做 TGSR 时判断 teacher 是否真的找到脊柱。现在不用因为 map 阻塞代码。

---

## 一、我看你当前代码，GCVD 的可行性很高

目前 teacher 已经直接输出：

```python
teacher_backbone_output_dict["x_norm_patchtokens"]
```

也就是你需要的 teacher dense tokens；现在这些 patch tokens主要拿去算 iBOT。

student 同样已经有：

```python
student_local_backbone_output_dict
```

现在只取了：

```python
student_local_backbone_output_dict["x_norm_clstoken"]
```

用于标准 local→global DINO，但完全可以再取：

```python
student_local_backbone_output_dict["x_norm_patchtokens"]
```

用于 GCVD。

因此网络主干完全够用：

$$
F_t^a\in R^{B\times1369\times768}
$$

如果 anchor = 518：

$$
518/14=37,\qquad 37^2=1369
$$

local = 196：

$$
196/14=14,\qquad 14^2=196
$$

所以就是：

```text
Teacher anchor
[B, 1369, 768]
        ↓
[B, 37, 37, 768]

Student locals
[8B, 196, 768]
        ↓
[8B, 14, 14, 768]
```

尺寸上非常干净。

---

# 二、原来的 DINOv2 代码，我建议从现在起视为“只读 baseline”

我明确赞成你新建目录。

不要变成：

```text
upstream/dinov2-main/dinov2/
    data/
        augmentations.py     # 改
    train/
        ssl_meta_arch.py     # 改
    ...
```

否则三个月以后你会分不清：

* 什么是官方 DINOv2；
* 什么是你之前为 RAD-DINO 做的兼容；
* 什么是 GeoTopo-DINO；
* 哪一个 modification 导致结果变化。

建议改成：

```text
Spine-dino/
│
├── upstream/
│   ├── dinov2-main/                 # baseline，原则上不再修改
│   └── dinov3-main/
│
├── methods/
│   └── geotopo_dino/
│       ├── __init__.py
│       │
│       ├── configs/
│       │   └── vitb14_gcvd.yaml
│       │
│       ├── data/
│       │   ├── geometry_augmentations.py
│       │   └── collate.py
│       │
│       ├── layers/
│       │   ├── geometry_warp.py
│       │   └── projection_head.py
│       │
│       ├── losses/
│       │   └── gcvd_loss.py
│       │
│       ├── train/
│       │   ├── ssl_meta_arch.py
│       │   └── train.py
│       │
│       ├── tests/
│       │   └── test_geometry.py
│       │
│       └── tools/
│           └── visualize_correspondence.py
│
└── ...
```

**不要复制整个 DINOv2 仓库。**

你的 `methods/geotopo_dino` 应该 import：

```python
from dinov2....
```

然后只重写/继承确实变化的部分。

这样论文消融也非常方便：

```text
upstream/dinov2-main       = DINOv2 baseline
methods/geotopo_dino       = ours
```

甚至之后：

```text
configs/
    baseline.yaml
    gcvd.yaml
    gcvd_dense.yaml
    gcvd_order.yaml
    full.yaml
```

天然就是你的 ablation。

---

# 三、但是我建议对你现在的网络设计做一个小改动：MVP 阶段先把 anchor 做成“额外 teacher anchor”

你当前文档设计是：

```text
Anchor global
Random global
8 local
```

即把标准 DINO 的两个 random global 改成：

```text
1 full-FOV anchor
+
1 random global
```

最终模型可以这样。

但**第一轮实验我反而建议暂时保留标准 DINO 的两个 random global，再额外加一个 teacher-only full-FOV anchor。**

也就是：

```text
              ┌─ random global 1 ─┐
原图 ─────────┼─ random global 2 ─┼── 原来的 DINO/iBOT
              │
              ├─ full-FOV anchor ─── 只给 teacher，产生 GCVD spatial target
              │
              └─ 8 local crops ───── student GCVD
```

原因非常重要：

### 这样你第一次实验只改变一件事情：local target。

标准 DINO：

$$
local\rightarrow global\ CLS
$$

GCVD：

$$
local\rightarrow corresponding\ teacher\ region
$$

而：

* global sampling 不变；
* iBOT 不变；
* global-global DINO 不变；
* masking 不变。

结果一旦变好，你可以很有把握地说：

> 是 local target 改变造成的。

如果你第一版同时把一个 random global 换成 full-FOV anchor，那么 reviewer 或你自己都会问：

> 到底是 GCVD 有效，还是新的 crop policy 有效？

后面主模型为了节省计算，可以再把 anchor 合并成 global view。

---

# 四、第一阶段千万不要直接写 loss，先把 geometry 做对

这是整个模块最容易出现**“训练正常、loss正常下降、但实验完全错”**的地方。

你当前 augmentation 是：

```python
RandomResizedCrop
RandomHorizontalFlip
```

直接藏在 `torchvision.transforms.Compose` 里。

所以你现在根本不知道 local 实际用了：

```text
top
left
height
width
flip
```

必须改成自己调用：

```python
RandomResizedCrop.get_params()
```

然后保存 transform。

对于每个 local：

```text
source image
   ↓ crop
   ↓ resize 196×196
   ↓ optional horizontal flip
local
```

记录：

$$
T_{\mathrm{local}}:
source\rightarrow local
$$

对于 anchor：

```text
source
↓ keep aspect ratio resize
↓ letterbox padding
anchor 518×518
```

记录：

$$
T_{\mathrm{anchor}}:
source\rightarrow anchor
$$

最后：

$$
p_{\text{anchor}}
=
T_{\text{anchor}}
T_{\text{local}}^{-1}
p_{\text{local}}
$$

这个数学关系没问题。

---

# 五、这里还有一个很容易漏掉的细节：不是简单 `/14`

你文档里写：

> 除以 patch size 14，得到 feature grid 坐标。

概念没错，但代码不能简单：

```python
x_feature = x_pixel / 14
```

因为 ViT patch token 对应的是 **patch center**。

第 0 个 token 大约对应：

$$
x=7
$$

而不是：

$$
x=0
$$

所以要严格统一：

### pixel-center convention

和：

```python
grid_sample(..., align_corners=False)
```

否则很容易整体错半个 patch。

半个 patch 就是：

$$
7\ pixels
$$

对于关键点/局部表示研究，这个误差已经不小了。

所以一定要测试：

```text
identity
crop only
resize
horizontal flip
letterbox
crop + resize + flip + letterbox
```

全部必须精确。

---

# 六、我甚至建议第一阶段做一个很直观的验证图

假设一个 local：

```text
196 × 196
```

有：

```text
14 × 14
```

个 patch center。

把这 **196 个 patch center 全部反投影到 anchor 原图上**。

你应该得到：

```text
Anchor X-ray

┌─────────────────────────┐
│                         │
│      • • • • • •        │
│      • • • • • •        │
│      • • • • • •        │
│      • • • • • •        │
│                         │
└─────────────────────────┘
```

然后同时画出这个 local 的真实 crop box。

### 所有点必须严格落在这个 crop 对应的位置里。

如果 local 被 horizontal flip：

左侧 patch 映回原图时必须变成右侧。

这个 visualization 通过以后，再谈 `grid_sample`。

---

# 七、然后才进入真正的 4.3

第二阶段再加：

```python
aligned_teacher = warp_anchor_features_to_local(...)
```

输出：

```text
[8B, 196, 768]
```

和：

```python
student_local_patch_tokens
[8B, 196, 768]
```

一一对应。

第一版甚至可以暂时直接做：

$$
1-\cos(F_s,\operatorname{sg}(F_t))
$$

### 先不加 256 projection head。

也就是：

```python
loss = 1 - cosine_similarity(
    student_local_patch,
    teacher_aligned_patch.detach(),
)
```

原因不是说 projection head 不需要。

而是第一轮我们在验证：

> **对应区域监督到底有没有效果。**

768→256 projection head 又增加了一套变量。

等 raw-token 版本证明训练链没问题，再加：

```text
768
↓
MLP
↓
256
↓
L2 normalize
```

做正式版本。

这样 debugging 会轻很多。

---

# 八、如果以后加入 projection head，你的当前 FSDP 代码还有一个特殊要求

你当前：

```python
self.student = nn.ModuleDict(...)
self.teacher = nn.ModuleDict(...)
```

teacher 更新是：

```python
for k in self.student.keys():
    ...
```

也就是说：

### student / teacher 的 key 必须一致。



所以正式加入：

```python
student["geom_head"]
```

必须同时：

```python
teacher["geom_head"]
```

然后：

```text
student geom_head
        ↓ EMA
teacher geom_head
```

这反而很适合 GCVD。

但你的：

```python
prepare_for_distributed_training()
```

又会访问：

```python
cfg.compute_precision.student[k]
cfg.compute_precision.teacher[k]
```

所以配置中也必须新增：

```yaml
compute_precision:
  student:
    geom_head: ...
  teacher:
    geom_head: ...
```

否则 FSDP 会直接报 key 不存在。

所以我建议**先 raw 768 cosine，再 256 head**。

---

# 九、取消 standard local-DINO 时还有一个隐藏 bug，必须提前告诉 Codex

你现在标准 DINO 是：

```python
dino_local_crops_loss
```

这里。

当然要取消。

但不能只是删除：

```python
loss_accumulator += dino_local_crops_loss
```

因为你的 global DINO 当前也除以：

```python
n_global_crops_loss_terms + n_local_crops_loss_terms
```



假设：

```text
8 locals
2 globals
```

那么你把 local loss 删除，但 denominator 不改：

### global DINO 权重会突然被缩小很多。

这会产生一个非常隐蔽的错误。

GCVD 版本应该把 global-global DINO 重新归一化为：

$$
n_{\text{global terms}}
$$

而不是再包含 local terms。

这个一定要让 Codex 写测试/注释。

---

# 十、iBOT 第一阶段完全不要碰

现在的：

```text
MaskingGenerator
collated_masks
mask_indices_list
iBOT loss
```

全部保持原样。你当前训练入口就是在 collate 中产生这些 mask。

暂时：

$$
L
=
L_{\mathrm{DINO}}^{G-G}
+
L_{\mathrm{iBOT}}
+
\lambda_gL_{\mathrm{GCVD}}
+
L_{\mathrm{KoLeo}}
$$

就够了。

**不要现在动 topology masking。**

否则 GCVD 出问题还是 mask 出问题根本分不清。

---

# 十一、4.4 OLRL 我现在建议暂缓

不是说不做。

而是当前：

```text
superior / overlap / inferior
distance ranking
```

我之前提到过，它和已经存在的 spatial relation SSL 工作有 novelty overlap 风险。

所以代码开发顺序应该是：

1. **Geometry metadata**
2. **Geometry correctness visualization**
3. **teacher anchor → local feature warp**
4. **GCVD dense loss**
5. **GCVD region pooled loss**
6. 短训练验证
7. 再决定 relation modeling 最终怎么设计
8. 最后才 OLRL / TGSR

这也是我认为你现在最快的路线。

---

# 十二、我建议你现在先建立 Git 分支

如果当前工作已经提交：

```bash
cd ~/spine-dino

git status
git switch -c feature/gcvd
```

然后就让：

```text
upstream/
```

保持不动。

你的新代码只进入：

```text
methods/geotopo_dino/
```

每完成一个阶段 commit 一次：

```text
feat: add geometry-aware crop metadata
feat: add anchor-local correspondence warp
feat: add GCVD dense loss
feat: integrate GCVD training
```

之后你做论文 ablation 和 code review 会轻松很多。

---

# 十三、第一轮先这样让 Codex 写，不要一次把 GCVD 全塞给它

下面这一段我建议你**直接复制给 Codex**。

你现在位于 `~/spine-dino` 仓库。

目标：开始实现 GeoTopo-DINO 的第一阶段——GCVD 的 geometry infrastructure。当前阶段只实现几何信息记录、anchor/local correspondence、可视化与单元测试，暂时不要实现 GCVD loss、OLRL、TGSR，也不要修改 DINO/iBOT 的训练目标。

### 总体原则

1. `upstream/dinov2-main/` 作为 baseline，原则上保持不修改。
2. 新建 `methods/geotopo_dino/`，所有创新代码放在这里。
3. 尽量复用 `upstream/dinov2-main` 中已有的 DINOv2 dataset、augmentation utilities、training utilities。
4. 不复制整个 DINOv2 源码。
5. 当前阶段不得改变现有 baseline 的训练行为。
6. 所有 geometry 使用统一的 pixel-center coordinate convention，并明确记录在代码注释中。
7. 所有矩阵统一定义为 `source image pixel -> transformed view pixel` 的 3×3 homogeneous transform。
8. horizontal flip、crop、resize、letterbox padding 必须包含在变换矩阵中。

### 新建目录

创建：

`methods/geotopo_dino/`

至少包含：

`data/geometry_augmentations.py`

`data/collate.py`

`layers/geometry_warp.py`

`tests/test_geometry.py`

`tools/visualize_correspondence.py`

以及必要的 `__init__.py`。

### 数据增强

保留现有 DINOv2 的两个 random global crops 和 local crops，用于保证 baseline 行为不变。

额外生成一个 teacher-only full-FOV anchor：

* 保持原图 aspect ratio；
* resize 后 letterbox 到 518×518；
* 不随机裁掉原始 FOV；
* 返回 3×3 `anchor_transform`；
* 返回 padding-aware `anchor_valid_mask`。

local crop 不再使用无法取得参数的纯 `transforms.Compose(RandomResizedCrop(...))`。

使用 `RandomResizedCrop.get_params()` 显式获得：

* top
* left
* height
* width

再调用 torchvision functional API 完成 crop 和 resize。

horizontal flip 同样显式采样并记录。

每个 local 必须返回：

* transformed image；
* `local_transform`：source image → local image；
* crop box；
* flip flag；
* sample/view index。

原有 photometric augmentation、Gaussian blur、normalization 尽量保持和 baseline 一致。

### Collate

实现独立的 GCVD collate，不修改 baseline `dinov2/data/collate.py`。

保持 DINOv2 当前的 view-major layout：

先 batch 中所有样本的 local-0，
再所有样本的 local-1，
依次类推。

需要返回现有 baseline 所需字段，同时额外返回：

* `collated_anchor_crops`
* `anchor_transforms`
* `anchor_valid_masks`
* `local_transforms`
* `local_sample_ids`
* `local_view_ids`

必须写 assertion 检查 metadata 和 `collated_local_crops` 的排列完全一致。

### Geometry warp

实现：

`warp_anchor_features_to_local(...)`

目标接口类似：

```python
aligned_teacher, valid_mask = warp_anchor_features_to_local(
    anchor_features,
    anchor_transforms,
    local_transforms,
    local_sample_ids,
    anchor_valid_mask,
    patch_size=14,
    local_grid_size=(14, 14),
)
```

输入假设：

* anchor token grid：37×37；
* local token grid：14×14；
* feature dim 任意。

对每个 local patch center：

1. local pixel coordinate；
2. 通过 `inverse(T_local)` 映回 source；
3. 通过 `T_anchor` 映射到 anchor；
4. 正确转换到 anchor patch-feature coordinate；
5. 使用 `torch.nn.functional.grid_sample`；
6. 明确固定 `align_corners=False`；
7. 排除 anchor padding、越界点以及无效 correspondence。

特别注意 patch center offset，不能简单使用 `pixel_coordinate / 14` 而忽略 patch 中心。

### 单元测试

至少完成以下测试：

* identity mapping；
* pure crop；
* crop + resize；
* horizontal flip；
* letterbox；
* crop + resize + flip + letterbox；
* batch 中多个 sample；
* view-major local ordering。

建议使用 synthetic coordinate image / checkerboard，而不是医学影像来验证数值正确性。

测试应检查 projected coordinates 的数值误差，而不仅仅是人工看图。

### 可视化工具

实现一个无需训练的可视化脚本。

随机读取若干 SpinePretrain 图像，并保存：

1. 原始图；
2. full-FOV anchor；
3. 一个或多个 local crops；
4. local crop 的 14×14 patch centers 投影到 anchor 后的位置；
5. local crop boundary 在 anchor 上的位置。

如果 horizontal flip 生效，映射后的左右关系也必须正确。

输出图用于人工验证 geometry pipeline。

### 当前阶段禁止做的事情

不要：

* 修改 ViT backbone；
* 添加 GCVD loss；
* 添加 projection head；
* 添加 OLRL；
* 添加 topology masking；
* 修改 iBOT mask；
* 删除标准 local DINO loss；
* 改变 baseline 配置；
* 对 `upstream/dinov2-main/` 做大范围复制或修改。

### 完成后请返回

1. 新增/修改的文件清单；
2. 每个文件的职责；
3. geometry coordinate convention；
4. 单元测试结果；
5. 一条运行 `visualize_correspondence.py` 的实际命令；
6. 当前仍未实现的内容；
7. `git diff --stat`；
8. 检查确认 `upstream/dinov2-main/` 是否保持未修改。

不要继续实现下一阶段，等我确认 geometry visualization 和测试正确后再继续。

这一轮 Codex 做完以后，**不要马上叫它继续训练**。

你先运行：

```text
test_geometry.py
visualize_correspondence.py
```

把它生成的 **anchor + local crop + 14×14 映射点** 给我看。

只要这一步正确，下一轮我再给你一个 Codex prompt，接入：

$$
F_t^a
\rightarrow
grid\_sample
\rightarrow
\widetilde F_t^k
$$

以及：

$$
L_{\text{dense}}+L_{\text{region}}
$$

并正确删除 `local→whole-global DINO loss`、修正 global DINO loss normalization。

另外，你上面关于 DINO 的解释有一个小地方需要改：**DINO 的 centering vector 不是通过梯度“可学习”的参数，而是根据 teacher 输出做 EMA 更新的统计量**；你现在配置确实使用的是 `centering: "centering"`。 65536 维可以直观理解成 prototype assignment，但不要在论文里把每个维度严格写成一个固定、可解释的“视觉词类别”。

**所以现在不是“直接完成整个模块一”，而是先把模块一最关键、也最容易暗错的 geometry infrastructure 锁死。** 这一步通过后，4.1–4.3 接进去其实反而是比较简单的部分。
