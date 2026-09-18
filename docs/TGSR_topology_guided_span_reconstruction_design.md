# TGSR：Topology-Guided Span Reconstruction 设计说明与实现任务书

> 项目：Spine-DINO  
> 目标：在 DINOv2 / iBOT 框架中引入脊柱结构感知的掩码建模，使 masked image modeling 不再只使用随机 block mask，而是显式利用 teacher 中自动形成的脊柱结构响应，构造连续的 topology span，并进一步约束 masked span 与可见结构之间的拓扑关系。  
> 本文档既是方法设计说明，也是 Codex 实现时的上下文说明。  
> 当前阶段建议优先完成 **TGSR Masking MVP**，relation reconstruction 作为第二阶段实现。

---

## 1. 背景

Spine-DINO 当前以 DINOv2 为基础框架，包含：

- DINO global/local crop self-distillation；
- iBOT masked patch prediction；
- KoLeo regularization；
- 已扩展的 GCVD（Geometry-Conditioned View Distillation）；
- teacher 使用 EMA 更新；
- student 对 masked global crops 进行前向；
- teacher 对未遮挡 global crops 进行前向。

当前 iBOT 的 masking 仍然主要来自标准随机 block masking。

随机 block mask 的优点是简单、通用、不依赖先验，但对脊柱 X-ray 有一个明显问题：

> 它不知道图像中哪些 patch 属于脊柱结构，也不知道脊柱存在连续、链状、具有局部重复和长程拓扑关系的解剖组织。

因此，随机矩形 block 很可能：

- mask 到背景；
- mask 到软组织；
- mask 到图像边缘；
- 只遮住与下游脊柱任务无关的区域；
- 无法刻意破坏脊柱连续结构；
- 无法训练模型通过上下游可见解剖结构恢复中间缺失结构。

对于脊柱 X-ray，下游任务如：

- 椎体分类；
- 椎体关键点检测；
- 脊柱分割；
- 椎体定位；
- 畸形分析；

都高度依赖局部结构与连续解剖上下文。

因此希望将 iBOT 的 mask 从：

\[
\text{random spatial block}
\]

升级为：

\[
\text{anatomy-aware continuous topology span}
\]

---

# 2. TGSR 的核心目标

TGSR = **Topology-Guided Span Reconstruction**

它不是单纯“换一种 mask”。

完整目标是：

1. 从 teacher 中自动发现脊柱相关的结构响应；
2. 从结构响应中提取一个连续的、无需椎体标签的拓扑路径；
3. 沿拓扑路径选择连续区间进行 mask；
4. student 不仅恢复被 mask patch 的 latent representation；
5. 还恢复被 mask span 与可见上下游结构之间的组织关系。

最终形式：

\[
\boxed{
\text{Teacher structural response}
\rightarrow
\text{Topology path}
\rightarrow
\text{Continuous span mask}
\rightarrow
\text{iBOT latent recovery}
+
\text{Topology relation recovery}
}
\]

---

# 3. 为什么不能继续使用 CLS-patch cosine similarity

早期方案曾计划使用：

\[
r_i=\cos(z_{\mathrm{CLS}}, z_i)
\]

即利用 teacher CLS token 和 patch token 的 cosine similarity 构造 soft response map。

但实际可视化发现：

- cosine similarity 并不能稳定定位脊柱；
- AP 图中腹部、软组织、身体边缘常出现高响应；
- 侧位图中脊柱甚至可能处于低相似区域；
- 不同 layer 的 cosine response 差异大；
- 高 cosine 不应被解释为“属于脊柱的概率”。

因此：

> **不要再使用 CLS-patch cosine similarity 作为 TGSR 的核心 spine response。**

它可以保留作为分析工具，但不作为默认 mask 生成依据。

---

# 4. 当前可视化实验发现

目前对两个模型进行了 attention / cosine 可视化：

1. DINOv2 random initialization，训练约 100 passes；
2. MAIRA-2 initialization，继续训练约 50 passes。

主要观察：

### 4.1 Random-init 模型

中间层 attention，尤其约 Layer 6–10，出现明显的脊柱结构响应。

部分 attention heads 会：

- 沿椎体链形成连续响应；
- 对脊柱中央结构明显聚焦；
- 对椎体重复结构产生一串局部峰值。

Layer 8 的部分 head 表现尤其明显。

### 4.2 MAIRA-init 模型

attention 视觉上未必像 random-init 那么集中于脊柱，但 downstream performance 明显更好。

因此必须注意：

\[
\text{attention concentration}
\neq
\text{representation quality}
\]

“attention 看起来更像脊柱”不能作为模型优劣依据。

### 4.3 不同 head 的功能差异明显

某些 head：

- 关注脊柱；
- 关注椎体链；

而另一些 head：

- 关注边缘；
- 关注骨盆；
- 关注高密度区域；
- 关注软组织；
- 响应非常稀疏。

因此：

> 不能直接平均所有 heads。

TGSR 应考虑：

\[
\text{automatic structure-sensitive head selection}
\]

而不是固定某一个 head，也不是所有 head 无差别平均。

---

# 5. TGSR 设计原则

TGSR 应满足以下原则。

## 5.1 不依赖椎体标签

不能要求：

- C1；
- T1；
- L1；
- 固定 17 节椎体；

因为预训练数据包含不同 FOV：

- 颈椎；
- 胸椎；
- 腰椎；
- 全脊柱；
- 胸腰段；
- 部分脊柱；
- AP / LAT。

因此 TGSR 只建模：

\[
\text{relative topology}
\]

而不是具体椎体编号。

---

## 5.2 不假设脊柱永远竖直

由于数据增强和真实成像中可能存在：

- rotation；
- scoliosis；
- patient positioning difference；
- LAT view；
- 不同 crop；

不能使用：

> 从图像最上方到最下方逐行寻找中心点

这种严格纵向设计。

需要使用：

\[
\text{spatial graph / dominant continuous path}
\]

来描述脊柱。

---

## 5.3 不将 attention 当作 segmentation

teacher attention 只能解释为：

- structure-sensitive response；
- anatomical response prior；
- structural saliency；

不能写成：

> teacher 自动分割出了脊柱。

TGSR 只利用 attention 作为无监督结构先验。

---

## 5.4 必须保留 random mask

TGSR 不能 100% 替代 random block mask。

原因：

- teacher early stage 不稳定；
- attention 可能产生错误偏置；
- topology mask 可能形成 self-reinforcing bias；
- random mask 仍具有泛化性和 regularization 价值。

因此最终使用：

\[
M=
\text{Topology Mask}
\cup
\text{Random Block Mask}
\]

或者以概率混合：

\[
p_{\text{topology}}<1
\]

---

# 6. 总体前向流程

当前 TGSR 必须发生在 teacher forward 之后。

原因：

teacher attention / dense token 必须先计算，才能生成 topology mask。

正确流程：

```text
Input global crops
        |
        v
Teacher forward on UNMASKED crops
        |
        +--> teacher patch tokens
        |
        +--> selected intermediate attention maps
        |
        v
Topology response estimation
        |
        v
Topology path extraction
        |
        v
Continuous span selection
        |
        v
Topology/random mixed mask
        |
        v
Student forward on MASKED crops
        |
        +--> original DINO loss
        +--> original iBOT loss
        +--> GCVD loss
        +--> TGSR relation loss (second stage)
```

注意：

> 不允许为了 TGSR 再额外跑一次完整 teacher forward。

attention 必须在现有 teacher forward 中一起提取。

---

# 7. Stage 1：Teacher Structural Response

## 7.1 输入

从 teacher 的若干中间 ViT layers 提取：

\[
A_l^h
\]

其中：

- \(l\)：layer；
- \(h\)：attention head；
- 关注 CLS-to-patch attention。

第一版建议候选中间层：

```text
6, 7, 8, 9
```

或者：

```text
6, 7, 8, 9, 10
```

必须 config 可配置。

不要硬编码：

```text
Layer 8
Head 8
```

因为：

- 不同初始化可能不同；
- 不同训练阶段可能不同；
- 不同 backbone 深度可能不同。

---

# 8. Head Scoring

对于每个候选 head：

\[
A_l^h \in \mathbb{R}^{H_p\times W_p}
\]

计算一个无监督结构分数：

\[
q_l^h
\]

第一版至少考虑以下因素。

---

## 8.1 Response Concentration

希望 response 不是完全均匀。

例如可以使用：

- normalized variance；
- entropy；
- top-k mass concentration。

示意：

\[
q_{\text{conc}}
=
1-\frac{H(A)}{\log N}
\]

其中 \(N\) 为 patch 数。

---

## 8.2 Spatial Continuity

脊柱 response 应该具有一定空间连续性，而不是随机散点。

可以 threshold 后计算：

- largest connected component ratio；
- 邻接 patch response consistency。

例如：

\[
q_{\text{cont}}
=
\frac{
|\text{largest connected component}|
}{
|\text{active patches}|
}
\]

---

## 8.3 Border Penalty

很多 attention head 会关注：

- 黑边；
- collimation 边缘；
- marker；
- 图像两侧高对比区域。

因此需要惩罚边缘响应。

例如：

\[
q_{\text{border}}
=
1-
\frac{
\sum_{i\in B}A_i
}{
\sum_i A_i
}
\]

其中 \(B\) 是 patch grid 外围若干圈。

---

## 8.4 Optional: Elongation / Chain-likeness

脊柱整体偏连续长条 / 曲线结构。

可以对 threshold 后的 component 计算：

- PCA 主轴长度比；
- skeleton length；
- eccentricity。

第一版可以作为可选项，不强制。

---

## 8.5 综合 head score

第一版建议：

\[
q_l^h
=
\alpha q_{\text{conc}}
+
\beta q_{\text{cont}}
+
\gamma q_{\text{border}}
\]

或者乘积形式。

选取 top-k heads：

\[
\mathcal H^*=\text{TopK}(q_l^h)
\]

然后融合：

\[
R
=
\frac{
\sum_{(l,h)\in\mathcal H^*}
q_l^h A_l^h
}{
\sum_{(l,h)\in\mathcal H^*}q_l^h
}
\]

得到：

\[
R\in\mathbb R^{H_p\times W_p}
\]

作为 teacher structural response。

---

# 9. Response Preprocessing

对 \(R\) 进行：

1. valid-mask 去除 padding；
2. normalization；
3. optional Gaussian smoothing；
4. optional percentile clipping。

例如：

\[
\hat R
=
\frac{R-\min(R)}
{\max(R)-\min(R)+\epsilon}
\]

注意：

不能将 \(\hat R\) 命名为：

```text
spine probability
```

建议称：

```text
structural response
anatomical response
teacher structure response
```

---

# 10. Stage 2：Topology Path Extraction

TGSR 不应依赖固定 vertical axis。

将 patch grid 构建为图：

\[
G=(V,E)
\]

每个 patch 是一个 node。

使用：

- 4-neighborhood；
- 或 8-neighborhood。

建议 8-neighborhood。

---

## 10.1 Node Score

每个 node：

\[
s_i=\hat R_i
\]

高 structural response 的 patch 更值得进入 topology path。

---

## 10.2 Edge Cost

对相邻 patch \(i,j\)：

\[
c_{ij}
=
\lambda_d d(i,j)
-
\lambda_r\frac{s_i+s_j}{2}
\]

其中：

- \(d(i,j)\) 是 patch-grid distance；
- 高 response 降低 cost；
- 太曲折路径可以额外加入 curvature penalty。

可选：

\[
c_{ij}
=
\lambda_d d(i,j)
-
\lambda_r\frac{s_i+s_j}{2}
+
\lambda_\theta c_{\text{curve}}
\]

第一版可先不加 curvature。

---

# 11. Dominant Path

目标不是：

```text
top -> bottom
```

而是：

> 在高 structural-response 区域内找到一条足够长、足够连续的 dominant path。

可以考虑：

### 方法 A：threshold + largest component + skeleton

1. threshold response；
2. 取最大 connected component；
3. morphological skeleton；
4. 找 skeleton longest geodesic path。

优点：

- 简单；
- 易可视化；
- implementation 成本低。

第一版非常适合。

### 方法 B：graph shortest / longest path

在 weighted graph 上寻找：

- 最大累计 response path；
- 或最低 cost path。

第二版再做。

当前建议：

> MVP 先使用 component + skeleton + longest geodesic path。

---

# 12. Topology Confidence

必须有 confidence。

如果 topology extraction 不可靠，直接 fallback 到标准 block mask。

建议：

\[
C
=
C_{\text{response}}
\cdot
C_{\text{continuity}}
\cdot
C_{\text{nonborder}}
\cdot
C_{\text{length}}
\]

例如：

### Response confidence

\[
C_{\text{response}}
=
\frac{
\operatorname{mean}(R_{\text{path}})
}{
\operatorname{mean}(R_{\text{valid}})+\epsilon
}
\]

再映射到 \([0,1]\)。

### Continuity confidence

来自 component continuity。

### Non-border confidence

路径如果大量贴图像边缘，则降低。

### Length confidence

路径过短意味着不是有意义的脊柱结构。

若：

\[
C < \tau
\]

则：

```python
fallback = True
final_mask = standard_block_mask
```

必须记录 fallback ratio。

---

# 13. Stage 3：Relative Topology Coordinate

得到 path：

\[
P=\{p_1,p_2,\dots,p_K\}
\]

计算归一化累计弧长：

\[
s_k
=
\frac{
\sum_{i=1}^{k-1}\|p_{i+1}-p_i\|
}{
\sum_{i=1}^{K-1}\|p_{i+1}-p_i\|
}
\]

因此：

\[
s_k\in[0,1]
\]

这就是 relative topology coordinate。

它不表示：

- C1；
- T1；
- L1；

只表示：

> 当前可见脊柱结构中的相对位置。

---

# 14. Stage 4：Continuous Span Mask

随机采样：

\[
s_0\sim U(0,1-\Delta)
\]

以及 span length：

\[
\Delta\sim U(\Delta_{\min},\Delta_{\max})
\]

得到：

\[
[s_0,s_0+\Delta]
\]

选择 path 上：

\[
s_k\in[s_0,s_0+\Delta]
\]

的连续节点。

然后沿 path 两侧扩展 band width：

\[
w
\]

得到 topology span mask。

概念：

```text
visible path
   ●
   ●
   ●
masked span
   ×
   ×
   ×
visible path
   ●
   ●
```

不是随机矩形：

```text
██████
██████
```

---

# 15. Mask Ratio Matching

当前 iBOT 已有目标 mask 数量。

TGSR 不能随意改变整体 masked patch 数。

因此：

1. 先生成 topology span mask；
2. 若 patch 数小于 target：
   - 优先扩 band；
   - 或用原 candidate block mask 补充；
3. 若 patch 数大于 target：
   - 从 topology span 边缘裁剪；
   - 或随机选择 topology patches 保持连续性优先。

最终要求：

\[
|M_{\text{final}}|
\approx
|M_{\text{iBOT target}}|
\]

这样可以避免性能提升只是来自不同 mask ratio。

---

# 16. Curriculum Mixing

训练早期 teacher 结构 response 不稳定。

因此：

\[
p_{\text{topology}}(t)
\]

必须逐渐增加。

例如：

### warm-up

\[
t<0.05T
\]

使用：

\[
p_{\text{topology}}=0
\]

### ramp

\[
0.05T\le t<0.20T
\]

线性从 0 增加到：

\[
p_{\max}=0.6
\]

### stable stage

\[
t\ge0.20T
\]

保持：

\[
p_{\text{topology}}=0.6
\]

仍然保留约 40% standard block masking。

Config 示例：

```yaml
tgsr:
  enabled: true

  attention_layers: [6, 7, 8, 9]
  head_topk: 4

  warmup_fraction: 0.05
  ramp_fraction: 0.15
  max_topology_probability: 0.60

  response:
    use_concentration_score: true
    use_continuity_score: true
    use_border_penalty: true

  topology:
    threshold_percentile: 75
    min_path_length_ratio: 0.25
    confidence_threshold: 0.45

  span:
    min_length_ratio: 0.15
    max_length_ratio: 0.35
    band_width_patches: 2

  fallback:
    use_standard_block_mask: true
```

---

# 17. Stage 5：Masked Latent Reconstruction

这一部分 **不要重新设计**。

当前 iBOT 已经负责：

> student 对 masked patch 预测 teacher unmasked patch latent target。

因此：

\[
\mathcal L_{\text{iBOT}}
\]

本身就是 masked latent reconstruction。

TGSR 的第一阶段只需要：

> 将 iBOT 原本的 random block mask 替换 / 混合为 topology-aware span mask。

因此：

\[
\boxed{
\text{TGSR Masking MVP}
=
\text{Topology Mask Generator}
+
\text{Original iBOT Loss}
}
\]

不需要新增第二个重复 latent reconstruction loss。

---

# 18. Stage 6：Topology Relation Reconstruction

这是 TGSR 第二阶段。

目的：

> 不仅让 student 恢复 masked patch 本身，还恢复被遮挡结构与周围可见结构之间的组织关系。

这部分是 TGSR 不只是“换 mask”的关键。

---

# 19. Relative Parts

沿 topology path 按弧长划分为 \(K\) 个 relative parts：

\[
P_1,P_2,\dots,P_K
\]

例如：

```text
P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8
```

这些不是椎体标签。

即：

```text
P1 != C1
P4 != T4
P8 != L5
```

它们只是：

> 当前 crop 中按 topology coordinate 划分的相对结构片段。

例如：

\[
P_k=
\{p_i\mid s_i\in[(k-1)/K,k/K)\}
\]

---

# 20. Teacher Part Features

teacher 看 unmasked anchor。

对于 part \(P_k\)：

\[
z_k^T
=
\frac{1}{|P_k|}
\sum_{i\in P_k} t_i
\]

其中：

\[
t_i
\]

是 teacher dense patch token。

可选使用：

- mean pooling；
- attention-weighted pooling。

MVP 推荐 mean pooling。

---

# 21. Student Part Features

student 看 masked crop。

同样：

\[
z_k^S
=
\frac{1}{|P_k|}
\sum_{i\in P_k} s_i
\]

其中：

\[
s_i
\]

为 student patch representation。

---

# 22. Part Relation Matrix

定义 teacher relation：

\[
R_{ij}^T
=
\cos(z_i^T,z_j^T)
\]

student relation：

\[
R_{ij}^S
=
\cos(z_i^S,z_j^S)
\]

因此：

\[
R^T,R^S\in\mathbb R^{K\times K}
\]

relation matrix 描述：

> 不同 topology parts 在 latent space 中的结构关系。

---

# 23. 只约束 Masked-to-Visible Relations

不要监督全部 \(K\times K\)。

定义：

- \(\mathcal M\)：被 topology span 覆盖的 parts；
- \(\mathcal V\)：仍然可见的 parts。

只计算：

\[
(i,j),\quad
i\in\mathcal M,\;
j\in\mathcal V
\]

即：

> 被遮挡结构与上下游可见结构之间的关系。

损失：

\[
\mathcal L_{\text{rel}}
=
\frac{1}{|\Omega|}
\sum_{(i,j)\in\Omega}
\rho(R_{ij}^S-R_{ij}^T)
\]

其中：

\[
\Omega=\mathcal M\times\mathcal V
\]

\(\rho\) 可用：

- L1；
- Smooth L1；
- MSE。

第一版建议：

```text
Smooth L1
```

---

# 24. 为什么 Relation Loss 不建议直接做 Prototype CE

GCVD 中 prototype CE 合理，因为 GCVD 处理的是：

> 跨 view token / prototype distribution alignment。

TGSR relation reconstruction 处理的是：

> 结构片段之间的相对关系。

它天然是连续值：

\[
[-1,1]
\]

因此 cosine relation + SmoothL1 / L1 更直接。

不要机械照搬 DINO/iBOT 的 cross-entropy。

---

# 25. TGSR 完整 Loss

最终：

\[
\mathcal L
=
\mathcal L_{\text{DINO}}
+
\lambda_{\text{iBOT}}\mathcal L_{\text{iBOT}}
+
\lambda_{\text{KoLeo}}\mathcal L_{\text{KoLeo}}
+
\lambda_{\text{GCVD}}\mathcal L_{\text{GCVD}}
+
\lambda_{\text{rel}}\mathcal L_{\text{TGSR-rel}}
\]

注意：

TGSR 本身的 mask generation 不需要一个额外 loss。

它通过改变 iBOT mask 影响训练。

---

# 26. 为什么要先做 Mask-only，再做 Relation

不要一次性实现所有 TGSR。

推荐实验顺序：

```text
Baseline DINO/iBOT
        |
        v
+ GCVD
        |
        v
+ TGSR topology mask only
        |
        v
+ TGSR relation reconstruction
        |
        v
GCVD + full TGSR
```

原因：

如果一次加入：

- topology response；
- path；
- span mask；
- relation loss；

最后性能变化时无法知道是哪一部分起作用。

---

# 27. 初始化策略

最终主模型建议：

```text
MAIRA-2 initialization
```

原因：

已有 downstream 实验显示 MAIRA-init 明显优于 random-init。

注意：

\[
\text{attention 更集中}
\neq
\text{representation 更好}
\]

random-init 模型虽然 attention 更容易集中在 spine，但 downstream representation 仍然较弱。

---

# 28. Random Init 的作用

Random init 不应作为主模型初始化。

建议作为：

### Initialization Ablation

比较：

```text
Random + baseline
Random + TGSR

MAIRA + baseline
MAIRA + TGSR
```

这样可以判断：

> TGSR 是否只对某一种 initialization 下的 attention pattern 有效。

理想结果：

\[
\Delta_{\text{TGSR}}^{Random}>0
\]

且：

\[
\Delta_{\text{TGSR}}^{MAIRA}>0
\]

说明方法不依赖特定初始化。

---

# 29. 当前最大风险

## 29.1 Attention 不是 segmentation

不能期待每张图都准确得到 spine mask。

TGSR 只需要：

> 找到足够稳定的 structure-sensitive topology cue。

---

## 29.2 Initialization dependence

Random-init 和 MAIRA-init attention 差异明显。

所以不能固定：

```text
Layer 8 + Head 8
```

必须做动态 head selection。

---

## 29.3 Training-stage drift

attention 会随着训练变化。

因此：

- warm-up；
- curriculum；
- confidence fallback；

必须存在。

---

## 29.4 Self-reinforcing bias

teacher：

```text
认为区域 A 重要
```

于是 TGSR：

```text
反复 mask A
```

student：

```text
更加学习 A
```

EMA teacher：

```text
以后更加关注 A
```

形成：

\[
attention
\rightarrow
mask
\rightarrow
learning
\rightarrow
stronger attention
\]

因此必须长期保留 random mask。

---

## 29.5 Border / marker shortcut

X-ray 中：

- 左右 marker；
- 金属；
- collimation；
- black border；

可能拥有很强 attention。

因此必须：

- valid-mask；
- border penalty；
- confidence fallback。

---

## 29.6 Curved / rotated spine

不能基于固定纵轴。

必须基于：

```text
graph / skeleton / geodesic path
```

---

# 30. 代码实现建议

推荐新增：

```text
methods/geotopo_dino/
├── masking/
│   ├── __init__.py
│   ├── block.py
│   ├── topology_response.py
│   ├── topology_path.py
│   └── topology_mixed.py
│
├── losses/
│   ├── gcvd_loss.py
│   └── tgsr_loss.py
│
├── models/
│   └── ssl_meta_arch.py
│
└── configs/
    ├── gcvd_proto.yaml
    ├── gcvd_proto_tgsr_mask.yaml
    └── gcvd_proto_tgsr_full.yaml
```

---

# 31. 对现有训练主干的要求

尽量不改 DINOv2 核心训练结构。

当前理想顺序：

```python
teacher_output = teacher_backbone(unmasked_global_crops)

teacher_patch = teacher_output["patch_tokens"]
teacher_attn = teacher_output["selected_attentions"]

final_masks, topology_state = mask_policy.select(
    candidate_masks=candidate_masks,
    teacher_anchor_tokens=teacher_patch,
    teacher_anchor_attention=teacher_attn,
    anchor_valid_mask=anchor_valid_mask,
    progress=progress,
)

student_output = student_backbone(
    global_crops,
    masks=final_masks,
)
```

关键点：

> 不要为了拿 attention 再执行一次 teacher forward。

---

# 32. mask_policy 接口建议

保持现有 mask-policy abstraction。

建议：

```python
class TopologyMixedMaskPolicy:
    def select(
        self,
        candidate_masks,
        teacher_anchor_tokens,
        teacher_anchor_attention,
        anchor_valid_mask,
        progress,
    ):
        ...
        return final_masks, topology_state
```

---

# 33. topology_state 建议保存

至少包括：

```python
topology_state = {
    "enabled": True,
    "used_topology": ...,
    "fallback": ...,
    "confidence": ...,
    "selected_layers": ...,
    "selected_heads": ...,
    "head_scores": ...,
    "path_length": ...,
    "path_response_mean": ...,
    "topology_mask_ratio": ...,
    "random_fill_ratio": ...,
}
```

relation reconstruction 第二阶段还需：

```python
"topology_coordinate": ...,
"part_ids": ...,
"masked_part_ids": ...,
"visible_part_ids": ...,
```

---

# 34. 训练前必须做离线可视化

在真正预训练 TGSR 前，先写 validation / visualization script。

至少抽取 200–500 张：

```text
AP
LAT
cervical
thoracic
lumbar
thoracolumbar
whole-spine
rotated
scoliosis
different FOV
different exposure
```

每张输出：

```text
1. original X-ray
2. candidate attention heads
3. head scores
4. fused structural response
5. thresholded component
6. topology path
7. topology band
8. selected topology span
9. final mixed mask
```

---

# 35. 必须统计的离线指标

至少记录：

```text
selected head distribution
confidence distribution
fallback ratio
mean path length
mean path response
mean topology mask ratio
random fill ratio
border-touch ratio
```

其中最重要的是：

\[
\boxed{\text{fallback ratio}}
\]

如果大量样本都 fallback：

例如：

```text
fallback > 30%-40%
```

说明 topology extractor 尚不稳定，不建议开始大规模预训练。

---

# 36. 单元测试

至少覆盖以下 synthetic cases：

### Case 1：Vertical path

```text
|
|
|
|
```

应该正确提取。

### Case 2：Rotated path

```text
////
```

不能依赖 top-bottom。

### Case 3：Curved path

```text
(
 (
  (
```

应保持连续。

### Case 4：Low confidence

全图随机噪声 response：

```text
fallback == True
```

### Case 5：Strong border response

边缘非常亮，中央有中等连续结构：

```text
border head should be penalized
```

### Case 6：Padding

padding patch 不得进入 topology path。

### Case 7：Mask count

最终 mask patch 数必须与目标 iBOT mask 数接近。

---

# 37. 推荐的第一阶段 Codex 任务

请优先实现：

## TGSR Masking MVP

只做：

```text
Teacher attention
    ↓
Head scoring
    ↓
Fused structural response
    ↓
Topology extraction
    ↓
Continuous span
    ↓
Topology/random mixed mask
    ↓
Original iBOT reconstruction
```

暂时不要：

- relation loss；
- 学习式 topology detector；
- segmentation supervision；
- 固定 vertebra count；
- CLS-patch cosine；
- 第二次 teacher forward。

---

# 38. 第一阶段实现要求

1. 保持 teacher-unmasked → mask selection → student-masked 的顺序。
2. 从 teacher 已有 forward 中导出指定中间层 CLS→patch attention。
3. attention layer 由 config 指定。
4. 不硬编码 Layer 8。
5. 不硬编码具体 head。
6. 实现 head scoring。
7. 至少使用 concentration + continuity + border penalty。
8. top-k heads 融合。
9. valid mask 去除 padding。
10. topology extraction 不依赖 vertical axis。
11. 建立 normalized arc-length topology coordinate。
12. continuous span masking。
13. mask count matching。
14. random block 补齐。
15. confidence fallback。
16. curriculum mixing。
17. topology probability 永远小于 1。
18. 输出 topology_state。
19. 加日志。
20. 加离线可视化。
21. 加 unit tests。
22. 不修改 GCVD 逻辑。
23. 不修改 downstream。
24. 不覆盖 baseline config。

---

# 39. 第二阶段 Codex 任务

在 Masking MVP 验证稳定后，再实现：

## TGSR Relation Reconstruction

流程：

```text
Topology path
    ↓
Relative part partition
    ↓
Teacher part pooling
    ↓
Student part pooling
    ↓
Teacher relation matrix
Student relation matrix
    ↓
Masked-to-visible relation loss
```

建议：

```yaml
tgsr:
  relation:
    enabled: true
    num_parts: 8
    loss_type: smooth_l1
    weight: 0.1
```

---

# 40. Relation Reconstruction 的关键实现细节

## Teacher

teacher 使用：

```text
unmasked anchor dense tokens
```

且：

```python
teacher_part_features = teacher_part_features.detach()
```

不允许 relation loss 反向传播到 teacher。

---

## Student

student 使用：

```text
masked global crop dense tokens
```

注意 part mapping 必须与 topology coordinate 保持一致。

---

## Masked / visible part 判断

若某个 relative part 中：

\[
\frac{\text{masked patches}}{\text{part patches}}
>
\tau_m
\]

则将该 part 视为 masked。

例如：

\[
\tau_m=0.5
\]

否则视为 visible。

---

## Relation Loss

推荐：

\[
\mathcal L_{\text{rel}}
=
\operatorname{SmoothL1}
(
R^S_{\mathcal M,\mathcal V},
R^T_{\mathcal M,\mathcal V}
)
\]

只计算有效 pair。

如果：

- masked part 太少；
- visible part 太少；
- topology confidence 太低；

则：

```python
L_rel = 0
```

而不是强行计算。

---

# 41. 最终方法故事线

论文里 TGSR 可以描述为：

> Standard iBOT masking treats all spatial regions uniformly and ignores the structured anatomical organization of spinal radiographs.  
> We therefore introduce Topology-Guided Span Reconstruction (TGSR), which mines structure-sensitive responses from the EMA teacher and constructs a latent topology path without vertebral annotations. Continuous spans along this path are masked, forcing the student to infer missing spinal structures from surrounding anatomical context. Beyond patch-level latent recovery, TGSR further preserves relations between masked and visible topology parts, encouraging the encoder to model both local vertebral content and long-range spinal organization.

中文：

> 标准 iBOT 随机掩码将所有空间区域等价处理，忽略了脊柱 X-ray 中连续且具有解剖组织关系的脊柱结构。为此，引入 Topology-Guided Span Reconstruction（TGSR）。TGSR 从 EMA teacher 的中间层注意力中挖掘结构敏感响应，在无需椎体标签的情况下构建相对拓扑路径，并沿该路径遮挡连续结构片段，使 student 必须利用上下游可见解剖信息恢复缺失区域。除原始 iBOT 的 patch-level latent reconstruction 外，TGSR 进一步约束 masked topology parts 与 visible topology parts 之间的关系，从而同时建模局部椎体内容和长程脊柱组织结构。

---

# 42. 与 GCVD 的区别

GCVD 和 TGSR 解决的是不同问题。

## GCVD

关注：

\[
\text{cross-view local correspondence}
\]

核心问题：

> 同一物理区域在不同 crop / augmentation view 中应保持合理的局部对应关系。

---

## TGSR

关注：

\[
\text{within-image anatomical topology}
\]

核心问题：

> 随机 masking 忽略脊柱的连续结构，应利用无监督解剖 topology 产生结构化遮挡，并要求恢复缺失结构与周围结构的关系。

因此：

```text
GCVD = view consistency / geometry-conditioned local alignment

TGSR = topology-aware masking / structural reconstruction
```

两者互补，不重复。

---

# 43. 不应做的事情

请避免以下实现：

```text
1. 使用 CLS-patch cosine 作为默认 spine probability
2. 固定 Layer 8
3. 固定 Head 8
4. 假设脊柱永远 top-to-bottom
5. 假设固定 17 个椎体
6. 将 attention 当 segmentation GT
7. 100% 使用 topology mask
8. early training 就完全相信 teacher
9. 为提取 attention 再 forward 一次 teacher
10. 一开始同时实现 mask + relation + 复杂 graph learning
11. 修改 iBOT target 定义
12. 修改 GCVD 已验证逻辑
13. 让 topology loss 反向传播进 teacher
```

---

# 44. 推荐开发顺序

```text
Step 1
现有 baseline 完全跑通

Step 2
attention extraction 接口

Step 3
offline head scoring visualization

Step 4
topology response

Step 5
path extraction

Step 6
span mask

Step 7
fallback + curriculum

Step 8
TGSR-mask-only 小规模训练

Step 9
下游验证

Step 10
relation reconstruction

Step 11
完整 TGSR

Step 12
GCVD + TGSR 联合训练
```

---

# 45. 推荐实验矩阵

## Initialization

```text
Random
MAIRA-2
```

## Method

```text
DINO/iBOT baseline
+ GCVD
+ TGSR-mask
+ TGSR-mask+relation
+ GCVD+TGSR
```

重点比较：

\[
\Delta_{\text{TGSR}}
\]

在不同初始化下是否都成立。

---

# 46. 最终评价 TGSR 是否成功

TGSR 不能只看 attention visualization。

至少需要：

### 训练稳定性

- loss 无异常；
- fallback 合理；
- topology confidence 随训练改善。

### representation

- classification；
- keypoint detection；
- segmentation / dense task。

### topology module 本身

- path 是否落在脊柱附近；
- 是否适配 AP/LAT；
- 是否适配 rotation；
- 是否适配不同 FOV；
- fallback ratio 是否可接受。

### ablation

至少：

```text
random mask only
topology mask only
mixed mask
mixed mask + relation
```

以及：

```text
fixed-head
all-head mean
adaptive-head selection
```

---

# 47. 当前推荐结论

现阶段不要直接实现完整 TGSR。

优先完成：

\[
\boxed{
\text{Adaptive Teacher Attention Mining}
\rightarrow
\text{Topology Path}
\rightarrow
\text{Continuous Span Mask}
}
\]

并保持原始 iBOT latent reconstruction。

确认 topology extractor 在：

- Random-init；
- MAIRA-init；
- AP；
- LAT；
- 不同 FOV；
- 不同旋转；

下都基本稳定之后，再加入：

\[
\boxed{
\text{Masked-to-Visible Topology Relation Reconstruction}
}
\]

这是目前最稳妥、最容易做消融、也最容易解释的方法路线。
