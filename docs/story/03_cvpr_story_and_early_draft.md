# CVPR 故事线、贡献边界与可提前撰写内容

> 这是“计划版论文语言”。没有实验支持前，不使用 demonstrate、outperform、solve 等结果性动词。

## 1. 推荐标题

首选：

> **When Invariance Erases Anatomy: Geometry- and Topology-Aware Self-Distillation for Spine Radiographs**

该版本保留你喜欢的核心标题，同时让副标题覆盖第二个 topology/masking 创新。

更稳妥的备选：

> **Learning Ordered Anatomy from Partial Views with Geometry-Conditioned Distillation and Topology-Guided Masking**

工作名：**GeoTopo-DINO**。

---

## 2. 一句话论文论点

### English

> In spine radiographs, we investigate the mismatch between unconditional crop invariance and ordered anatomy under partial observation, and introduce a self-distillation framework that aligns observed local regions with their geometric counterparts while reconstructing missing anatomical spans from their topological context.

### 中文含义

本文不是单独讲“local-global 对齐”或“脊柱 mask”，而是研究同一个问题的两面：可见的局部不能失去身份，缺失的局部不能只靠纹理插值恢复。

---

## 3. 故事应怎样展开

### 第一层：通用 SSL 的成功与隐含假设

DINO-style self-distillation 通过让不同视图预测一致表征学习强大的全局语义；iBOT 进一步通过 masked patch prediction 增强 dense features。两者隐含地把同图 crops 当作统一语义来源，并使用通用空间 mask。

### 第二层：脊柱影像暴露出的矛盾

脊柱不是一个无序目标，而是具有 superior–inferior 顺序、连续拓扑和重复局部外观的解剖链。在 partial-FOV X-ray 中：

- 两个 local crops 可能来自完全不同的椎体区间；
- 相似外观可能对应不同解剖位置；
- 连续一段缺失时，正确推断依赖上下游结构，而非附近像素。

因此，统一 local→global invariance 可能造成 anatomical identity ambiguity；random/block masking 也没有显式迫使模型利用 spinal topology。

### 第三层：一个统一解决方案

GeoTopo-DINO 将问题分成 observed 与 missing 两种 partial observation：

1. **GCVD** 使用已知 crop geometry，将 local dense features 对齐到 global teacher 的对应区域，并学习 local crops 之间的相对顺序；
2. **TGSR** 在线发现 teacher 的有序脊柱结构，遮挡连续 topology span，同时恢复该 span 的 latent content 及其与可见 parts 的关系。

### 第四层：最终希望证明的结论

若实验成立，论文最终可以声称：

> 将所有裁剪统一处理为不变视图并不是结构化影像的最佳自监督原则；对可见区域使用几何条件化对应，对不可见区域使用拓扑条件化预测，能够在保持全局语义的同时改善局部解剖区分与定位能力。

---

## 4. 为什么两个点能组成一篇论文

一个创新点并非天然“不够 CVPR”，两个模块也不天然更强。审稿人真正关心的是它们是否服务于同一问题，并分别有不可替代的证据。

这里的统一关系是：

| 部分观察状态 | 标准 DINO/iBOT 的处理 | GeoTopo-DINO |
|---|---|---|
| 区域可见，但只出现在 local crop | 对齐 whole-image semantics | 对齐对应 global region，并建模 local order |
| 连续区域被 mask，不可见 | 独立预测 masked patches | 利用上下游 parts 恢复 span content 和 topology relation |

因此第二个点不能只写成“我们设计了一个 spine mask”。它必须写成 **topology-guided span reconstruction**，并证明 relation target 在只更换 mask 之外仍有增益。

---

## 5. Introduction 四段草稿

### Paragraph 1：任务与价值

Self-supervised vision models have substantially improved representation learning by enforcing consistency across augmented views. This principle is particularly attractive for radiographs, where large-scale annotations are scarce and downstream tasks range from image-level recognition to fine-grained anatomical localization. However, the invariances that benefit global recognition do not necessarily preserve the spatial identity required by dense anatomical tasks.

### Paragraph 2：具体矛盾

This mismatch is pronounced in spine radiographs. The spine forms a repetitive yet ordered anatomical chain, and radiographs frequently capture only a variable field of view. Local crops from different vertebral intervals may therefore share similar appearance while carrying different anatomical identities. Nevertheless, DINO-style multi-crop distillation encourages all local views to predict a common global target. Meanwhile, generic masked patch prediction removes spatial blocks without explicitly requiring the model to reason over the superior–inferior topology of the spine.

### Paragraph 3：方法

We investigate geometry- and topology-conditioned self-distillation for learning from such partial observations. Our framework, GeoTopo-DINO, first aligns each local representation with its geometrically corresponding region in a full-field teacher feature map and models the ordered relations among local views. It then discovers an ordered spinal topology from momentum-teacher features, masks a connected anatomical span, and reconstructs both its latent content and its relations to visible spinal parts. The two components respectively preserve the identity of observed regions and infer the organization of missing regions.

### Paragraph 4：贡献占位版

Our study is designed to test whether this formulation provides a better balance between global semantic robustness and local anatomical sensitivity than unconditional crop consistency and unstructured masking. We evaluate this question through representation diagnostics, dense correspondence analysis, variable-field-of-view stress tests, and downstream classification and localization tasks. **[Evidence needed: insert the strongest quantitative result and external/generalization result after experiments.]**

---

## 6. 贡献点草稿

实验前建议写成“我们提出/研究”，不要写成“我们首次证明”。

1. We formulate spine self-supervised learning as representation learning from partial observations of an ordered anatomy, distinguishing appearance invariance from geometry- and topology-sensitive information.
2. We introduce Geometry-Conditioned View Distillation, which replaces uniform local-to-global matching with dense region correspondence and ordered local relations derived without anatomical labels.
3. We introduce Topology-Guided Span Reconstruction, which masks a connected sub-chain discovered from teacher features and reconstructs both masked latent content and its relations to visible anatomical parts.
4. We design evaluations that jointly measure global semantics, local anatomical discriminability, geometric correspondence, topology awareness and robustness to variable fields of view. **[Results pending.]**

---

## 7. Figure 1 应该表达什么

Figure 1 不要只画网络结构。它应先展示 failure，再展示原则：

### 左侧：标准 DINO/iBOT

- full spine global；
- 胸椎 local 与腰椎 local；
- 两者都指向同一个 global target；
- random blocks 被独立预测；
- 标注：`Uniform crop invariance` 与 `Unstructured missing regions`。

### 右侧：GeoTopo-DINO

- 每个 local 指向 global 中对应 ROI；
- local pair 标出 superior/inferior/overlap；
- 沿中心线遮挡连续 span；
- masked span 与上下游 visible parts 建立 relation reconstruction；
- 标注：`Preserve observed identity` 与 `Infer missing topology`。

图下方只放一句：

> **Not every crop should share the same target, and not every missing region should be reconstructed independently.**

---

## 8. 审稿人会接受这条故事所需的证据

### Claim A：存在 anatomical over-invariance

必须比较标准 DINO 的不同 checkpoint，展示：

- global 指标提升时，far-apart vertebral regions 是否越来越相似；
- same-level consistency 与 different-level discriminability 是否出现冲突；
- local→whole-global loss 是否比 global-only + region loss 更容易产生该问题。

若没有这一现象，不能把 “erases anatomy” 写成已证实事实。

### Claim B：GCVD 不是普通 ROI matching

至少比较：

- 标准 DINO local→global CLS；
- 对应区域 mean pooling；
- dense geometry warp；
- dense warp + local order；
- shuffled correspondence。

关键结果应出现在 vertebra-level retrieval、keypoint localization、segmentation/detection 和 variable-FOV 测试，而不只是 pretext loss。

### Claim C：TGSR 不是换一种 mask

必须保持 mask ratio、训练时长和算力一致，比较：

- random mask；
- DINOv2 block mask；
- attention mask；
- vertical contiguous span；
- teacher-derived curved span；
- curved span + topology relation reconstruction。

最后两行的差异才证明“重建拓扑关系”有独立作用。

### Claim D：两个组件互补

核心消融：

| Variant | GCVD | Ordered Local | Topology Span | Topology Relation |
|---|---:|---:|---:|---:|
| DINOv2/iBOT |  |  |  |  |
| +GCVD | ✓ |  |  |  |
| +GCVD+OLRL | ✓ | ✓ |  |  |
| +TGSR |  |  | ✓ | ✓ |
| Full | ✓ | ✓ | ✓ | ✓ |

Full model 必须体现稳定互补；如果某一列无独立贡献，应删掉而不是为了“两个创新点”强行保留。

---

## 9. 与已有工作的边界

论文不能把以下内容单独声称为创新：

- 已知 crop 坐标下的 dense matching；
- attention-guided mask；
- anatomy-aware mask；
- 将连续区域换成 block mask；
- teacher latent reconstruction；
- 单独预测 crop 相对位置。

更有防御力的边界是：

> 本文将视图关系和遮挡关系统一为对 partial ordered anatomy 的条件化学习：可见 local 由真实几何对应约束，缺失 span 由有序上下文关系约束。

直接相关的阅读边界包括 DINO/DINOv2/iBOT、VICRegL、EsViT、DenseCL、AttMask、HAP、Evolved Part Masking、AnatoMask、SCE-MAE 和医学 anatomical embedding。正式 Related Work 需要逐篇核实后再落引用。

---

## 10. 当前可以提前写、不能提前写的内容

### 现在可以写

- Introduction 前三段；
- Related Work 的主题结构；
- Problem formulation；
- GCVD/TGSR 的数学定义；
- 数据集、预训练设置和评估协议；
- Figure 1/2 草图；
- claim–evidence 表与实验矩阵。

### 现在不能定稿

- “DINO 确实擦除了 anatomy”的结论；
- abstract 中的性能数字；
- outperform/SOTA/generalize 等结果性措辞；
- 最终模块数量和 loss 权重；
- teacher topology 的可靠性结论。

---

## 11. Claim–Evidence Map

| Claim | Evidence | 当前状态 |
|---|---|---|
| 统一 crop invariance 可能弱化局部解剖身份 | checkpoint similarity、retrieval、FOV 双指标 | 待实验 |
| GCVD 恢复 local anatomical identity | dense correspondence、椎体探针、keypoint | 待实验 |
| local order 提供超越 ROI matching 的信息 | shuffled-order、direction/rank、跨患者 retrieval | 待实验 |
| topology span 促进跨区域推断 | 同 mask ratio 的 masking 对比 | 待实验 |
| topology relation target 超越 mask policy | span-only vs span+relation | 待实验 |
| 两个模块兼顾 global 与 local | classification 与 dense tasks 的联合结果/Pareto | 待实验 |

---

## 12. 当前建议与导师确认的五个决定

1. 是否接受 “partial observations of ordered anatomy” 作为整篇论文的中心问题；
2. 是否将 standard local→whole-global DINO loss 从主模型中移除；
3. 是否使用 full-FOV letterbox anchor 作为一个 global view；
4. 第二贡献是否从“spine mask”升级为“continuous span + topology relation reconstruction”；
5. 是否先做两周 failure diagnosis，再决定标题中的 “Erases Anatomy” 能否保留。

