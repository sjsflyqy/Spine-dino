# TGSR：基于 attention 的脊柱连续 span 掩码与代码实施方案

日期：2026-09-17。分析对象：本地 `random_init_100`、`random_init_100eps_8layers_heads` 以及本次下载的 Spine-dino main 分支快照（commit `815ad732828aa41ff1f33a991856489afa5c958a`，由 GitHub archive comment 核验）。本文是实施设计，未修改训练代码、未运行新模型训练。

## 1. 结论与证据边界

建议保留 TGSR 的主干：未遮挡 teacher → 粗脊柱路径及带状候选区域 → 删除连续有序片段 → student 恢复 masked patch 的 teacher 目标和 masked-to-visible 的关系。

需要调整的是：attention 的融合与验证、路径端点、mask 预算、可见上下游保护、relation 的 part 定义，以及训练阶段的回退机制。

attention 应当作为粗定位信号，不应称为准确的脊柱分割概率。构造的是 patch 级训练遮挡区域，而非用于临床测量的像素级分割标签。路径及 band 都是候选估计。

本地仅有 PNG 可视化，没有原始 attention 数值、模型权重和配套 metadata。因此本次能给出定性分析及实现规划，不能计算 Dice、中心线误差或可靠的自动置信度。颜色叠加图不能可靠反演 attention。

## 2. 六张第 8 层 head 图的定性观察

以下 head 编号与图中一致，采用 1-based 编号；代码转换为 0-based。

| 图像 | 观察 | 掩码设计含义 |
|---|---|---|
| buu2000_ap_00000002 | Head 5、8 在脊柱位置有较长纵向响应；Head 4 较局部；Head 9、11、12 明显关注图像侧边 | 固定平均全部 head 会引入边缘；较长响应与局部响应可以互补 |
| csxa_00000629 | Head 2、4、7 较集中于下部颈椎，Head 5、8 有更长但夹杂其他结构的响应 | 单一强 head 容易只遮最显著的一小段，融合必须兼顾长度和污染 |
| nanning_00000706 | Head 4、7 聚焦颈椎局部；Head 2、5、8 的响应分布不同；Head 1、11、12 主要有外侧响应 | 高响应并不等于完整脊柱覆盖；不宜用固定阈值直接分割 |
| nhanes2_00009665 | Head 8 接近较长连续链；Head 4、5 主要局部；Head 9、11、12 有边缘污染 | 连续链可用于路径提取，但仍需要 band 而非细线 mask |
| ningbo_00000356 | Head 9 较明显关注上部椎体，Head 2、5 更偏下部；Head 4、7 位于中部；Head 8 有较长后侧响应并混入轮廓 | head 的作用依赖视图，不能把 Head 9 永久标记为坏 head，也不能把 Head 8 永久当作椎体中心 |
| vindr_train_00006336 | Head 4 聚焦椎体中段，Head 5、8 呈更长纵向响应；Head 1、9、11、12 混有外侧/边缘 | 支持比较筛选融合与全部 head 平均 |

第 8 层是合理的首选候选，但这些样本无法证明它跨数据源、训练阶段、输入分辨率均为最佳层。建议先固定第 8 层建立可复现首版，再比较第 7/8/9 层；不要一开始就在线搜索所有层和所有 head。

可先比较三个确定性 response baseline：全部 head 平均、Head 4/5/8 的等权融合、Head 2/4/5/7/8 的等权融合。后两组是当前 checkpoint 的待验证候选，不是已验证最佳组合。只有小型验证集支持后，才升级到逐图动态筛选。侧位宁波样本应额外检查 Head 9 的补充作用。

## 3. 先验证训练输入尺度下的定位

仓库 `visualization/dino_spine_maps/visualize.py` 默认 long-side=896，preprocessing 仅将尺寸补到 patch size 的倍数，通常得到矩形 token grid。

GeoTopo anchor 则等比例缩放到 518×518 的方形画布，产生 37×37 token grid；全脊柱的有效宽度可能只有其中一小部分。两种 token 密度不同，不能把高分辨率可视化上的 band 宽度和阈值直接搬进训练。本地图的实际 long-side 无 metadata 证实，896 是代码默认值而非已确认的出图配置。

第一步应在训练使用的 full-FOV anchor 预处理下重新导出：原图、model input、valid mask、每 head 的 patch-grid attention、融合 response、中心线、band、最终 span/block masks。可用仓库现成 raw_maps.npz 导出逻辑，但应新增完全匹配 anchor 的预处理模式，避免重复造数据读取逻辑。

挑选约 100–200 张用于定位诊断的图像，覆盖六个数据源、AP/侧位、局部/全脊柱及弯曲明显病例。若允许，人工画粗中心线或粗 ROI 只用于验证，不进入预训练损失；用于选 head 和阈值的子集应与最终质量评估子集分开。若完全不使用任何人工标注，应明确只验证结构一致性，不能据此宣称解剖定位准确率。

检查中心线误差、band 的脊柱覆盖与背景污染、上下端点误差、失败类型、按来源/视图的回退率；若评价 Dice，需另有真正可比较的分割标注。质控不能只展示最成功的热图。

## 4. Attention response：选择和融合

第 8 个 block 的 CLS attention 来自该 block 的 normalized 输入，即第 7 个 block 输出再经 norm1，而非第 8 层输出 token 与 CLS 的 cosine。

对 head h：

```text
a_h = softmax(q_cls,h @ k_all,h.T / sqrt(d_head))
patch_start = 1 + num_register_tokens
a_patch,h = a_h[patch_start:]
valid_patch_mass,h = sum(a_patch,h * valid)
p_h = a_patch,h * valid / (valid_patch_mass,h + eps)
```

softmax 的分母包含 CLS/register/patch keys，随后再切 patch 并对 valid patch 重新归一化。保留归一化前的 mass 作为诊断，不要让一个几乎不关注有效 patch 的 head 被归一化后误判为优秀。

图中的 patch mass 约 1 仅表示 attention 大多分配到 patch keys；它不表示 attention 落在脊柱上。这六张图几乎所有 head 的 mass 都高，因而单用 mass 无法筛出脊柱 head。

首版用验证通过的固定 head 集，做 valid-aware 等权平均。随后对融合 response 做归一化卷积：smooth(A*valid)/smooth(valid)，并重新排除无效区域；普通零填充平均会压低 band 边界。

动态版本可在候选 head 中综合以下指标：局部纵向覆盖、有限宽 band 相对同排外围的响应增益、孤立热点比例、图像侧边污染、跨弱光度增强的路径一致性。不建议只用低熵、最高峰或平滑度评分：集中 hotspot 可能只有一节椎体，身体边缘也可能很平滑。

互补 head 不能用逐点几何平均强制交集，否则一个关注胸椎、一个关注腰椎时会互相抑制。建议加权算术平均、权重上限、候选 group 比较，并验证融合后的覆盖。强 head 也不能无限占据权重。

## 5. 路径和 spine band

不要求从图像第一行一直走到最后一行。颈椎图上方是颅骨，侧位胸腰椎图也有不属于脊柱的边界；强制整高贯通会将无证据区域编进拓扑链。

首版先尝试受支持的上下端点/纵向区间，再在每个候选区间 [y0,y1] 内求路径：

```text
J(c) = sum_y U(y,c_y)
       - lambda_step * sum_y |c_y-c_(y-1)|
       - lambda_curve * sum_y |c_y-2*c_(y-1)+c_(y-2)|
subject to: valid(y,c_y), |c_y-c_(y-1)| <= delta
```

U 建议使用候选中心附近一个有限宽窗口的 response 累积，再相对同排外围响应归一化，而不是只看单点峰值。椎体双侧缘可能比椎体中心更亮，尤其侧位，单点 DP 易走到皮肤轮廓或椎体后缘。

简单首版用一阶步长惩罚即可；加入 curvature 二阶项时 DP 状态需含前一步位移，不能继续声称普通一阶 DP 精确求解该目标。所有平滑后的路径都要重新满足 valid 与步长约束，不能平滑到 padding。

端点选择必须设最小支持长度，避免算法只取最高亮的两节椎体。AP 和侧位分别诊断；不要用硬性“图像正中央”约束，侧位或偏心拍摄的脊柱常偏离中心。轻微位置先验只能作为可消融的候选。

以路径为中心形成 band。首版在 37×37 网格测试半宽 1、2、3 个 patch，总宽 3、5、7；这些是搜索起点，必须结合全脊柱/局部图的实际比例验证。粗定位允许宽一些，但不能随目标 mask ratio 无限扩张。若估计局部椎体宽度，应平滑并限定上下界。

沿有效路径累计弧长，分为有序 P1…PK，初版 K=6；每个 band patch 分配到一个有序 part。短局部图按有效 token 数降低 K 或跳过 relation，而不是产生大量空 part。这里上下顺序依赖输入已经标准化为上方对应头侧；若存在旋转/倒置，先规范方向，否则只能称图像轴上的相对顺序。

## 6. 连续 span：分别定义长度和全图预算

对 anchor 保留原来是否 mask 的采样及目标 token 数，第一版避免同时增加监督数量。仓库现有 mask_sample_probability=0.5，mask_ratio_min_max=[0.1,0.5] 是配置值，真实训练若覆盖参数应以日志/运行配置为准。

设 valid region V、band B，目标数量 m=floor(r_global*|V|)。选连续 1–2 个完整内部 parts 形成 S，K=6 时大致覆盖可见链的 17%–33%；不是全图比例，也不保证精确等于 band 面积比例。

```text
M = S union M_background
M_background subset of V \ B
|M| = m  （若可行）
```

选 span 时保留其上、下至少各一个完整可见 part。补充 mask 不触碰 band，保证可见上下游干净；不采用按 attention 阈值挖孔的 span。选中纵向段的整个候选 band 应一起 mask，并可测试有限周边 margin，避免椎体边缘仍露出使任务变成近邻抄写。margin 对 relation 所用的可见 parts 也应留出距离。

必须先检查可行性：|S|≤m 且 m-|S|≤|V\B|。若 span 超预算，优先换更短的完整 span；若低目标比例无法容纳最短 span，回退原 block。不要随机删 span token 来凑数，那会破坏连续片段删除。如果 band 外容量不足，采用预定义回退，不让背景补充块破坏可见 parts。

background block 建议在 V\B 内独立生成并补足数量。现成 candidate block 如与 band 相交，不能原样保留；可保留其 band 外部分再补足，也可以重新采样。第一版最终每行 token 数尽量与 candidate mask 完全一致，从而保留现有 upperbound 和 iBOT 监督预算。

举例：有效区域 500 个 token，band 100 个，遮其中 30 个时，span 仅占全图 6%。若目标总比例 30%，仍需补 120 个 band 外 token。这说明不应把 span 长度 30% 当成全图 mask ratio 30%。

## 7. 置信度和课程

建议先做硬性失败判定，再做连续可靠性权重。失败包括：支持长度不足、band 容量异常、无完整上下游可见 part、valid/padding 越界、预算不可行。连续指标包含 band 对背景的对比、纵向支持覆盖、合理跳变、头之间/弱增强之间的几何一致性。

平滑路径可以始终被 DP 算出来，因此“DP 有结果”不是置信度；多个 head 同意皮肤边缘也不保证正确。人工小型验证集用于校准阈值，粗定位失败时回退原 block，relation 对该样本置零。

```text
p_topo(t) = p_max * clamp((t-t0)/(t1-t0), 0, 1)
```

以全程 optimizer iteration/实际 data passes 表达 t，断点续训不重置。配置中的 pseudo-epoch 与数据真实 passes 不一定相同。

从随机初始化训练时，可先用 10%–20% 总步数保持 block，再用下一段 10%–20% 步数升到 p_max=0.5，且需定位质控达标；这些是初始搜索值。若继续当前 100-pass teacher checkpoint，可以缩短纯 block 阶段，但仍应在新 anchor 分辨率与增强下验证，而非按 checkpoint 名字直接认定可靠。

relation 的权重课程应独立，待 topology mask 稳定后再升温，不把两个新因素一起打开。弱增强一致性评估可离线/低频进行；初版不额外增加每 iteration 的 teacher forward。

## 8. 目标一：保留 iBOT

masked patches 使用 teacher 未遮挡同坐标 patch 经原 DINO/iBOT head 得到的 centered/sharpened prototype probability，student 同坐标预测做交叉熵。这是 latent/prototype reconstruction，不能称像素或真实椎体形态重建。

student 只在 patch embedding 后替换输入 token。relation pooling 用 student 经过完整 transformer 得到的输出 patch tokens，不能使用尚未恢复的输入 mask_token。

原 DINO/iBOT/KoLeo 权重先保持不变；TGSR 的 patch 项就是已有 iBOT 改 mask 后的项，不另外再重复加一份相同 iBOT loss。

## 9. 目标二：masked-to-visible part relation

首版直接用最终层 normalized patch tokens，无新可学习 head，降低引入因素及 FSDP/EMA 改动。

```text
z_k^t = normalize(mean_{p in P_k}(teacher_patch_p))
z_k^s = normalize(mean_{p in P_k}(student_output_patch_p))
R_kj^t = dot(z_k^t, z_j^t)
R_kj^s = dot(z_k^s, z_j^s)
```

teacher 和 student 使用同一份 detached part membership 及相同 pooling 支持。首版采用均匀 pooling，不用 attention 权重只池化最亮的几个 token，否则同一个 part 的 latent content 会被热点定义。

只计算完全 masked parts I_M 到完整 visible parts I_V 的 Smooth-L1；先用全部有效 visible parts，额外报告紧邻上下游项。若单列邻近 loss，再在消融中证明它有价值。排除空 part、token 太少 part、预算回退、无有效 pair 样本。

```text
L_rel = mean_over_eligible_images(
          c_i * mean_{k in I_M, j in I_V} Huber(R_kj^s-R_kj^t)
        )
L_total = L_DINO + lambda_ibot*L_iBOT + L_KoLeo
          + lambda_GCVD*L_GCVD + lambda_rel(t)*L_rel
```

c_i 是 detached 连续置信度；按有效样本数做分母时记录实际 c 均值，避免不同归一化方式掩盖可靠性权重作用。分布式训练应明确是局部样本均值还是跨 rank 全局有效样本均值；建议全局有效样本计数，处理不同 rank 上可用图像数不同的情况，确保梯度缩放匹配 DDP/FSDP 的 rank 平均。

若某 rank 没有有效 relation 样本，返回与 student tokens 保持计算图关联的零值，且所有 rank 仍按一致顺序参与必要 collective。无 mask rank/全局无 mask 的 iBOT center 和 Sinkhorn 也要单独处理，不能除零或一部分 rank 跳 collective。

lambda_rel 首轮可试 0.05–0.1，但必须比较实际 weighted loss 和梯度大小，数值不是可跨损失直接照搬的标准。

语义限制：cosine relation matrix 保留 teacher 的部件间特征相似关系；它本身不会保证解剖顺序、相邻关系或距离。预定义有序 part 与连续 span 负责引入路径结构。不要由这一个损失就声称 student 学会椎体编号或完整解剖拓扑。若 teacher 所有 part 的 cosine 都接近 1，则 relation 目标近似常数，应先记录 off-diagonal 方差/分布，暂停增加权重。若要声称顺序恢复，应另做同图的顺序/距离任务和验证，避免把绝对坐标直接泄露给预测器。

## 10. 对应仓库代码的实施点

本次快照主要文件：

| 位置 | 现状 | 建议 |
|---|---|---|
| methods/geotopo_dino/models/ssl_meta_arch.py:62 | tgsr.enabled=true 抛 NotImplementedError | 增加实际 TGSR 初始化及配置校验 |
| 同文件:106 | anchor/random policy 限于 block/block | 允许 anchor=topology_mixed，random_global=block |
| 同文件:169–184 | teacher 输出后调用 policy.select 并 pack_masks | 复用既有 teacher-first 流程，新增 attention 参数及 topology state |
| methods/geotopo_dino/masking/base.py:11 | policy 仅接 teacher_anchor_tokens、valid、progress | 新增 teacher_anchor_attention 和 target_counts；保留 tokens 给 pooling |
| methods/geotopo_dino/data/collate.py:150–164 | 已给 candidate mask、mask_target_counts、upperbound | 保留采样/回退；向 policy 显式传 target counts，不在 collate 推模型 |
| methods/geotopo_dino/masking/packing.py:28 | 将 final mask 转 indices、weights、masked count | final mask 确定后统一调用，禁止使用旧 indices/weights |
| visualization/dino_spine_maps/map_extractor.py:45 | 有 CLS-only attention 数学逻辑，但 batch=1 并转 CPU/NumPy | 抽取训练可用 batched GPU tensor 版本，保持定义一致 |
| upstream/dinov2-main/dinov2/layers/attention.py | MemEffAttention 默认不返回 attention | teacher-only collector 或显式可选输出；维持 xFormers 主路径 |
| methods/geotopo_dino/configs/gcvd_mvp.yaml | TGSR disabled | 新建 tgsr_mvp.yaml，保留原 MVP 配置作为对照 |

推荐新增：

```text
methods/geotopo_dino/
  models/teacher_attention.py
  masking/attention_response.py
  masking/spine_path.py
  masking/topology_span.py
  losses/topology_relation_loss.py
  tools/visualize_topology_masks.py
  configs/tgsr_mvp.yaml
```

### Teacher attention 提取的最小侵入方案

首版可在 teacher 的第 8 个 block 的 attn.qkv 上注册局部 forward hook，QKV 输出本来已经计算，不再额外调用 qkv linear。hook 中重排 QKV，取 q_cls 与 k_all，FP32 计算 CLS row attention；仅保留前 B 个 anchor 的 [B,H,N] 结果，保持 GPU tensor。

teacher 当前用单个 tensor global_crops forward，通常 QKV 是 [2B,T,3D]，适合该方案；student 的 nested tensor 有不同打包形式，因此 collector 仅绑定 teacher 并断言输入布局。block_chunks=0 的本配置对应 blocks[7]；若开启 chunked blocks，需找到实际第 8 个 block，不能仍索引 blocks[7]。

collector 每次前向清空，验证触发次数、batch/grid、patch_start；完成 policy.select 后释放 QKV 临时引用，只保留需要的 attention/topology。hook 必须在 FSDP 模块正常 forward 内执行，不能 teacher reshard 后直接读取 sharded qkv 参数重算。所有 checkpoint/buffer 输出维持原格式；collector 状态是运行时临时数据。

这仍有 CLS-row FP32 matmul/softmax 和临时 K 类型转换成本，但无需构造 [B,H,T,T] 的完整 attention，也无需第二遍 backbone。训练版应该与可视化版同输入输出逐 head 数值对齐，并在单 GPU 与 FSDP 下检验 hook 的执行和参数状态。

更长期可在 attention 模块中做显式 opt-in CLS row 返回，代码更直观，但涉及 upstream source 的更改。已有 backbone 输出签名与 checkpoint key 不应因该功能变化。

### Policy 与 topology state

```python
# 设计接口，非已运行实现
final_masks, topology = policy.select(
    candidate_masks,                  # [2B,N], True=mask
    teacher_anchor_attention=attn,    # [B,H,N], detached GPU tensor
    teacher_anchor_tokens=t_patch,   # [B,N,D], final teacher tokens
    anchor_valid_mask=valid,          # [B,Ha,Wa]
    target_counts=target_counts,     # [2B]
    progress=global_progress,
)
```

topology 包含 part_ids（[B,N]，非 band 处 -1）、part_valid（[B,K]）、masked_parts、visible_parts、confidence、used_topology、fallback_reason、路径及诊断值。优先 part_ids 而非无限增长的 dense [B,K,N] masks；K 很小时二者均可，但数据语义需一致。

只修改 final_masks[:B] 中已选中且预算可行的 anchor 行；final_masks[B:] 维持标准 block。第 0…B-1 行都是 anchor，第 B…2B-1 行都是 random global，是 view-major 排列，不能写为奇偶交替。

建议首版每行 masked count 与 candidate 一致，从而保留现有 upperbound；pack_masks 自带 upperbound 检查。若未来增加 token 数，必须同时更新 teacher/student 分配 buffer 用的 upperbound，不能只更改 mask tensor。

### 前向伪代码

```python
with torch.no_grad():
    collector.clear()
    t = teacher.backbone(global_crops, is_training=True)  # 一遍，无 mask
    attn = collector.pop_anchor(batch_size)
    masks, topo = policy.select(...)                    # teacher-conditioned
    mask_state = pack_masks(masks, upperbound=upperbound, topology=topo)
    teacher_targets = existing_teacher_targets(t, mask_state)
    t_parts = pool_parts(t['x_norm_patchtokens'][:B], topo)  # 在 reshard 前

reshard_fsdp_model(teacher)
s_global, s_local = student.backbone(
    [global_crops, local_crops],
    masks=[mask_state.masks, None],
    is_training=True,
)
loss_existing = existing_losses(s_global, s_local, teacher_targets, mask_state)
s_parts = pool_parts(s_global['x_norm_patchtokens'][:B], topo)
loss_rel = masked_visible_relation_loss(s_parts, t_parts, topo)
loss = loss_existing + scheduled_relation_weight * loss_rel
backprop_loss(loss)
```

首版不新增 relation_head。若以后引入 part_head，需要 student 与 EMA teacher 配对注册，并配置 compute_precision，检查 optimizer/checkpoint/EMA。若只有 student 辅助 predictor，应使用已有 student_aux 接口并审计分布式包装与同步，不能只放入 self.student 导致 update_teacher 按相同键查 teacher 时出错。

## 11. 质量检查与训练日志

实现检查应针对真实失败条件：CLS row 与参考提取器一致；head 编号/CLS/register 切分正确；禁止 padding mask；span 连续完整且上下游可见；预算保持；fallback 等于原 mask；indices/weights/upperbound 一致；teacher 无梯度；masked 输出有有效 student 梯度；空 part/零 pair/零 masked patch 可运行；view-major 不混图；断点续训课程不重置；两 GPU collective 不死锁。

训练日志至少记录：申请 topology 的比例、实际使用比例、eligible anchor 比例、分原因 fallback、path 支持长度/跳变、band 占 valid 比例、span 弧长与 token 数、span 占总 mask 比例、背景补充比例、实际 mask ratio、有效 part/pair 数、teacher relation 方差、relation raw/weighted loss、课程权重、时间/显存增量。指标 denominators 应在字段文档中定义，例如 topology 使用率以所有 anchor 为分母还是以 eligible anchor 为分母。

## 12. 建议实验顺序及归因

1. 冻结当前 teacher，离线检查真实训练 anchor 下的 response/path/band；比较固定 head 组，不训练 relation。
2. 用同样 full-FOV anchor 和原有损失跑 block 对照，与 topology span 版本比较；保持初始化、总更新步数、视图、增强、mask token 数一致。
3. 比较相同 ROI/token 预算内的随机位置 mask 与连续 span mask，区分收益来自定位还是连续删除。
4. 固定已通过的 topology mask，单独加入 relation，再比较 masked-to-visible 限制与全部 relation。
5. 最后比较固定 head 与动态融合、固定 band 与自适应宽度、课程与回退开关。完整组合与 GCVD 叠加另做归因。

如果两种算法改变了可处理样本比例，不能只报平均 loss；需报告实际监督预算和使用率。下游优先看与目标相关的 keypoint、椎体编号等指标，并用重复 seed/不确定性评估，预训练 loss 更低不等于拓扑表示更好。

## 13. 科学措辞建议

可表述为：利用 EMA teacher 的中间层 attention 和连续性约束估计候选脊柱带，删除其中一个连续的有序片段，通过既有 masked-token 蒸馏及部件关系蒸馏促进上下游条件下的 latent reconstruction。

当前证据不支持：无标签准确椎体分割、真正 C/T/L 节段识别、自动恢复完整解剖拓扑、特征相似度严格对应解剖距离。创新贡献需要通过定位、连续性和 relation 三组独立对照建立。

仓库链接：https://github.com/sjsflyqy/Spine-dino
