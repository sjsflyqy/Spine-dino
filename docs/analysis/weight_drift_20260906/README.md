# 权重变化量审计

比较对象：已抽取的DINOv2、RAD-DINO、MAIRA-2和DINOv3 teacher backbone，与各自官方原始权重；医疗权重先按仓库转换规则将HF分离Q/K/V合并到native格式，再逐张量核验与训练初始化一致。random-init系列没有已保存的对应初始随机状态，本次不与官方权重混作“预训练变化量”比较。未纳入来源不清楚的spine-dino-v1历史目录。

统计仅包含backbone参数，包含CLS/position/mask/register等参数；不包含SSL投影头、下游探针、LoRA。排除DINOv3 RoPE periods及attention bias_mask buffer。所有计算逐张量转float64后累计，避免把不同层的相对变化率直接平均。

指标解释：

- 整体相对L2变化：`100 * ||teacher - original||₂ / ||original||₂`。把所有匹配参数视作一个长向量。它不是变化元素比例，不是性能下降比例，也不是遗忘比例。
- 参数余弦：同一个长向量的cosine。接近1说明整体方向接近，不能保证输出特征或任务表现接近。
- 新/旧范数：`||teacher||₂ / ||original||₂`，可辅助区分整体尺度变化与方向变化。
- 精确变化元素比例：数值不完全相同的标量比例。长期浮点更新通常会让它接近100%，单独诊断价值有限。
- 逐层相对L2：分母是该层的初始范数。初始接近零的bias/LayerScale会出现很大的比例；应同时看绝对差值和占全模型平方差的份额。分母为零时留空。
- 平方差贡献：该层`||Δθ_layer||² / ||Δθ_all||²`，按层求和为100%。它衡量参数空间贡献，不能直接视为功能重要性。

`results.md`为全部checkpoint汇总，`summary.csv`为完整整体指标，`layers.csv`为block/embedding/norm等组别指标，`tensors.csv`为每个张量的指标，`initialization_checks.json`为原始权重到初始化的核对记录，`run.log`记录运行信息。

复现命令（CPU，读取权重，不修改模型）：

```bash
/home/think/mnt/zpj2025/miniconda3/envs/spine_dino_downstream/bin/python docs/analysis/weight_drift_20260906/compare_weights.py
```

权重距离只能确认“改了多少、哪里改变”。判断变化是否损害解剖信息，还需要在相同影像上比较feature、充分收敛的探针和公平的下游评价。不同初始化的参数尺度与网络参数化不同，不宜按相对L2大小直接排性能好坏。
