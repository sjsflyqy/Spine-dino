# 脊柱连续掩码：离线检查工具

本工具只运行冻结 backbone 的推理并保存可视化，不启动训练、不更新权重，
不修改或接入原有训练入口、mask policy、loss、默认配置。
生成逻辑位于 `methods/geotopo_dino/masking/spine_span.py`，供以后训练接入时复用；
目前没有任何现有训练模块导入该文件。

## 运行

在仓库根目录，使用现有预训练环境：

```bash
conda activate /home/think/mnt/zpj2025/miniconda3/envs/spine_dino_pretrain
python -m visualization.spine_masks.visualize \
  --demo \
  --weights weights/pretrained/dinov2_vitb14_random_init_100pass_v2/teacher_training_378562.pth \
  --output-dir outputs/visualization/spine_masks_random100_l8
```

`--demo` 使用此前查看的六张**原始 X-ray**，不把 attention 拼图作为模型输入。
也可用 `--image /path/to/source.png`（可重复）或 `--image-dir /path/to/images`，
目录可加 `--recursive`、`--limit 100`。输出目录必须是新目录，防止覆盖其他实验。

默认自动选择设备；无 GPU 时使用 CPU，默认四个 CPU 线程。
本版明确支持现有 DINOv2 ViT-B/14 extracted backbone；不要传完整 FSDP 训练 checkpoint。
可替换 `--weights` 对比 MAIRA2 或其他训练阶段。无需下载权重或额外安装训练包。
绘图依赖与 `visualization/dino_spine_maps` 相同。

## 预处理与响应

- 默认 `--anchor-mode both`：每图同时输出 `clean` 和 `train`。
- `clean`：518 方形 letterbox、相同归一化和 valid mask，关闭光度增强与翻转。
- `train`：直接调用当前 `DataAugmentationGeoTopoDINO._full_fov_anchor()`，
  复用实际增强实现与默认翻转概率；这是固定随机种子的一次训练视图抽样。
  不另行假定 blur 参数的语义，不重写训练增强。
- `--anchor-size 518` 默认对应 37×37 patch 网格。
- `--layer 8 --heads 4,5,8` 均为 **1-based**。这些头只是一组待验证候选，
  不保证适用于其他初始化、checkpoint 或随机种子。
- `--heads all` 使用全部头。
- `--response-source selected_attention|mean_attention|cosine`。
  前两者先按每个头在有效区域的 attention mass 归一化，再等权融合；
  `cosine` 使用最后一层的 CLS–patch cosine，并取非负部分。
- 保留原始各头 attention、所选层及最后一层 cosine、valid attention mass。
  patch mass、图的亮度和本工具的 confidence proxy 都不是解剖定位概率。

## 四种同预算 mask

1. **block**：调用现有训练 collate 的 `_block_mask_within_valid`。
2. **vertical_span**：沿上下方向选择完整横向区段，再补足预算；不依赖 teacher。
3. **band_random**：在候选脊柱带内随机选取 patch，带外补充 mask。
4. **topology_span**：在候选脊柱带内遮挡连续完整行，再按带内配额与总预算补充 mask。

默认 `--mask-ratio 0.4`，分母是有效图像 patch 数，**不包含 padding**。
四种 mask 的总 patch 数严格相同。成功提案中，band_random 与 topology_span
具有完全相同的带内 patch 数和完全相同的带外补充 mask，用于分离区域与连续性因素。
补充遮挡优先选用 baseline 的可用 block 位置，余量随机补足；按配额取子集可能拆散原 block，
因此目前是位置优先的补充策略，不保证补充部分仍是完整矩形。

本版定位是一个简单、可检查的启发式：有效区域响应平滑 → 有限横向步长的
动态规划路径 → 保留多个有响应支持且足够长的区段 → 按有效图像宽度扩展粗带。
默认区段至少 7 行（连续 span 至少 3 行，上下各保留 2 行）。多个区段之间不跨空隙连接；
先等概率抽取可用区段，再抽取段内位置。
没有绝对横向居中先验；连续区段按 patch 行定义，尚非椎体分割、椎体编号或弧长分段。
带宽目前是按图像有效宽度缩放的配置值，不是估计出的真实椎体宽度。
它会受身体边缘、胸廓、金属和局部高响应干扰，应通过输出发现失败。

新版默认有三个不同分母的比例：

- `--span-fraction 0.5`：连续 span 约占**本次选中区段的长度** 50%，不是全部掩码的 50%。
- `--band-mask-ratio 0.5`：最终遮掉**全部候选带 patch** 的约 50%，整数取整；
  如单段 span 未达到该配额，可在其他允许的带内位置补充。
- `--mask-ratio 0.4`：最终总掩码仍占**有效图像 patch** 的约 40%，向下取整。

仅保护所选 span 紧邻的上下各 `--context-rows 2` 行带内 patch，其他带内位置可以补充遮挡。
这些橙色区域仅表示保留 token，不能保证已经保留完整相邻椎体。
span 只按完整行缩短，不随机打洞；带内配额若与总预算/上下文不兼容，会在可行范围调整并记录
`band_budget_adjusted`。随机带内对照保留同一上下文，带内数量与带外位置均与连续方案一致。
不依赖 teacher 的 `vertical_span` 对照维持旧版 25% 长度、上下各 1 行的设置。
如果覆盖不足、带内外对比不足、边缘响应主导、无有效路径，或预算与保留上下文
约束不兼容，则两个 teacher 引导策略均回退到**同一原始 block mask**。
候选带仍会展示，最终图明确标注 `BLOCK FALLBACK`，便于审查被拒绝的定位。
这些判断只控制启发式提案接受，不证明解剖正确。

可配置起点：

```bash
--span-fraction 0.5 --band-mask-ratio 0.5 --context-rows 2 \
--band-half-width-fraction 0.12 \
--min-coverage 0.25 --min-contrast 1.4 --max-border-mass 0.55
```

调参应依据独立开发集，不能以提高六张示例图的接受率为目标。

诊断量的定义：`coverage` 是候选路径被采用的行数 / 有效图像区域的行数，
**不是与人工标注比较的脊柱覆盖率**；`contrast` 是平滑响应的带内均值 / 带外均值；
`border_mass` 是有效矩形最外一圈 patch 的响应占比。身体内部的轮廓不一定落在
这一圈内，因此通过这些阈值仍可能错误定位。`span_band_fraction` 是连续 span / 全部候选带；
`band_mask_ratio_actual` 是最终掩码与候选带的交集 / 全部候选带；
`span_total_mask_fraction` 是连续 span / 全部掩码。三者不要混淆。
回退时 `band_mask_ratio_actual` 统计实际 block 与候选带的交集，不宣称满足目标配额。

## 使用已保存注意力做同输入对比

```bash
python -m visualization.spine_masks.replay \
  --from-run outputs/visualization/spine_masks_random100_l8_heads458_v1 \
  --output-dir outputs/visualization/spine_masks_random100_l8_heads458_v2
```

不重新推理，不改变权重，复用旧版的 anchor、原始响应、block mask 和样本种子。
新目录保存新版配置和来源信息，旧目录不会被覆盖。额外生成 `revision.png` 及
`revision_overview.jpg`，便于核对旧/新候选带与最终掩码。新版算法、配额和随机抽样流程均有改变，
所以连续段的位置可能变化；该对比不是只改变一个超参数的消融实验。

## 输出与审查

```text
output_dir/
├── run_config.json          # checkpoint 路径/大小/时间、参数、版本
├── overview.jpg             # 全部对比图缩略总览
├── summary.csv              # 覆盖、带内外对比、回退原因、实际遮挡预算
├── summary.json             # 完整诊断和失败样本
└── image_name/clean|train/
    ├── anchor.png           # 真正送入 teacher 的视图（反归一化显示）
    ├── mask_steps.png       # 优先看：中文分步说明，候选带/连续段/补充/最终结果
    ├── revision.png         # replay 专有：旧/新同输入比较
    ├── comparison.png       # 响应/带/四种掩码并排对比
    ├── heads.png            # 第 8 层等所选层的全部头
    ├── metadata.json        # 几何变换、seed、valid mass、诊断
    ├── valid_grid.png
    ├── mask_*.png           # patch 分辨率二值 mask，白=遮挡
    └── masks_and_maps.npz   # 原始响应、band、centerline、span、最终 masks
```

`mask_steps.png`：**绿=候选区域；蓝=连续遮挡；紫=补充遮挡；橙=保留可见的上下文**。
第⑥幅原 block 对照用蓝色标记所有遮挡位置；`comparison.png` 的四种最终掩码也统一用蓝色。
颜色只用于显示 token 位置，不是把彩色像素送入模型。后续训练接入应
在 patch embedding 后替换 mask token，与当前 iBOT 语义一致。
中心线路径坐标是 anchor patch 网格，`-1` 表示该行无被采用的路径。
`metadata.json` 中的变换矩阵可用于映射回原始图像。

先检查是否定位到脊柱、是否覆盖片段宽度、上下文是否保留，尤其比较 clean/train
及不同来源、正侧位、局部视野。视觉合理后还需同起点同预算的短程训练评估收益。
本工具没有启用 relation loss、没有改变 GCVD，也没有把候选头当作固定语义头。

## 验证

```bash
PYTHONPATH="$PWD/upstream/dinov2-main:$PWD" \
  python -m unittest discover -s methods/geotopo_dino/tests -v
```

新增测试覆盖同预算、padding 排除、连续 span、多区段抽样不跨空隙、带内配额、可见上下文、匹配随机对照、
均匀/无效/边缘响应回退、极端预算、确定性、输入与 RNG 不变，以及训练 anchor 的逐元素一致性。
