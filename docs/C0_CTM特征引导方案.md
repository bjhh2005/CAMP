# C0：CTM 特征引导的小波结构可信度方案

## 1. 结论先行

当前对 A / B / C1 的分析总体是合理的，但需要补上一条严格表述：

1. 可以认为 A / B 的负结果支持了“纯图像域边缘保护不足以稳定提升对抗恢复率”。
2. 但暂时还不能把结论写成“边缘保护本身一定伤害防御能力”，因为现有 A / B 与 C1 不是单变量对比。

根因在于当前实现里：

1. `adaptive_ms`（C1 对应主线）使用低频软锚定：`ll_final = (1 - a) * ll_pred + a * ll_orig`。
2. `adaptive_ms_edge` 与 `adaptive_ms_modmax`（A / B 对应主线）目前固定 `ll_final = ll_orig`。

也就是说，A / B 相对 C1 同时改变了两件事：

1. 高频收缩策略
2. 低频锚点策略

因此，现阶段最稳妥的论文叙事应写为：

> 在当前实现下，基于图像域能量或模极大值的多尺度边缘保护没有带来优于 C1 的恢复收益；结合方法机理，强烈提示图像域边缘显著性无法可靠区分真实结构与对抗污染结构。

## 2. 现有实现对应关系

相关入口如下：

1. [`experiments/wgcp_wavelet.py`](/d:/Repositories/CAMP/experiments/wgcp_wavelet.py)
   `adaptive_multiscale_fusion` 是 C1 主线。
2. [`experiments/wgcp_wavelet.py`](/d:/Repositories/CAMP/experiments/wgcp_wavelet.py)
   `adaptive_multiscale_edge_fusion` 是 A 方案。
3. [`experiments/wgcp_wavelet.py`](/d:/Repositories/CAMP/experiments/wgcp_wavelet.py)
   `adaptive_multiscale_modulus_fusion` 是 B 方案。
4. [`experiments/wgcp_wavelet.py`](/d:/Repositories/CAMP/experiments/wgcp_wavelet.py)
   `adaptive_multiscale_guided_fusion` 已经提供了“引入外部引导图”的接口思路，可作为 C0 的落脚点。
5. [`experiments/wgcp_purify.py`](/d:/Repositories/CAMP/experiments/wgcp_purify.py)
   `_reconstruct_from_prediction(...)` 是各 `replacement_mode` 的总调度入口。
6. [`experiments/wgcp_attack_eval.py`](/d:/Repositories/CAMP/experiments/wgcp_attack_eval.py)
   已暴露 `adaptive_ms` / `adaptive_ms_guided` / `adaptive_ms_edge` / `adaptive_ms_modmax` 等模式，是后续新增 C0 参数的主入口。

## 3. 为什么 C0 值得做

你提出的核心判断是成立的：

1. A 的边缘活动图来自局部能量，默认假设“高能量 = 可信边缘”。
2. B 的模极大值图本质上仍是“高频几何显著性”判据。
3. 对抗扰动恰恰会寄生在真实边缘附近，或直接制造看起来很像结构的高频响应。

因此，图像域判据很容易把“高频显著”误认为“结构可信”。

而 CTM 中间特征的价值在于：

1. 它不是直接用像素幅值做判决。
2. 它更接近“模型是否把这里当成稳定结构”的响应。
3. 如果响应图确实对真实结构强、对扰动弱，就能天然替代 A / B 里最脆弱的那一环。

## 4. C0 的目标定义

C0 不让 CTM 直接生成高频内容，也不让 CTM 接管像素重建。

CTM 只做一件事：

> 提供一张结构可信度图 `S(x)`，用于调制小波高频收缩强度。

这样能同时保留两条已有经验：

1. 继续继承 C1 “低频交给 CTM / 高频仍以解析收缩为主”的稳定性。
2. 避免重走 G1 一类“让 CTM 直接写高频细节”的老路。

## 5. 最小可行方案

### 5.1 输入与主干

对每张 `X_adv`：

1. 按现有流程得到 `X_hat`。
2. 保留 C1 的低频软锚定不变。
3. 仅替换高频收缩里的 `alpha` 生成方式。

### 5.2 结构可信度图

新增一个特征提取步骤：

1. 对 `X_adv` 或 `X_hat` 做一次 CTM 前向。
2. 从 CTM 的一个中间层拿到特征图 `F in R^{C×H×W}`。
3. 由 `F` 构造单通道可信度图 `S`。

第一版只做最简单、可解释的两种候选：

1. 激活能量图：`S = mean_c(abs(F_c))`
2. 特征梯度图：先对 `mean_c(F_c)` 求 Sobel/Laplacian，再归一化

推荐先从激活能量图开始，因为：

1. 工程最简单
2. 不引入额外导数不稳定性
3. 更容易先看可视化是否“有感觉”

### 5.3 从 `S` 到小波收缩

把 `S` resize 到每个小波 level / band 的空间大小，得到 `S_l`。

新的高频收缩因子定义为：

`alpha_l = clip(1 - eta_l * norm(S_l), alpha_min, 1.0)`

然后继续沿用 C1 / A 已有的软阈值框架：

`tau_l,b = gamma_l * sigma_l,b * sqrt(2 log N_l,b)`

`HF_final_l,b = soft_shrink(HF_orig_l,b, tau_l,b * alpha_l)`

这里最重要的是：

1. 高频仍然来自 `HF_orig`
2. `tau` 的统计估计仍然沿用鲁棒 MAD
3. 只把“哪里该少缩一点”从图像域边缘图改成 CTM 特征可信图

## 6. 必须先做的两步校准

在正式做 C0 前，先补两步最关键的控制实验，避免后面论文被人一句话打回来。

### 6.1 控制实验 E1：消除低频混杂

目标：

把 A / B 改成和 C1 完全相同的低频软锚定，只比较高频策略。

做法：

1. 为 `adaptive_ms_edge` 和 `adaptive_ms_modmax` 增加 `ll_alpha` 版本。
2. 让它们改成：
   `ll_final = (1 - ll_alpha) * ll_pred + ll_alpha * ll_orig`
3. 复跑 A / B / C1 的同集对照。

意义：

如果此时 A / B 仍低于 C1，就能更有力地支持“图像域边缘保护本身不可靠”。

### 6.2 轻量验证 E2：先看特征图有没有区分度

目标：

先验证 CTM 特征图是否真的对真实结构强、对扰动弱。

做法：

1. 选 5 到 10 张典型失败样本。
2. 可视化：
   `clean / adv / purified(C1) / CTM feature heatmap / image gradient map`
3. 肉眼比较热力图是否更集中在语义结构区，而不是攻击噪点区。

通过标准：

只要在大多数样本上，CTM 热力图比像素梯度图更少追随扰动纹理，就值得继续做 C0。

## 7. 实施路径

### Stage 0：只做分析与可视化

新增一个最小脚本或评估分支，用于导出：

1. `X_clean`
2. `X_adv`
3. `X_hat`
4. CTM 中间层热力图
5. 图像域梯度热力图
6. 小波 level-1 / level-2 对应的 `alpha map`

目标不是跑分，而是确认“信号是否存在”。

### Stage 1：做最小 C0 原型

建议新增模式：

1. `adaptive_ms_ctm_guided`

建议复用的位置：

1. 在 [`experiments/ctm_adapter_sony.py`](/d:/Repositories/CAMP/experiments/ctm_adapter_sony.py) 中增加可选的中间特征导出能力。
2. 在 [`experiments/wgcp_wavelet.py`](/d:/Repositories/CAMP/experiments/wgcp_wavelet.py) 中新增 `adaptive_multiscale_ctm_guided_fusion(...)`。
3. 在 [`experiments/wgcp_purify.py`](/d:/Repositories/CAMP/experiments/wgcp_purify.py) 中接入新的 `replacement_mode`。
4. 在 [`experiments/wgcp_attack_eval.py`](/d:/Repositories/CAMP/experiments/wgcp_attack_eval.py) 中暴露参数。

第一版建议参数：

1. `--ms_ctm_feature_source adv|hat`，默认 `hat`
2. `--ms_ctm_feature_reduce mean_abs|grad`，默认 `mean_abs`
3. `--ms_ctm_layer <name-or-index>`
4. `--ms_ctm_alpha_min`
5. `--ms_ctm_eta_levels`

### Stage 2：跑最小对照矩阵

只保留 4 组，避免一开始铺太大：

1. `C1`: `adaptive_ms`
2. `A'`: 与 C1 同低频锚定的 `adaptive_ms_edge`
3. `B'`: 与 C1 同低频锚定的 `adaptive_ms_modmax`
4. `C0`: `adaptive_ms_ctm_guided`

固定：

1. 同一批样本
2. 同一攻击预算
3. 同一 wavelet
4. 同一 `t_star`
5. 同一 `self_correct_k`

### Stage 3：决定是否扩展

若 C0 相对 C1 至少满足以下任一条件，再扩张实验：

1. `recover_rate_on_attacked` 提升 >= 1.5 个百分点
2. 恢复率持平，但 `clean_pred_consistency_rate` 与边缘保真指标同步更优
3. 失败案例中出现稳定的“扰动区抑制、结构区保留”可视化模式

否则就及时止损，把 C0 降级为“探索性负结果”。

## 8. 指标与可视化要求

除了现有聚合指标，C0 阶段建议新增三类诊断量。

### 8.1 结构图一致性

记录：

1. `mean(S on clean edges)`
2. `mean(S on adv-only residual regions)`
3. `S / image-gradient` 的相关性

目的：

证明 `S` 不是简单复刻图像梯度。

### 8.2 高频保留诊断

按 level / band 记录：

1. `alpha_mean`
2. `tau_mean`
3. `||HF_final|| / ||HF_orig||`

目的：

看 C0 是否只是“整体少缩了”，还是在空间上更有选择性。

### 8.3 失败案例面板

每个模式至少保存：

1. 原图
2. 对抗图
3. 净化图
4. 高频差分图
5. 可信度图 / 边缘图

目的：

后续论文可以直接用作机制解释图。

## 9. 风险与对应策略

### 风险1：CTM 中间特征提取不稳定

处理：

1. 第一版只取单层
2. 不做复杂多层融合
3. 优先使用前向 hook 读取，不改 CTM 主推理路径

### 风险2：特征图分辨率太低

处理：

1. 先只作用于 level-2 / level-3
2. level-1 仍沿用 C1 原始 shrinkage

### 风险3：特征图只是另一种边缘图

处理：

1. 必须做 E2 可视化
2. 必须记录 `S` 与像素梯度图的相关性
3. 如果高度同质，就不继续加大投入

### 风险4：工程改动过大

处理：

1. 第一版不碰自校正循环
2. 第一版不碰 patch 模式
3. 先在 `self_correct_k=0` 的主线下验证

## 10. 推荐执行顺序

1. 先补 E1，把 A / B 与 C1 对齐到同一低频锚定。
2. 再做 E2，只看 CTM 热力图有没有结构区分度。
3. 只有 E2 过关，才实现最小 C0 原型。
4. 只跑 `C1 / A' / B' / C0` 四组最小矩阵。
5. 若 C0 有正向信号，再考虑 patch 版或 self-correct 版扩展。

## 11. 当前建议的论文表述

如果现在就要写阶段性结论，建议使用下面这个版本：

> C1 建立了稳定的小波净化基线；A / B 说明单纯依赖图像域边缘显著性进行高频保护，无法稳定提升对抗恢复率。结合对抗扰动在高频边缘附近的寄生特性，我们进一步提出 C0：利用 CTM 中间特征提供结构可信度，仅参与高频收缩调制，而不直接参与像素级高频生成。

这个版本既保住了你现在这批实验的价值，也给下一步留下了非常清晰、可执行的技术路线。
