# AWDD v2.1 实验文档（strict 落地版）

## 1. 文档目标

这份文档用于统一 3 件事：

1. 讲清楚仓库里“当前可运行方法”的真实步骤（避免口径漂移）。
2. 说明实验原理与评测逻辑（为什么这么评）。
3. 给出可直接执行的实验流程，为后续代码升级到 `AWDD-v2.1-strict` 提供基线。

说明：

- 当前仓库已落地 `AWDD-v2.1-strict` 的核心行为（在 `adaptive_ms` 模式下生效）。
- `hard/fused` 仍保留用于消融对照。

---

## 2. 方法总览（当前实现）

当前 WGCP/CTM 流程可概括为：

1. 对输入图像生成对抗样本 `X_adv`（FGSM/PGD）。
2. 将 `X_adv` 加噪到 `t*`，送入 predictor（高斯占位或 CTM 模块）得到 `X_hat`。
3. 在小波域做低/高频重组，输出第一次净化结果 `X_corrected`。
4. 可选执行 `self-correct` 循环（`k` 次），得到最终 `X_final`。
5. 用分类器比较 `clean / adv / purified` 的预测一致性与恢复率。

---

## 3. 频域细节（按模式）

## 3.1 `hard` 模式

单层 DWT，按配置直接选低频锚点和高频来源：

- 低频：`LL_anchor = LL_orig` 或 `LL_hat`
- 高频：`HF_selected = HF_hat` 或 `HF_orig`
- 重建：`IDWT(LL_anchor, HF_selected)`

用途：最小可解释基线，速度快，便于对比。

## 3.2 `fused` 模式

单层 DWT，低频同 `hard`，高频做“预测高频 + 原图收缩高频”融合：

1. 估计鲁棒噪声尺度（按通道）：
   `sigma = median(|HF_orig|) / 0.6745`
2. 软阈值：
   `HF_orig_denoised = soft_shrink(HF_orig, tau)`，`tau = hf_shrink * sigma`
3. 高频融合：
   `HF_final = (1 - preserve) * HF_hat + preserve * HF_orig_denoised`

用途：比 `hard` 更平滑，通常视觉指标更稳。

## 3.3 `adaptive_ms` 模式（AWDD v2.1 strict）

多层 DWT（默认 3 层），对 `X_adv` 与 `X_hat` 同态分解：

1. 低频融合（CTM 主导）：
   `LL_final = (1 - a) * LL_pred + a * LL_orig`，其中 `a` 建议 `0.05~0.1`。
2. 高频净化（仅原图子带）：
   - `sigma_{l,b} = MAD(HF_orig_{l,b}) / 0.6745`
   - `lambda_{l,b} = gamma_l * sigma_{l,b} * sqrt(2 log N_{l,b})`
   - `HF_final_{l,b} = soft_shrink(HF_orig_{l,b}, lambda_{l,b})`
3. 不再融合 `HF_pred`（CTM 不参与高频重建）。
4. `IDWT` 重建输出。

用途：实现“低频归扩散，高频归解析”的非对称解耦。

---

## 4. 实现冻结点（当前代码）

为保证可复现，当前版本固定以下行为：

1. 默认 `wavelet=db4`。
2. DWT/IDWT/WaveDec/WaveRec 统一使用 `mode=reflect`。
3. `adaptive_ms` 的高频为“orig-only soft-shrinkage”。
4. `ms_w_min/ms_w_max` 仅为兼容旧命令保留，当前为 no-op。

---

## 5. 实验原理与评测口径

## 5.1 伪标签评测（当前脚本）

`experiments/wgcp_attack_eval.py` 默认使用 clean 图预测类别作为伪标签，不依赖真值标注：

1. `clean_pred`：分类器对 clean 图预测。
2. 攻击目标：让 `adv_pred != clean_pred`。
3. 净化恢复：看 `purified_pred == clean_pred`。

优点：可在任意无标签图集快速评估。  
限制：不是标准 top-1 准确率，不可直接替代带 GT 的鲁棒准确率。

## 5.2 关键聚合指标

- `attack_success_rate`：攻击成功率
- `recover_rate_on_attacked`：在被攻破样本上的恢复率
- `clean_pred_consistency_rate`：净化后与 clean 预测一致比例
- `clean_vs_purified.SSIM/PSNR`：感知保真
- `NFE`、`single_step_infer_ms`：效率成本

## 5.3 防御结论的边界

当前实验主要回答“经验有效性”，不能直接得出“免疫自适应攻击”结论。  
后续需要补充：端到端自适应白盒（长步 PGD、多重重启、AutoAttack/BPDA 近似）。

---

## 6. 复现实验步骤（strict 基线）

## 6.1 小规模冒烟（6 张）

```bash
python experiments/wgcp_attack_eval.py \
  --input_dir data/imagenet_real \
  --glob "*.JPEG" \
  --output_dir outputs/awdd_v21_smoke \
  --max_images 6 \
  --attack pgd \
  --eps 0.0313725 \
  --pgd_steps 10 \
  --pgd_alpha 0.0078431 \
  --classifier resnet50 \
  --t_star 40 \
  --self_correct_k 0 \
  --replacement_mode adaptive_ms \
  --ms_levels 3 \
  --ms_gamma_levels 1.6,1.2,0.9 \
  --ms_w_min 0.05 \
  --ms_w_max 0.95 \
  --ms_ll_alpha 0.08 \
  --lightweight_mode \
  --save_reference_every 1
```

## 6.2 基线统计（100 张）

```bash
python experiments/wgcp_attack_eval.py \
  --input_dir data/imagenet_real \
  --glob "*.JPEG" \
  --output_dir outputs/awdd_v21_current_baseline \
  --max_images 100 \
  --attack pgd \
  --eps 0.0313725 \
  --pgd_steps 10 \
  --pgd_alpha 0.0078431 \
  --classifier resnet50 \
  --t_star 40 \
  --self_correct_k 0 \
  --replacement_mode adaptive_ms \
  --ms_levels 3 \
  --ms_gamma_levels 1.6,1.2,0.9 \
  --ms_w_min 0.05 \
  --ms_w_max 0.95 \
  --ms_ll_alpha 0.08 \
  --lightweight_mode \
  --save_reference_every 10
```

## 6.3 小波基对比（strict 下）

保持其余参数不变，单独扫描：

1. `--wavelet db4`
2. `--wavelet bior4.4`
3. `--wavelet sym4`（可选）

建议每组先跑 30 张，再扩到 100 张。

---

## 7. 结果归档与记录规范

每次实验至少保留：

1. `summary.json`
2. 命令行参数（脚本已写入 `run_meta.command`）
3. git commit（脚本已写入 `run_meta.git_commit`）
4. 对应输出目录命名（含日期/实验名）

推荐目录命名：

- `outputs/awdd_v21_current_baseline_*`
- `outputs/awdd_v21_wavelet_db4_*`
- `outputs/awdd_v21_wavelet_bior44_*`

---

## 8. 下一阶段（评测升级）

在 strict 基线稳定后，建议直接做自适应攻击升级评测：

1. 同数据、同攻击预算、同分类器。
2. 仅替换方法实现，不改评测脚本。
3. 对比 `recover_rate_on_attacked + SSIM/PSNR + NFE`。

这样可以把“算法增益”与“实验条件变化”严格分离。
