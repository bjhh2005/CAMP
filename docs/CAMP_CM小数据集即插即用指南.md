# CAMP-CM CIFAR-10 运行指南

这份文档只保留当前要用的 CIFAR-10 实验流程。`CM4IR/` 和 `FreqPure/` 仍只作为参考代码；CAMP 自有代码在 `experiments/camp/`。

## 1. 已配置路径

当前模板默认使用你的服务器路径：

```text
数据集:     /SSD_Data01/HHY/datasets/CIFAR-10
分类器:     /SSD_Data01/HHY/classifiers/cifar10_resnet56/cifar10_resnet56.pt
一致性模型: /SSD_Data01/HHY/generators/cm_cifar_10/cd-lpips-cifar10.pt
```

路径规则：

- `dataset.root` 写数据集根目录，不写文件名。你当前目录下有 `cifar-10-batches-py`，这是正确的。
- `classifier.kwargs.checkpoint` 推荐写到具体 `.pt/.pth/.ckpt` 文件。
- `purification.model_kwargs.checkpoint` 推荐写到具体 CM checkpoint 文件。
- `purification.model_kwargs.ctm_repo` 必须是 Sony/CTM 代码仓库目录，且里面应存在 `code/cm/script_util.py`。如果 `/SSD_Data01/HHY/generators/cm_cifar_10` 只是权重目录，就需要把 `ctm_repo` 改成真实代码仓库路径。

数据、模型、输出与代码分离。不要把数据集、权重、运行输出提交到 Git。

## 2. 直接运行

先检查配置：

```bash
bash scripts/camp_cifar10_dry_run.sh
```

这个脚本会做两件事：

1. 展开并打印配置。
2. 检查数据、分类器 checkpoint、CM checkpoint、`ctm_repo/code/cm/script_util.py` 是否存在。

跑 16 张冒烟测试：

```bash
bash scripts/camp_cifar10_smoke.sh
```

跑 baseline：

```bash
MAX_SAMPLES=256 OUTPUT_DIR=outputs/camp/cifar10_baseline \
  bash scripts/camp_cifar10_baseline.sh
```

跑小波噪声版本：

```bash
MAX_SAMPLES=256 OUTPUT_DIR=outputs/camp/cifar10_wavelet_noise \
  bash scripts/camp_cifar10_wavelet.sh
```

扫描加噪强度 `iN`：

```bash
MAX_SAMPLES=256 I_N_VALUES="20 40 80" \
  bash scripts/camp_cifar10_sweep_iN.sh
```

## 3. 输出怎么看

每次运行会在 `OUTPUT_DIR` 下生成：

```text
resolved_config.json   # 本次实际配置
summary.json           # 指标与逐样本结果
analysis.md            # 自动分析文档
images/*_triplet.png   # clean / adv / purified 三联图
```

优先看 `analysis.md`。里面包含：

- 关键指标：攻击成功率、净化恢复率、净化后一致率
- 关键参数：攻击、CM 时间步、小波、BP 等
- 若干 `clean / adv / purified` 可视化图
- 失败样本表

已有结果也可以补生成报告：

```bash
bash scripts/camp_make_report.sh outputs/camp/cifar10_baseline
```

## 4. configs 是什么

日常只需要关注两个配置：

| 配置 | 作用 |
|---|---|
| `experiments/camp/configs/cifar10_cm_baseline.yaml` | 无小波、无 BP 的 CM 净化基线 |
| `experiments/camp/configs/cifar10_cm_wavelet_noise.yaml` | 在 `z_hat_minus` 上启用小波高频增益 |

其他配置是通用模板或 debug 用：

| 配置 | 作用 |
|---|---|
| `small_debug.yaml` | 无真实权重的工程冒烟 |
| `small_cm_template.yaml` | 其他小数据集模板 |
| `cm_purification_*.yaml` | 后续扩展到通用图像/高分辨率时用 |

## 5. 后续怎么改参数

不要直接改模板。复制一份本地配置：

```bash
cp experiments/camp/configs/cifar10_cm_baseline.yaml local_cifar10_iN40.yaml
```

`local_*.yaml` 已被 `.gitignore` 忽略，适合放临时实验参数。

常改字段：

```yaml
dataset:
  max_samples: 512

attack:
  steps: 20

purification:
  schedule:
    sampling_steps: 4
    iN: 40
    gamma: 0.02
    eta: 0.0
  wavelet_noise:
    enabled: true
    wavelet: db2
    levels: 1
    gains: [1.1]
  bp:
    enabled: false
```

用本地配置运行：

```bash
CONFIG=local_cifar10_iN40.yaml \
MAX_SAMPLES=512 \
OUTPUT_DIR=outputs/camp/local_cifar10_iN40 \
bash scripts/camp_cifar10_baseline.sh
```

## 6. checkpoint 注意事项

当前配置允许 `checkpoint` 指向目录，程序会自动选择目录下排序后的第一个 `.pt/.pth/.ckpt`。

为了复现实验，建议确认后改成具体文件：

```yaml
classifier:
  kwargs:
    checkpoint: /SSD_Data01/HHY/classifiers/cifar10_resnet56/resnet56_best.pth

purification:
  model_kwargs:
    ctm_repo: /path/to/sony_ctm_repo
    checkpoint: /SSD_Data01/HHY/generators/cm_cifar_10/cm_cifar10.pt
```
