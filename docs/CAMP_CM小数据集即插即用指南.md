# CAMP-CM CIFAR-10 运行指南

这份文档只保留当前 CIFAR-10 实验流程。`CM4IR/` 和 `FreqPure/` 只作为参考代码；CAMP 自有代码在 `experiments/camp/`。

## 1. 当前路径

```text
数据集:     /SSD_Data01/HHY/datasets/CIFAR-10
分类器:     /SSD_Data01/HHY/classifiers/cifar10_resnet56/cifar10_resnet56.pt
一致性模型: /SSD_Data01/HHY/generators/cm_cifar_10/cd-lpips-cifar10.pt
```

路径规则：

- `dataset.root` 写数据集根目录，不写文件名；你当前目录下有 `cifar-10-batches-py`，这是正确的。
- `classifier.kwargs.checkpoint` 写到具体分类器权重文件。
- `purification.model_kwargs.checkpoint` 写到具体 CM 权重文件。
- 数据、模型、输出与代码分离，不要提交到 Git。

## 2. 生成模型 backend

现在主流程只调用统一接口：

```text
backend.predict_x0(x_t, context) -> x0_hat
```

不同 checkpoint 格式由不同 backend adapter 处理：

| backend | 用途 | 状态 |
|---|---|---|
| `debug_gaussian` | 无权重冒烟 | 可用 |
| `sony_cm` | Sony/CTM 代码体系中的 PyTorch CM/CTM checkpoint | 可用，但要求 checkpoint 兼容 Sony 代码 |
| `openai_cifar_jax` | OpenAI CIFAR-10 JAX/Flax CM checkpoint | 已接入，会动态导入 OpenAI JAX repo 并调用 distiller 推理 |

你的 `cd-lpips-cifar10.pt` 如果 `torch.load` 报：

```text
invalid load key, '\xa4'
```

大概率是 JAX/Flax 格式，不应该用 `sony_cm`。应走：

```yaml
purification:
  backend: openai_cifar_jax
  model_input_range: minus_one_one
  model_output_range: minus_one_one
  model_kwargs:
    repo: /SSD_Data01/HHY/openai_cm_cifar10
    checkpoint: /SSD_Data01/HHY/generators/cm_cifar_10/cd-lpips-cifar10.pt
```

其中 `repo` 是 OpenAI CIFAR-10 CM 的 JAX 源码目录，目录下应有 `jcm/models/utils.py`、`jcm/checkpoints.py` 等文件。运行环境需要安装该 repo 的 JAX/Flax/Haiku/Optax 依赖。

检查 checkpoint 格式：

```bash
python -m experiments.camp.checkpoint_format \
  /SSD_Data01/HHY/generators/cm_cifar_10/cd-lpips-cifar10.pt
```

## 3. 直接运行

先检查配置和路径：

```bash
bash scripts/camp_cifar10_dry_run.sh
```

跑 16 张冒烟：

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

扫描 `iN`：

```bash
MAX_SAMPLES=256 I_N_VALUES="20 40 80" \
  bash scripts/camp_cifar10_sweep_iN.sh
```

## 4. 输出怎么看

每次运行会在 `OUTPUT_DIR` 下生成：

```text
resolved_config.json
summary.json
analysis.md
images/*_triplet.png
```

优先看 `analysis.md`。它包含关键指标、关键参数、三联图和失败样本表。

已有结果也可以补报告：

```bash
bash scripts/camp_make_report.sh outputs/camp/cifar10_baseline
```

## 5. configs 是什么

日常主要看：

| 配置 | 作用 |
|---|---|
| `cifar10_cm_openai_jax.yaml` | OpenAI CIFAR-10 JAX/Flax CM 入口 |
| `cifar10_cm_baseline.yaml` | Sony/PyTorch 兼容 checkpoint 的 baseline |
| `cifar10_cm_wavelet_noise.yaml` | Sony/PyTorch 兼容 checkpoint 的小波噪声版本 |

其他配置：

| 配置 | 作用 |
|---|---|
| `small_debug.yaml` | 无真实权重的工程冒烟 |
| `small_cm_template.yaml` | 其他小数据集模板 |
| `cm_purification_*.yaml` | 后续扩展到通用图像/高分辨率时用 |

## 6. 后续怎么改参数

不要直接改模板，复制本地配置：

```bash
cp experiments/camp/configs/cifar10_cm_openai_jax.yaml local_cifar10_iN40.yaml
```

`local_*.yaml` 已被 `.gitignore` 忽略。

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

运行本地配置：

```bash
CONFIG=local_cifar10_iN40.yaml \
MAX_SAMPLES=512 \
OUTPUT_DIR=outputs/camp/local_cifar10_iN40 \
bash scripts/camp_cifar10_baseline.sh
```
