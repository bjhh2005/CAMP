# CAMP-CM 小数据集即插即用指南

## 1. 项目边界

CAMP 自有代码只放在 `experiments/camp/`。`CM4IR/` 和 `FreqPure/` 只作为参考代码，不直接修改。

数据、模型、输出必须和代码分离：

- 数据放在仓库外，例如 `D:\Datasets\camp_small` 或 `/data/camp_small`
- 模型放在仓库外，例如 `D:\Models\camp` 或 `/data/models/camp`
- 输出放在仓库外或 `outputs/`，不要提交到 Git

## 2. 推荐目录

Windows 示例：

```text
D:\Repositories\CAMP                  # 代码仓库
D:\Datasets\camp_small\images         # 小数据集图片
D:\Models\camp                        # 分类器和生成模型权重
D:\Users\<you>\camp_runs              # 实验输出，或任何你有写权限的目录
```

Linux 服务器示例：

```text
/home/you/CAMP                        # 代码仓库
/data/camp_small/images               # 小数据集图片
/data/models/camp                     # 分类器和生成模型权重
/data/runs/camp                       # 实验输出
```

`image_folder` 数据格式：

```text
images/
  000001.png
  000002.png
  ...
```

`class_folder` 数据格式：

```text
images/
  class_a/
    000001.png
  class_b/
    000002.png
```

当前评估默认使用分类器 clean prediction 作为 pseudo-label，所以 `image_folder` 足够先跑 PGD 净化链路。

## 3. 环境变量

Windows PowerShell：

```powershell
$env:CAMP_DATA_ROOT="D:\Datasets\camp_small"
$env:CAMP_MODEL_ROOT="D:\Models\camp"
$env:CAMP_OUTPUT_ROOT="$HOME\camp_runs"
```

Linux：

```bash
export CAMP_DATA_ROOT=/data/camp_small
export CAMP_MODEL_ROOT=/data/models/camp
export CAMP_OUTPUT_ROOT=/data/runs/camp
```

配置文件会自动展开 `${CAMP_DATA_ROOT}`、`${CAMP_MODEL_ROOT}`、`${CAMP_OUTPUT_ROOT}`。

## 4. 本地冒烟测试

这个命令不需要真实 CM 权重，只检查数据读取、PGD、净化循环、图片输出是否连通：

```bash
python -m experiments.camp.run_purification \
  --config experiments/camp/configs/small_debug.yaml
```

只检查配置展开、不读写数据：

```bash
python -m experiments.camp.run_purification \
  --config experiments/camp/configs/small_debug.yaml \
  --dry_run
```

输出：

```text
${CAMP_OUTPUT_ROOT}/small_debug/
  resolved_config.json
  summary.json
  images/
```

注意：`small_debug.yaml` 使用未训练分类器和 Gaussian debug purifier，只用于检查工程链路，不代表研究结果。

## 5. 真实小数据集实验

复制模板，不要直接改模板：

```bash
cp experiments/camp/configs/small_cm_template.yaml local_small_cm.yaml
```

需要替换：

- `classifier.module`
- `classifier.kwargs.checkpoint`
- `purification.model_module`
- `purification.model_kwargs.ctm_repo`
- `purification.model_kwargs.checkpoint`
- `dataset.root`
- `evaluation.output_dir`

运行：

```bash
python -m experiments.camp.run_purification \
  --config local_small_cm.yaml
```

建议小数据集第一轮参数：

- `dataset.image_size: 64`
- `dataset.max_samples: 64`
- `attack.steps: 10`
- `purification.schedule.sampling_steps: 2` 或 `4`
- `purification.schedule.iN: 20, 40, 80` 分别扫
- `purification.bp.enabled: false`

## 6. 三组最小对照

先跑这三组，不要一开始铺太大：

1. Baseline：`wavelet_noise.enabled=false`, `bp.enabled=false`
2. Wavelet noise：`wavelet_noise.enabled=true`, `bp.enabled=false`
3. Joint：`wavelet_noise.enabled=true`, `bp.enabled=true`, `bp.mu=0.05` 或 `0.1`

可从以下模板开始：

- `experiments/camp/configs/cm_purification_baseline.yaml`
- `experiments/camp/configs/cm_purification_wavelet_noise.yaml`
- `experiments/camp/configs/cm_purification_wavelet_noise_bp.yaml`
- `experiments/camp/configs/small_cm_template.yaml`

## 7. 输出指标

每次运行生成：

- `resolved_config.json`：完整展开后的配置
- `summary.json`：聚合结果和逐样本结果

重点看：

- `attack_success_rate`
- `recover_rate_on_attacked`
- `purified_same_as_clean_rate`
- 每个样本的 `sigma_schedule`

## 8. 当前源码入口

- `experiments/camp/run_purification.py`：主入口
- `experiments/camp/cm_purifier.py`：少步 CM 净化循环
- `experiments/camp/wavelet_ops.py`：小波噪声注入和小波 BP
- `experiments/camp/configs/`：配置模板
