# CTM 服务器一键配置与 WGCP 结合指南

## 1. 仅服务器执行（不在 Windows 本地配置 CTM）

```bash
cd /path/to/CAMP
bash scripts/setup_ctm_server.sh
```

这个脚本会做：

1. 用 `environment.yml` 更新 `camp` conda 环境。
2. 按需安装 `gdown`（仅下载模式需要）。
3. 拉取/更新官方 CTM 仓库（`third_party/ctm`）。
4. 解析 checkpoint 路径（优先手动提供，其次本地已有文件，最后才可选下载）。
5. 安装 CTM 运行时依赖（默认：`blobfile einops mpi4py`）。
6. 安装 CTM 仓库 requirements（默认开启）。
7. 生成 `configs/ctm_server_config.json`。

可选环境变量：

```bash
CAMP_ENV=camp \
CTM_REPO_DIR=/data/repos/ctm \
CTM_CACHE_DIR=/data/model_cache/ctm \
CTM_CHECKPOINT_PATH=/data/model_cache/ctm/ema_0.999_049000.pt \
DOWNLOAD_CKPT=0 \
DOWNLOAD_FOLDER=0 \
INSTALL_CTM_RUNTIME_REQS=1 \
INSTALL_CTM_REQS=1 \
bash scripts/setup_ctm_server.sh
```

如果你手动下载了 `ema_0.999_049000.pt`，推荐直接这样执行（不会触发 Google Drive 下载）：

```bash
CTM_CHECKPOINT_PATH=/home/HHY/models/ema_0.999_049000.pt \
DOWNLOAD_CKPT=0 \
bash scripts/setup_ctm_server.sh
```

脚本会在以下优先级选择 checkpoint：
1. `CTM_CHECKPOINT_PATH`
2. `$CTM_CACHE_DIR/ctm_imagenet64_ema999.pt`
3. `$CTM_CACHE_DIR/ema_0.999_049000.pt`
4. 仅当 `DOWNLOAD_CKPT=1` 时才尝试联网下载

说明：项目已内置 `flash_attn`/`xformers` 的 Python fallback shim，避免服务器上编译 `flash-attn` 失败导致流程中断。
说明：因此不建议在服务器手工 `pip install flash_attn`（常见编译失败），直接用本项目 fallback 更稳。
说明：`run_wgcp_ctm_server.sh` 会自动设置 `PYTHONPATH`，你不需要再手动 `export PYTHONPATH=...`。

## 2. 为什么 CTM 直接用于净化会不稳

你文档里的 WGCP 思路是对的：锁定低频 + 重写高频。问题在于公开 CTM 大多在生成数据集上训练（常见 64x64），并不是专门为了“输入一张被攻击图并恢复语义”。

所以更稳妥的结合方式是（与你的 CAMP 文档一致）：

1. 仍然锁定 `LL_orig`（语义锚点不变）。
2. 让 CTM 只提供“高频先验”，不要接管低频语义。
3. 默认采用 **硬替换**（`replacement_mode=hard`）：`IDWT(LL_orig, HF_hat)`。
4. 通过中间重加噪（`t* -> t1 -> 0`）平滑拼接边界，这是 CAMP 的关键机制。
5. 如果某些数据细节损失仍明显，再切到 `replacement_mode=fused` 做软融合。

## 3. 项目里已经支持的结合接口

现在脚本支持外部 predictor：

- `--predictor_type module`
- `--predictor_module package.module:ClassName`
- `--predictor_kwargs_json '{...}'`
- `--predictor_image_size 64`（适配 CTM-64）

实际适配器（已接好 Sony CTM）：

- `experiments/ctm_adapter_sony.py`

## 4. 一键运行（服务器）

```bash
cd /path/to/CAMP
GLOB_PATTERN="*.JPEG" \
MAX_IMAGES=100 \
LIGHTWEIGHT_MODE=1 \
SAVE_REFERENCE_EVERY=10 \
bash scripts/run_wgcp_ctm_server.sh /path/to/imagenet_real /path/to/outputs/wgcp_eval_ctm
```

如需一次性跑完整消融矩阵（A0-A5）：

```bash
GLOB_PATTERN="*.JPEG" MAX_IMAGES=100 \
bash scripts/run_ablation_ctm_server.sh /path/to/imagenet_real /path/to/outputs/wgcp_ablation_ctm
```

详细解释见：`docs/消融实验方案.md`

可选环境变量：

```bash
CTM_CLASS_COND=1 \
CTM_CLASS_LABEL=0 \
TORCH_CACHE_DIR=/data/model_cache/camp_torch \
GLOB_PATTERN="*.JPEG" \
MAX_IMAGES=100 \
LIGHTWEIGHT_MODE=1 \
SAVE_REFERENCE_EVERY=10 \
bash scripts/run_wgcp_ctm_server.sh /data/imagenet_real /data/outputs/wgcp_eval_ctm
```

归档说明：

- 每次运行都会把 `summary.json` 额外归档一份到仓库外默认目录：
  `~/.camp_runs/CAMP/wgcp_attack_eval/<timestamp>.json`
- 可选参数：
  `--archive_dir /abs/path/to/archive_root`
  `--archive_tag your_tag`
  `--disable_archive`

## 5. 手动运行命令（可选）

```bash
python experiments/wgcp_attack_eval.py \
  --input_dir data/imagenet_real \
  --glob "*.JPEG" \
  --output_dir outputs/wgcp_eval_ctm \
  --attack pgd \
  --eps 0.0313725 \
  --pgd_steps 10 \
  --pgd_alpha 0.0078431 \
  --classifier resnet50 \
  --weights_cache_dir /data/model_cache/camp_torch \
  --predictor_type module \
  --predictor_module experiments.ctm_adapter_sony:CTMRepoPredictor \
  --predictor_kwargs_json '{"ctm_repo":"/path/to/CAMP/third_party/ctm","checkpoint":"/path/to/CAMP/.cache/ctm/ctm_imagenet64_ema999.pt"}' \
  --predictor_image_size 64 \
  --t_star 40 \
  --self_correct_k 0 \
  --replacement_mode hard \
  --lightweight_mode \
  --save_reference_every 10 \
  --mix 0.35 \
  --min_clean_conf 0.05
```

## 6. 先做的最小实验建议

1. 基于当前冒烟结果，优先使用：`self_correct_k=0`，`replacement_mode=hard`。
2. 先扫参数：`t_star in {20,30,40}`。
3. 如要验证 loop 价值，再单独扫：`self_correct_k=1` 且 `t_bridge in {5,10,15}`。
4. 如细节损失大，再对照 `replacement_mode=fused` 并扫描：`hf_preserve in {0.3,0.45,0.6}`。
5. 优先看 `recover_rate_on_attacked` 与 `clean_vs_purified.SSIM` 的折中，不要只看单指标。
