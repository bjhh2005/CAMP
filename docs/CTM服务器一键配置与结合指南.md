# CTM 服务器一键配置与 WGCP 结合指南

## 1. 适用范围

这份文档只针对 Linux 服务器场景（建议在服务器端完成全部 CTM 相关操作）。

## 2. 一次性配置（setup）

在仓库根目录执行：

```bash
bash scripts/setup_ctm_server.sh
```

脚本会完成：

1. 更新 `camp` conda 环境（`environment.yml`）。
2. 拉取或更新 `third_party/ctm`。
3. 解析并检查 checkpoint 路径。
4. 安装 CTM 运行时依赖和仓库 requirements（可通过环境变量关闭）。
5. 生成 `configs/ctm_server_config.json`。

手动 checkpoint（推荐，避免在线下载波动）：

```bash
CTM_CHECKPOINT_PATH=/home/you/models/ema_0.999_049000.pt \
DOWNLOAD_CKPT=0 \
bash scripts/setup_ctm_server.sh
```

checkpoint 选择优先级：

1. `CTM_CHECKPOINT_PATH`
2. `$CTM_CACHE_DIR/ctm_imagenet64_ema999.pt`
3. `$CTM_CACHE_DIR/ema_0.999_049000.pt`
4. 仅 `DOWNLOAD_CKPT=1` 时尝试下载

## 3. 推荐主流程（非消融）

```bash
GLOB_PATTERN="*.JPEG" \
MAX_IMAGES=100 \
LIGHTWEIGHT_MODE=1 \
SAVE_REFERENCE_EVERY=10 \
bash scripts/run_wgcp_ctm_server.sh ./data/imagenet_real ./outputs/wgcp_eval_ctm
```

启用 Patch-WGCP（推荐首轮）：

```bash
PATCH_MODE=1 \
PATCH_SIZE=64 \
PATCH_STRIDE=32 \
PATCH_BATCH_SIZE=64 \
PATCH_LOWFREQ_ALPHA=0.1 \
PATCH_LL_SOURCE=hat \
GLOB_PATTERN="*.JPEG" MAX_IMAGES=100 LIGHTWEIGHT_MODE=1 SAVE_REFERENCE_EVERY=10 \
bash scripts/run_wgcp_ctm_server.sh ./data/imagenet_real ./outputs/wgcp_eval_ctm_patch
```

最小正式对照（推荐，替代 A0-A5 全跑）：

```bash
MAX_IMAGES=100 \
SAVE_DETAIL_EVERY=10 \
GLOB_PATTERN="*.JPEG" \
bash scripts/run_patch_minimal_ctm_server.sh ./data/imagenet_real ./outputs/wgcp_patch_min
```

该脚本固定运行 3 组：

1. `A5_global`
2. `A5_patch_main`（默认 `patch_stride=32`, `patch_lowfreq_alpha=0.1`）
3. `A5_patch_alpha0`（仅把 `patch_lowfreq_alpha` 设为 `0.0`）

当前脚本默认关键参数：

- `self_correct_k=0`
- `replacement_mode=hard`
- `t_star=40`
- `min_clean_conf=0.05`

## 4. 一键消融（A0-A5）

```bash
GLOB_PATTERN="*.JPEG" MAX_IMAGES=100 SAVE_DETAIL_EVERY=10 \
bash scripts/run_ablation_ctm_server.sh ./data/imagenet_real ./outputs/wgcp_ablation_ctm
```

详细矩阵与解读见：`docs/消融实验方案.md`。

## 5. tmux 长任务技巧（强烈建议）

### 5.1 创建与恢复会话

```bash
# 新建会话
tmux new -s camp_eval

# 退出但不断任务：Ctrl+b 然后按 d

# 查看会话
tmux ls

# 重新连接
tmux attach -t camp_eval

# 结束会话
tmux kill-session -t camp_eval
```

### 5.2 分屏与监控

```bash
# 水平分屏：Ctrl+b 然后按 "
# 垂直分屏：Ctrl+b 然后按 %

# 面板切换：Ctrl+b 然后按方向键
```

推荐布局：

1. 左侧跑实验命令。
2. 右侧 `watch -n 1 nvidia-smi` 监控显存和利用率。

### 5.3 保存日志（排错必备）

```bash
mkdir -p logs
bash scripts/run_wgcp_ctm_server.sh ./data/imagenet_real ./outputs/wgcp_eval_ctm \
  2>&1 | tee logs/wgcp_eval_$(date +%F_%H%M).log
```

## 6. 常见问题

1. 提示缺少配置文件 `configs/ctm_server_config.json`：先执行 `bash scripts/setup_ctm_server.sh`。
2. 首次下载分类器权重很慢：提前设置 `TORCH_CACHE_DIR` 指向高速磁盘。
3. `flash_attn/xformers` 导入问题：本项目已有 fallback shim，通常不需要手工安装 `flash_attn`。
4. 混合分辨率输入是否安全：流程支持混合分辨率；若做严格公平对比，仍建议在评估脚本中显式 `--resize H W`。
