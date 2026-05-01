# CAMP 文档导航（2026-05-01）

当前文档只保留 CAMP-CM 主线，避免旧 WGCP/AWDD 文档干扰新实验。

1. 小数据集即插即用：`docs/CAMP_CM小数据集即插即用指南.md`

## 当前工程边界

- CAMP 自有代码统一放在 `experiments/` 下；新主线放在 `experiments/camp/`。
- `CM4IR/` 与 `FreqPure/` 只作为本地参考代码，不作为 CAMP 主体代码维护，也不直接修改。
- 数据集、模型权重、输出结果不进仓库；服务器上通过配置文件填真实路径。
- Windows 本地以编码、配置整理、静态检查为主；Linux 服务器负责完整数据和权重实验。

## 已清理内容

- 旧 WGCP/AWDD/CTM 文档已移除，当前只维护 CAMP-CM 小数据集和后续高分辨率路线。
- 2026-05-01 新增 `experiments/camp/`，用于承载 CM 少步对抗净化、小波噪声注入、小波 BP。

## 你应该先看哪份

- 先看 `CAMP_CM小数据集即插即用指南.md`。
- 本地先跑 `small_debug.yaml`，服务器再复制 `small_cm_template.yaml` 填真实路径。
