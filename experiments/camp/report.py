from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _get(obj: Dict[str, Any], dotted: str, default: Any = "") -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(item) for item in value)
    return str(value)


def _md_path(path_text: str, report_path: Path) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        return os.path.relpath(path.resolve(), start=report_path.parent.resolve()).replace("\\", "/")
    except Exception:
        return path_text.replace("\\", "/")


def _select_visual_samples(samples: List[Dict[str, Any]], max_images: int) -> List[Dict[str, Any]]:
    with_triplets = [s for s in samples if _get(s, "artifacts.triplet", "")]
    if not with_triplets:
        return []
    recovered = [s for s in with_triplets if s.get("attack_success") and s.get("recover_to_clean_pred")]
    failed = [s for s in with_triplets if s.get("attack_success") and not s.get("recover_to_clean_pred")]
    unchanged = [s for s in with_triplets if not s.get("attack_success")]
    selected: List[Dict[str, Any]] = []
    for bucket in (recovered, failed, unchanged, with_triplets):
        for sample in bucket:
            if sample not in selected:
                selected.append(sample)
            if len(selected) >= max_images:
                return selected
    return selected[:max_images]


def build_report(run_dir: Path, output_path: Path, max_images: int) -> None:
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "resolved_config.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"resolved_config.json not found: {config_path}")

    summary = _load_json(summary_path)
    config = _load_json(config_path)
    aggregate = summary.get("aggregate", {})
    samples = summary.get("samples", [])
    visual_samples = _select_visual_samples(samples, max_images=max_images)

    lines: List[str] = []
    lines.append(f"# CAMP-CM 实验分析: {summary.get('experiment_name', run_dir.name)}")
    lines.append("")
    lines.append("## 1. 关键指标")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    for key in (
        "num_samples",
        "clean_accuracy_on_labeled",
        "attack_success_rate",
        "adv_same_as_clean_rate",
        "recover_rate_on_attacked",
        "purified_same_as_clean_rate",
    ):
        lines.append(f"| `{key}` | {_fmt(aggregate.get(key, ''))} |")

    lines.append("")
    lines.append("## 2. 关键参数")
    lines.append("")
    param_rows = [
        ("dataset.name", _get(config, "dataset.name")),
        ("dataset.root", _get(config, "dataset.root")),
        ("dataset.split", _get(config, "dataset.split")),
        ("dataset.max_samples", _get(config, "dataset.max_samples")),
        ("classifier.name", _get(config, "classifier.name")),
        ("classifier.checkpoint", _get(config, "classifier.kwargs.checkpoint")),
        ("attack.name", _get(config, "attack.name")),
        ("attack.eps", _get(config, "attack.eps")),
        ("attack.steps", _get(config, "attack.steps")),
        ("attack.step_size", _get(config, "attack.step_size")),
        ("backend", _get(config, "purification.backend")),
        ("model_module", _get(config, "purification.model_module")),
        ("cm_repo", _get(config, "purification.model_kwargs.ctm_repo")),
        ("openai_repo", _get(config, "purification.model_kwargs.repo")),
        ("cm_checkpoint", _get(config, "purification.model_kwargs.checkpoint")),
        ("sampling_steps", _get(config, "purification.schedule.sampling_steps")),
        ("iN", _get(config, "purification.schedule.iN")),
        ("gamma", _get(config, "purification.schedule.gamma")),
        ("eta", _get(config, "purification.schedule.eta")),
        ("wavelet_noise.enabled", _get(config, "purification.wavelet_noise.enabled")),
        ("wavelet_noise.wavelet", _get(config, "purification.wavelet_noise.wavelet")),
        ("wavelet_noise.levels", _get(config, "purification.wavelet_noise.levels")),
        ("wavelet_noise.gains", _get(config, "purification.wavelet_noise.gains")),
        ("bp.enabled", _get(config, "purification.bp.enabled")),
        ("bp.mu", _get(config, "purification.bp.mu")),
    ]
    lines.append("| 参数 | 值 |")
    lines.append("|---|---|")
    for key, value in param_rows:
        lines.append(f"| `{key}` | {_fmt(value)} |")

    lines.append("")
    lines.append("## 3. 可视化样本")
    lines.append("")
    if visual_samples:
        for idx, sample in enumerate(visual_samples, start=1):
            triplet = _md_path(_get(sample, "artifacts.triplet", ""), output_path)
            lines.append(
                f"### 样本 {idx}: clean={sample.get('clean_pred')} adv={sample.get('adv_pred')} purified={sample.get('purified_pred')}"
            )
            lines.append("")
            lines.append(
                f"- attack_success: `{sample.get('attack_success')}`, recover_to_clean_pred: `{sample.get('recover_to_clean_pred')}`"
            )
            lines.append(f"- confidence: clean `{_fmt(sample.get('clean_conf'))}`, adv `{_fmt(sample.get('adv_conf'))}`, purified `{_fmt(sample.get('purified_conf'))}`")
            lines.append("")
            lines.append(f"![clean_adv_purified]({triplet})")
            lines.append("")
    else:
        lines.append("本次运行没有可视化图片。重新运行时加 `--save_images`，或使用脚本里的 `SAVE_IMAGES=1`。")
        lines.append("")

    attacked = [s for s in samples if s.get("attack_success")]
    failed = [s for s in attacked if not s.get("recover_to_clean_pred")]
    lines.append("## 4. 失败样本概览")
    lines.append("")
    lines.append("| path | clean | adv | purified | clean_conf | adv_conf | purified_conf |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for sample in failed[:20]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(sample.get("path", "")),
                    str(sample.get("clean_pred", "")),
                    str(sample.get("adv_pred", "")),
                    str(sample.get("purified_pred", "")),
                    _fmt(sample.get("clean_conf", "")),
                    _fmt(sample.get("adv_conf", "")),
                    _fmt(sample.get("purified_conf", "")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## 5. 下一步建议")
    lines.append("")
    lines.append("- 先比较 baseline 与 wavelet_noise 的 `recover_rate_on_attacked`。")
    lines.append("- 若攻击成功率过低，增大 `attack.steps` 或检查分类器是否正确加载。")
    lines.append("- 若净化后 clean 一致性下降明显，降低 `iN` 或关闭/减小小波增益。")
    lines.append("- 若目录 checkpoint 自动选择不确定，把 YAML 里的 checkpoint 改成具体文件。")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CAMP-CM markdown analysis report")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max_images", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (args.run_dir / "analysis.md")
    build_report(run_dir=args.run_dir, output_path=output, max_images=args.max_images)
    print(f"Report saved to: {output}")


if __name__ == "__main__":
    main()
