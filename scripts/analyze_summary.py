import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze CAMP summary.json files and export comparison-friendly CSV tables."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more summary.json files or experiment output directories containing summary.json.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional labels matching --inputs order. Defaults to parent directory names.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/summary_analysis"),
        help="Directory used for exported CSV files.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of interesting sample rows to include in console summaries.",
    )
    return parser.parse_args()


def resolve_summary_path(path: Path) -> Path:
    if path.is_dir():
        summary_path = path / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"summary.json not found under directory: {path}")
        return summary_path
    if path.is_file():
        return path
    raise FileNotFoundError(f"Input does not exist: {path}")


def infer_label(summary_path: Path) -> str:
    if summary_path.name == "summary.json":
        return summary_path.parent.name
    return summary_path.stem


def load_summary(summary_path: Path) -> Dict[str, object]:
    with open(summary_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_summary_kind(summary_obj: Dict[str, object]) -> str:
    if "aggregate" in summary_obj:
        return "attack_eval"
    return "small_scale"


def safe_get(obj, path: Sequence[object], default=None):
    current = obj
    for key in path:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
            continue
        if isinstance(current, list):
            if not isinstance(key, int) or key < 0 or key >= len(current):
                return default
            current = current[key]
            continue
        return default
    return current


def float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def bool_to_int(value) -> Optional[int]:
    if value is None:
        return None
    return int(bool(value))


def round_or_blank(value: Optional[float], ndigits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{ndigits}f}"


def flatten_attack_eval_sample(sample: Dict[str, object], label: str) -> Dict[str, object]:
    image_metrics = safe_get(sample, ["image_metrics"], default={}) or {}
    edge_stats = safe_get(image_metrics, ["edge_stats"], default={}) or {}
    clean_adv = safe_get(image_metrics, ["clean_vs_adv"], default={}) or {}
    clean_purified = safe_get(image_metrics, ["clean_vs_purified"], default={}) or {}
    pseudo = safe_get(sample, ["pseudo_label_eval"], default={}) or {}
    wgcp = safe_get(sample, ["wgcp"], default={}) or {}
    attack = safe_get(sample, ["attack"], default={}) or {}
    artifacts = safe_get(sample, ["artifacts"], default={}) or {}

    return {
        "experiment": label,
        "image": sample.get("image"),
        "skipped": bool(sample.get("skipped", False)),
        "skip_reason": sample.get("skip_reason", ""),
        "detail_saved": artifacts.get("detail_saved"),
        "sample_dir": artifacts.get("sample_dir"),
        "classifier": sample.get("classifier"),
        "attack_type": attack.get("type"),
        "attack_eps": attack.get("eps"),
        "pgd_steps": attack.get("pgd_steps"),
        "pgd_alpha": attack.get("pgd_alpha"),
        "replacement_mode": wgcp.get("replacement_mode"),
        "wavelet": wgcp.get("wavelet"),
        "t_star": wgcp.get("t_star"),
        "nfe": wgcp.get("NFE"),
        "infer_ms": wgcp.get("single_step_infer_ms"),
        "clean_pred": pseudo.get("clean_pred"),
        "adv_pred": pseudo.get("adv_pred"),
        "purified_pred": pseudo.get("purified_pred"),
        "clean_conf": pseudo.get("clean_conf"),
        "adv_conf": pseudo.get("adv_conf"),
        "purified_conf": pseudo.get("purified_conf"),
        "attack_success": bool_to_int(pseudo.get("attack_success")),
        "recover_to_clean_pred": bool_to_int(pseudo.get("recover_to_clean_pred")),
        "ssim_clean_vs_adv": clean_adv.get("SSIM"),
        "psnr_clean_vs_adv": clean_adv.get("PSNR"),
        "ssim_clean_vs_purified": clean_purified.get("SSIM"),
        "psnr_clean_vs_purified": clean_purified.get("PSNR"),
        "edge_mean_gradient_clean": safe_get(edge_stats, ["clean", "mean_gradient"]),
        "edge_mean_gradient_adv": safe_get(edge_stats, ["adv", "mean_gradient"]),
        "edge_mean_gradient_purified": safe_get(edge_stats, ["purified", "mean_gradient"]),
        "edge_lap_var_clean": safe_get(edge_stats, ["clean", "laplacian_variance"]),
        "edge_lap_var_adv": safe_get(edge_stats, ["adv", "laplacian_variance"]),
        "edge_lap_var_purified": safe_get(edge_stats, ["purified", "laplacian_variance"]),
    }


def summarize_small_scale_samples(samples: List[Dict[str, object]]) -> Dict[str, Optional[float]]:
    metrics_rows = [sample.get("metrics", {}) for sample in samples if isinstance(sample.get("metrics"), dict)]
    keys = sorted({key for row in metrics_rows for key in row.keys() if isinstance(row.get(key), (int, float))})
    summary: Dict[str, Optional[float]] = {}
    for key in keys:
        values = [float(row[key]) for row in metrics_rows if isinstance(row.get(key), (int, float))]
        summary[f"mean_{key}"] = mean(values) if values else None
    return summary


def flatten_small_scale_sample(sample: Dict[str, object], label: str) -> Dict[str, object]:
    row = {
        "experiment": label,
        "image": sample.get("image"),
        "replacement_mode": sample.get("replacement_mode"),
        "wavelet": sample.get("wavelet"),
        "t_star": sample.get("t_star"),
        "predictor_type": sample.get("predictor_type"),
        "predictor_image_size": sample.get("predictor_image_size"),
        "patch_mode": sample.get("patch_mode"),
        "self_correct_steps": json.dumps(sample.get("self_correct_steps", []), ensure_ascii=False),
    }
    metrics = sample.get("metrics", {})
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            row[f"metric_{key}"] = value
    return row


def build_overview_row(label: str, summary_path: Path, summary_obj: Dict[str, object]) -> Dict[str, object]:
    kind = detect_summary_kind(summary_obj)
    samples = summary_obj.get("samples", [])
    run_meta = summary_obj.get("run_meta", {})
    row: Dict[str, object] = {
        "label": label,
        "kind": kind,
        "summary_path": str(summary_path),
        "run_id": safe_get(run_meta, ["run_id"]),
        "timestamp_local": safe_get(run_meta, ["timestamp_local"]),
        "git_commit": safe_get(run_meta, ["git_commit"]),
        "num_samples": len(samples) if isinstance(samples, list) else None,
    }
    if kind == "attack_eval":
        aggregate = summary_obj.get("aggregate", {})
        if isinstance(aggregate, dict):
            row.update(aggregate)
        row["replacement_mode"] = safe_get(summary_obj, ["samples", 0, "wgcp", "replacement_mode"])
        row["classifier"] = safe_get(summary_obj, ["samples", 0, "classifier"])
        row["attack_type"] = safe_get(summary_obj, ["samples", 0, "attack", "type"])
    else:
        samples_list = samples if isinstance(samples, list) else []
        if samples_list:
            row["replacement_mode"] = samples_list[0].get("replacement_mode")
            row["predictor_type"] = samples_list[0].get("predictor_type")
        row.update(summarize_small_scale_samples(samples_list))
    return row


def get_indexable(summary_obj: Dict[str, object], label: str) -> List[Dict[str, object]]:
    kind = detect_summary_kind(summary_obj)
    samples = summary_obj.get("samples", [])
    if not isinstance(samples, list):
        return []
    if kind == "attack_eval":
        return [flatten_attack_eval_sample(sample, label) for sample in samples]
    return [flatten_small_scale_sample(sample, label) for sample in samples]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_attack_eval_comparison(
    named_summaries: List[Tuple[str, Path, Dict[str, object]]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    indexed: Dict[str, Dict[str, Dict[str, object]]] = {}
    for label, _, summary_obj in named_summaries:
        indexed[label] = {}
        for row in get_indexable(summary_obj, label):
            image = row.get("image")
            if image:
                indexed[label][str(image)] = row

    all_images = sorted({image for by_image in indexed.values() for image in by_image.keys()})
    comparison_rows: List[Dict[str, object]] = []
    disagreement_rows: List[Dict[str, object]] = []
    for image in all_images:
        row: Dict[str, object] = {"image": image}
        recovery_values: List[Tuple[str, Optional[int]]] = []
        for label, _, _ in named_summaries:
            sample = indexed[label].get(image, {})
            row[f"{label}__skipped"] = sample.get("skipped")
            row[f"{label}__attack_success"] = sample.get("attack_success")
            row[f"{label}__recover_to_clean_pred"] = sample.get("recover_to_clean_pred")
            row[f"{label}__purified_conf"] = sample.get("purified_conf")
            row[f"{label}__ssim_clean_vs_purified"] = sample.get("ssim_clean_vs_purified")
            row[f"{label}__psnr_clean_vs_purified"] = sample.get("psnr_clean_vs_purified")
            row[f"{label}__edge_mean_gradient_purified"] = sample.get("edge_mean_gradient_purified")
            row[f"{label}__edge_lap_var_purified"] = sample.get("edge_lap_var_purified")
            recovery_values.append((label, sample.get("recover_to_clean_pred")))
        comparison_rows.append(row)

        observed = {value for _, value in recovery_values if value is not None}
        if len(observed) > 1:
            disagreement = {"image": image}
            for label, value in recovery_values:
                disagreement[f"{label}__recover_to_clean_pred"] = value
            disagreement_rows.append(disagreement)
    return comparison_rows, disagreement_rows


def print_overview(rows: List[Dict[str, object]]) -> None:
    print("Overview")
    for row in rows:
        kind = row.get("kind", "")
        label = row.get("label", "")
        if kind == "attack_eval":
            print(
                "  "
                f"{label}: recover_rate_on_attacked={round_or_blank(float_or_none(row.get('recover_rate_on_attacked')), 4)}, "
                f"clean_pred_consistency_rate={round_or_blank(float_or_none(row.get('clean_pred_consistency_rate')), 4)}, "
                f"attack_success_rate={round_or_blank(float_or_none(row.get('attack_success_rate')), 4)}, "
                f"mean_gradient_purified={round_or_blank(float_or_none(row.get('mean_gradient_purified')), 6)}, "
                f"laplacian_variance_purified={round_or_blank(float_or_none(row.get('laplacian_variance_purified')), 6)}"
            )
        else:
            print(
                "  "
                f"{label}: mean_SSIM={round_or_blank(float_or_none(row.get('mean_SSIM')), 4)}, "
                f"mean_PSNR={round_or_blank(float_or_none(row.get('mean_PSNR')), 4)}, "
                f"mean_NFE={round_or_blank(float_or_none(row.get('mean_NFE')), 3)}, "
                f"mean_single_step_infer_ms={round_or_blank(float_or_none(row.get('mean_single_step_infer_ms')), 3)}"
            )


def print_attack_eval_highlights(
    named_summaries: List[Tuple[str, Path, Dict[str, object]]],
    top_k: int,
) -> None:
    if not named_summaries:
        return
    if any(detect_summary_kind(summary_obj) != "attack_eval" for _, _, summary_obj in named_summaries):
        return

    print("\nRecovery Highlights")
    for label, _, summary_obj in named_summaries:
        rows = [row for row in get_indexable(summary_obj, label) if not row.get("skipped")]
        recovered = [row for row in rows if row.get("recover_to_clean_pred") == 1]
        failed = [row for row in rows if row.get("attack_success") == 1 and row.get("recover_to_clean_pred") == 0]

        recovered_sorted = sorted(
            recovered,
            key=lambda row: (
                -(row.get("purified_conf") or -math.inf),
                -(row.get("ssim_clean_vs_purified") or -math.inf),
            ),
        )[:top_k]
        failed_sorted = sorted(
            failed,
            key=lambda row: (
                row.get("purified_conf") or math.inf,
                row.get("ssim_clean_vs_purified") or math.inf,
            ),
        )[:top_k]

        print(f"  {label} top recovered samples:")
        for row in recovered_sorted:
            print(
                "    "
                f"{Path(str(row.get('image'))).name}: purified_conf={round_or_blank(float_or_none(row.get('purified_conf')), 4)}, "
                f"ssim={round_or_blank(float_or_none(row.get('ssim_clean_vs_purified')), 4)}, "
                f"edge_grad={round_or_blank(float_or_none(row.get('edge_mean_gradient_purified')), 6)}"
            )
        print(f"  {label} top failed attacked samples:")
        for row in failed_sorted:
            print(
                "    "
                f"{Path(str(row.get('image'))).name}: purified_conf={round_or_blank(float_or_none(row.get('purified_conf')), 4)}, "
                f"ssim={round_or_blank(float_or_none(row.get('ssim_clean_vs_purified')), 4)}, "
                f"edge_grad={round_or_blank(float_or_none(row.get('edge_mean_gradient_purified')), 6)}"
            )


def main() -> None:
    args = parse_args()

    resolved_paths = [resolve_summary_path(path) for path in args.inputs]
    if args.labels and len(args.labels) != len(resolved_paths):
        raise ValueError("--labels count must match --inputs count when provided")
    labels = args.labels if args.labels else [infer_label(path) for path in resolved_paths]

    named_summaries: List[Tuple[str, Path, Dict[str, object]]] = []
    for label, summary_path in zip(labels, resolved_paths):
        named_summaries.append((label, summary_path, load_summary(summary_path)))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    overview_rows = [build_overview_row(label, path, summary_obj) for label, path, summary_obj in named_summaries]
    write_csv(args.output_dir / "overview.csv", overview_rows)

    for label, _, summary_obj in named_summaries:
        sample_rows = get_indexable(summary_obj, label)
        write_csv(args.output_dir / f"{label}__samples.csv", sample_rows)

    if named_summaries and all(detect_summary_kind(summary_obj) == "attack_eval" for _, _, summary_obj in named_summaries):
        comparison_rows, disagreement_rows = build_attack_eval_comparison(named_summaries)
        write_csv(args.output_dir / "comparison_attack_eval.csv", comparison_rows)
        write_csv(args.output_dir / "comparison_attack_eval_recovery_disagreements.csv", disagreement_rows)

    print_overview(overview_rows)
    print_attack_eval_highlights(named_summaries, top_k=max(1, args.top_k))

    print("\nExports")
    print(f"  overview.csv -> {args.output_dir / 'overview.csv'}")
    for label, _, _ in named_summaries:
        print(f"  {label}__samples.csv -> {args.output_dir / f'{label}__samples.csv'}")
    if named_summaries and all(detect_summary_kind(summary_obj) == "attack_eval" for _, _, summary_obj in named_summaries):
        print(f"  comparison_attack_eval.csv -> {args.output_dir / 'comparison_attack_eval.csv'}")
        print(
            "  comparison_attack_eval_recovery_disagreements.csv -> "
            f"{args.output_dir / 'comparison_attack_eval_recovery_disagreements.csv'}"
        )


if __name__ == "__main__":
    main()
