import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    from wgcp_attack_eval import (
        classify,
        fgsm_attack,
        load_classifier,
        parse_loop_steps,
        pgd_attack,
    )
    from wgcp_predictor import build_alpha_bars, build_predictor, run_predictor_with_features
    from wgcp_purify import purify_adv_tensor
    from wgcp_utils import (
        archive_run_record,
        blend_heatmap_on_image,
        build_run_meta,
        ensure_dir,
        list_images,
        make_triplet_panel,
        normalize_spatial_map,
        pil_to_tensor,
        save_tensor_image,
        set_seed,
        spatial_map_to_rgb,
    )
except ImportError:
    from experiments.wgcp_attack_eval import (
        classify,
        fgsm_attack,
        load_classifier,
        parse_loop_steps,
        pgd_attack,
    )
    from experiments.wgcp_predictor import build_alpha_bars, build_predictor, run_predictor_with_features
    from experiments.wgcp_purify import purify_adv_tensor
    from experiments.wgcp_utils import (
        archive_run_record,
        blend_heatmap_on_image,
        build_run_meta,
        ensure_dir,
        list_images,
        make_triplet_panel,
        normalize_spatial_map,
        pil_to_tensor,
        save_tensor_image,
        set_seed,
        spatial_map_to_rgb,
    )


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2: probe CTM intermediate feature heatmaps on clean/adv/hat samples.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory of clean images")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/ctm_feature_probe"))
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--glob", type=str, default="*.png")
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument(
        "--archive_dir",
        type=Path,
        default=Path.home() / ".camp_runs" / "CAMP",
        help="Directory for archived run records (default outside repo).",
    )
    parser.add_argument("--archive_tag", type=str, default="", help="Optional tag appended to archive filename.")
    parser.add_argument("--disable_archive", action="store_true", help="Disable extra archived run record.")

    parser.add_argument("--classifier", type=str, default="resnet50", choices=["resnet18", "resnet50", "mobilenet_v3_large"])
    parser.add_argument("--clf_resize_short", type=int, default=256, help="Classifier pre-resize short side")
    parser.add_argument("--clf_crop_size", type=int, default=224, help="Classifier center-crop size")
    parser.add_argument("--min_clean_conf", type=float, default=0.05, help="Skip samples with low clean confidence")
    parser.add_argument(
        "--weights_cache_dir",
        type=Path,
        default=Path.home() / ".cache" / "camp_torch",
        help="Torch weight cache directory (outside repo by default)",
    )

    parser.add_argument("--attack", type=str, default="pgd", choices=["fgsm", "pgd"])
    parser.add_argument("--eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument("--pgd_alpha", type=float, default=2.0 / 255.0)
    parser.add_argument("--only_attack_success", action="store_true", help="Save only samples where adv prediction differs from clean")

    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--num_diffusion_steps", type=int, default=1000)
    parser.add_argument("--t_star", type=int, default=40)
    parser.add_argument("--self_correct_k", type=int, default=0)
    parser.add_argument("--self_correct_steps", type=str, default="", help="Comma-separated time steps, e.g. 60,40,20")
    parser.add_argument(
        "--t_bridge",
        type=int,
        default=-1,
        help="Bridge step t1 for interleaved projection. If <=0, auto uses round(t_star*0.25).",
    )
    parser.add_argument(
        "--predictor_type",
        type=str,
        default="module",
        choices=["gaussian", "module"],
        help="Predictor backend: built-in gaussian debug model or external module",
    )
    parser.add_argument(
        "--predictor_module",
        type=str,
        default="",
        help="When predictor_type=module, use format 'package.module:ClassName'",
    )
    parser.add_argument(
        "--predictor_config_path",
        type=Path,
        default=None,
        help="Optional CAMP CTM config json; fills predictor_module / ctm_repo / checkpoint defaults.",
    )
    parser.add_argument(
        "--predictor_kwargs_json",
        type=str,
        default="{}",
        help="JSON string for predictor constructor kwargs when predictor_type=module",
    )
    parser.add_argument(
        "--predictor_image_size",
        type=int,
        default=64,
        help="If >0, resize input to this square size before predictor, then resize back",
    )

    parser.add_argument(
        "--replacement_mode",
        type=str,
        default="adaptive_ms",
        choices=["hard", "fused", "adaptive_ms", "adaptive_ms_guided", "adaptive_ms_edge", "adaptive_ms_modmax"],
    )
    parser.add_argument("--hf_preserve", type=float, default=0.35)
    parser.add_argument("--hf_shrink", type=float, default=0.6)
    parser.add_argument("--ablation_ll_source", type=str, default="orig", choices=["orig", "hat"])
    parser.add_argument("--ablation_hard_hf_source", type=str, default="pred", choices=["pred", "orig"])
    parser.add_argument("--ms_levels", type=int, default=3)
    parser.add_argument("--ms_gamma_levels", type=str, default="1.6,1.2,0.9")
    parser.add_argument("--ms_w_min", type=float, default=0.05)
    parser.add_argument("--ms_w_max", type=float, default=0.95)
    parser.add_argument("--ms_ll_alpha", type=float, default=0.08)
    parser.add_argument("--ms_eps", type=float, default=1e-6)
    parser.add_argument("--ms_ll_gate_tau", type=float, default=0.75)
    parser.add_argument("--ms_ll_gate_gain", type=float, default=4.0)
    parser.add_argument("--ms_hf_pred_levels", type=str, default="0.20,0.12,0.08")
    parser.add_argument("--ms_hf_gate_tau", type=float, default=0.5)
    parser.add_argument("--ms_hf_gate_gain", type=float, default=4.0)
    parser.add_argument("--ms_edge_eta_levels", type=str, default="0.5,0.5,0.5")
    parser.add_argument("--ms_edge_eta_hh_levels", type=str, default="0.3,0.3,0.3")
    parser.add_argument("--ms_edge_sigma_divisor", type=float, default=30.0)
    parser.add_argument("--ms_edge_alpha_min", type=float, default=0.1)
    parser.add_argument("--ms_edge_channel_agg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ms_modmax_threshold", type=float, default=0.15)
    parser.add_argument("--ms_modmax_boost_levels", type=str, default="0.10,0.08,0.05")

    parser.add_argument("--ctm_feature_layer", type=str, default="", help="Exact module name, substring, or leaf index for CTM feature hook")
    parser.add_argument("--ctm_feature_sources", type=str, default="adv,hat", help="Comma-separated: clean,adv,xt,hat")
    parser.add_argument("--ctm_feature_reduce", type=str, default="mean_abs", choices=["mean_abs", "l2", "grad"])
    parser.add_argument("--ctm_feature_t_index", type=int, default=-1, help="Feature probe t-index; <=0 falls back to t_star")
    parser.add_argument("--ctm_feature_overlay_alpha", type=float, default=0.45)
    parser.add_argument("--ctm_feature_leaf_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--list_feature_modules", action="store_true", help="Dump CTM module names and exit")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def setup_weight_cache(cache_dir: Path) -> None:
    cache_dir = cache_dir.expanduser().resolve()
    ensure_dir(cache_dir)
    os.environ["TORCH_HOME"] = str(cache_dir)


def apply_predictor_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.predictor_config_path is None:
        return args

    cfg_path = args.predictor_config_path.expanduser().resolve()
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))

    if not args.predictor_module:
        args.predictor_module = str(payload.get("predictor_module", args.predictor_module))
    if int(args.predictor_image_size) <= 0 and payload.get("predictor_image_size") is not None:
        args.predictor_image_size = int(payload["predictor_image_size"])

    predictor_kwargs = json.loads(args.predictor_kwargs_json)
    if payload.get("ctm_repo") and "ctm_repo" not in predictor_kwargs:
        predictor_kwargs["ctm_repo"] = payload["ctm_repo"]
    if payload.get("checkpoint") and "checkpoint" not in predictor_kwargs:
        predictor_kwargs["checkpoint"] = payload["checkpoint"]
    predictor_kwargs.setdefault("predictor_image_size", int(args.predictor_image_size))
    args.predictor_kwargs_json = json.dumps(predictor_kwargs)
    return args


def parse_feature_sources(text: str) -> List[str]:
    allowed = {"clean", "adv", "xt", "hat"}
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if not parts:
        raise ValueError("ctm_feature_sources must not be empty")
    for part in parts:
        if part not in allowed:
            raise ValueError(f"Unsupported ctm_feature_source: {part}")
    return parts


def reduce_feature_map(feature: Tensor, mode: str) -> Tensor:
    if feature.ndim != 4:
        raise ValueError(f"Expected feature map [B,C,H,W], got {tuple(feature.shape)}")
    if mode == "mean_abs":
        return feature.abs().mean(dim=1, keepdim=True)
    if mode == "l2":
        return torch.sqrt(feature.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12))
    if mode == "grad":
        base = feature.abs().mean(dim=1, keepdim=True)
        return gradient_magnitude_map(base)
    raise ValueError(f"Unknown reduce mode: {mode}")


def gradient_magnitude_map(x: Tensor) -> Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError("gradient_magnitude_map expects [B,C,H,W] or [C,H,W]")
    if x.shape[1] > 1:
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    else:
        gray = x
    dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-12)


def open_image_as_tensor(path: Path, resize_hw: Tuple[int, int] | None) -> Tensor:
    img = Image.open(path).convert("RGB")
    if resize_hw is not None:
        img = img.resize((resize_hw[1], resize_hw[0]), Image.BICUBIC)
    return pil_to_tensor(img).unsqueeze(0)


def save_feature_views(
    sample_dir: Path,
    source_name: str,
    source_image: Tensor,
    feature_map: Tensor,
    gradient_map: Tensor,
    overlay_alpha: float,
) -> Dict[str, float]:
    feature_norm = normalize_spatial_map(feature_map[0])
    gradient_norm = normalize_spatial_map(gradient_map[0])
    feature_rgb = spatial_map_to_rgb(feature_norm)
    gradient_rgb = spatial_map_to_rgb(gradient_norm)
    feature_overlay = blend_heatmap_on_image(source_image[0], feature_norm, alpha=overlay_alpha)
    gradient_overlay = blend_heatmap_on_image(source_image[0], gradient_norm, alpha=overlay_alpha)

    save_tensor_image(feature_rgb, sample_dir / f"20_{source_name}_ctm_heatmap.png")
    save_tensor_image(feature_overlay, sample_dir / f"21_{source_name}_ctm_overlay.png")
    save_tensor_image(gradient_rgb, sample_dir / f"22_{source_name}_grad_heatmap.png")
    save_tensor_image(gradient_overlay, sample_dir / f"23_{source_name}_grad_overlay.png")

    panel = make_triplet_panel([source_image[0], feature_overlay, gradient_overlay])
    panel.save(sample_dir / f"24_{source_name}_probe_panel.png")

    return {
        "feature_mean": float(feature_norm.mean().item()),
        "feature_max": float(feature_norm.max().item()),
        "gradient_mean": float(gradient_norm.mean().item()),
        "gradient_max": float(gradient_norm.max().item()),
    }


def main() -> None:
    args = parse_args()
    args = apply_predictor_config(args)
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    setup_weight_cache(args.weights_cache_dir)

    resize_hw = tuple(args.resize) if args.resize is not None else None
    feature_sources = parse_feature_sources(args.ctm_feature_sources)
    feature_t_index = args.ctm_feature_t_index if args.ctm_feature_t_index > 0 else args.t_star
    loop_steps = parse_loop_steps(args.self_correct_k, args.self_correct_steps, args.t_star, args.t_bridge)

    device = torch.device(args.device)
    predictor = build_predictor(args, device)

    if args.list_feature_modules:
        if not hasattr(predictor, "list_named_modules"):
            raise AttributeError("Predictor does not expose list_named_modules(...)")
        modules = predictor.list_named_modules(leaf_only=args.ctm_feature_leaf_only)
        out_path = args.output_dir / "ctm_modules.json"
        out_path.write_text(json.dumps({"modules": modules}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(modules)} CTM module names to {out_path}")
        return

    classifier = load_classifier(args.classifier, device)
    alpha_bars = build_alpha_bars(args.num_diffusion_steps).to(device)

    image_paths = list_images(args.input_dir, args.glob, args.max_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir} matching {args.glob}")

    summary_rows: List[Dict[str, object]] = []
    resolved_feature_layer = args.ctm_feature_layer.strip()

    for sample_idx, image_path in enumerate(tqdm(image_paths, desc="CTM feature probe"), start=1):
        x_clean = open_image_as_tensor(image_path, resize_hw=resize_hw).to(device)
        pred_clean, conf_clean = classify(
            classifier,
            x_clean,
            resize_short=args.clf_resize_short,
            crop_size=args.clf_crop_size,
        )
        if float(conf_clean.item()) < args.min_clean_conf:
            continue

        if args.attack == "fgsm":
            x_adv = fgsm_attack(
                classifier,
                x_clean,
                pred_clean,
                eps=args.eps,
                resize_short=args.clf_resize_short,
                crop_size=args.clf_crop_size,
            )
        else:
            x_adv = pgd_attack(
                classifier,
                x_clean,
                pred_clean,
                eps=args.eps,
                alpha=args.pgd_alpha,
                steps=args.pgd_steps,
                resize_short=args.clf_resize_short,
                crop_size=args.clf_crop_size,
            )

        pred_adv, conf_adv = classify(
            classifier,
            x_adv,
            resize_short=args.clf_resize_short,
            crop_size=args.clf_crop_size,
        )
        attack_success = bool(pred_adv.item() != pred_clean.item())
        if args.only_attack_success and not attack_success:
            continue

        x_corrected, x_final, trace, infer_ms = purify_adv_tensor(
            x_adv=x_adv,
            alpha_bars=alpha_bars,
            predictor=predictor,
            wavelet=args.wavelet,
            t_star=args.t_star,
            loop_steps=loop_steps,
            hf_preserve=args.hf_preserve,
            hf_shrink=args.hf_shrink,
            replacement_mode=args.replacement_mode,
            predictor_image_size=args.predictor_image_size,
            ablation_ll_source=args.ablation_ll_source,
            ablation_hard_hf_source=args.ablation_hard_hf_source,
            ms_levels=args.ms_levels,
            ms_gamma_levels=args.ms_gamma_levels,
            ms_w_min=args.ms_w_min,
            ms_w_max=args.ms_w_max,
            ms_ll_alpha=args.ms_ll_alpha,
            ms_eps=args.ms_eps,
            ms_ll_gate_tau=args.ms_ll_gate_tau,
            ms_ll_gate_gain=args.ms_ll_gate_gain,
            ms_hf_pred_levels=args.ms_hf_pred_levels,
            ms_hf_gate_tau=args.ms_hf_gate_tau,
            ms_hf_gate_gain=args.ms_hf_gate_gain,
            ms_edge_eta_levels=args.ms_edge_eta_levels,
            ms_edge_eta_hh_levels=args.ms_edge_eta_hh_levels,
            ms_edge_sigma_divisor=args.ms_edge_sigma_divisor,
            ms_edge_alpha_min=args.ms_edge_alpha_min,
            ms_edge_channel_agg=args.ms_edge_channel_agg,
            ms_modmax_threshold=args.ms_modmax_threshold,
            ms_modmax_boost_levels=args.ms_modmax_boost_levels,
        )
        pred_final, conf_final = classify(
            classifier,
            x_final,
            resize_short=args.clf_resize_short,
            crop_size=args.clf_crop_size,
        )
        x_hat = trace["x0_hat"]
        x_t = trace["x_t"]

        sample_name = image_path.stem
        sample_dir = args.output_dir / f"{sample_idx:03d}_{sample_name}"
        ensure_dir(sample_dir)

        save_tensor_image(x_clean[0], sample_dir / "00_x_clean.png")
        save_tensor_image(x_adv[0], sample_dir / "01_x_adv.png")
        save_tensor_image(x_t[0], sample_dir / f"02_x_t{args.t_star}.png")
        save_tensor_image(x_hat[0], sample_dir / "03_x_hat.png")
        save_tensor_image(x_corrected[0], sample_dir / "04_x_corrected.png")
        save_tensor_image(x_final[0], sample_dir / "05_x_final.png")

        main_panel = make_triplet_panel([x_clean[0], x_adv[0], x_hat[0], x_final[0]])
        main_panel.save(sample_dir / "10_overview_panel.png")

        source_map = {
            "clean": x_clean,
            "adv": x_adv,
            "xt": x_t,
            "hat": x_hat,
        }
        feature_meta: Dict[str, object] = {}
        for source_name in feature_sources:
            source_image = source_map[source_name]
            feature_info = run_predictor_with_features(
                predictor=predictor,
                x_t=source_image,
                t_index=feature_t_index,
                predictor_image_size=args.predictor_image_size,
                feature_layer=resolved_feature_layer,
                feature_leaf_only=args.ctm_feature_leaf_only,
            )
            if not resolved_feature_layer:
                resolved_feature_layer = str(feature_info["layer"])
            feature_map = reduce_feature_map(feature_info["feature"], args.ctm_feature_reduce)
            gradient_map = gradient_magnitude_map(source_image)
            stats = save_feature_views(
                sample_dir=sample_dir,
                source_name=source_name,
                source_image=source_image,
                feature_map=feature_map,
                gradient_map=gradient_map,
                overlay_alpha=args.ctm_feature_overlay_alpha,
            )
            feature_meta[source_name] = {
                "layer": str(feature_info["layer"]),
                "feature_shape": list(feature_info["feature_small"].shape),
                "input_small_shape": list(feature_info["input_small"].shape),
                "stats": stats,
            }

        summary_rows.append(
            {
                "sample_index": sample_idx,
                "image_path": str(image_path),
                "output_dir": str(sample_dir),
                "attack_success": attack_success,
                "predictor_infer_ms": float(infer_ms),
                "feature_layer": resolved_feature_layer,
                "predictions": {
                    "clean": {"pred": int(pred_clean.item()), "conf": float(conf_clean.item())},
                    "adv": {"pred": int(pred_adv.item()), "conf": float(conf_adv.item())},
                    "purified": {"pred": int(pred_final.item()), "conf": float(conf_final.item())},
                },
                "feature_probe": feature_meta,
            }
        )

    summary = {
        "run_meta": build_run_meta(args, script_path=Path(__file__).resolve()),
        "resolved_feature_layer": resolved_feature_layer,
        "num_saved_samples": len(summary_rows),
        "samples": summary_rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    archive_run_record(args, summary, script_name="ctm_feature_probe")
    print(f"Saved E2 feature probe summary to {summary_path}")


if __name__ == "__main__":
    main()
