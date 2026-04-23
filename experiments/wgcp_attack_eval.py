import argparse
import getpass
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    from wgcp_small_scale import (
        build_predictor,
        build_alpha_bars,
        coeff_to_vis,
        compute_metrics,
        ensure_dir,
        list_images,
        make_triplet_panel,
        pil_to_tensor,
        purify_adv_tensor,
        run_predictor,
        save_tensor_image,
        set_seed,
    )
except ImportError:
    from experiments.wgcp_small_scale import (
        build_predictor,
        build_alpha_bars,
        coeff_to_vis,
        compute_metrics,
        ensure_dir,
        list_images,
        make_triplet_panel,
        pil_to_tensor,
        purify_adv_tensor,
        run_predictor,
        save_tensor_image,
        set_seed,
    )


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate adversarial samples, run WGCP purification, and evaluate classifier behavior."
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory of clean images")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/wgcp_eval"))
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--glob", type=str, default="*.png")
    parser.add_argument(
        "--lightweight_mode",
        action="store_true",
        help="Save only summary and sparse reference panels (no per-sample folders/metrics).",
    )
    parser.add_argument(
        "--save_reference_every",
        type=int,
        default=10,
        help="In lightweight mode, save one clean/adv/purified panel every N samples (<=0 disables reference images).",
    )
    parser.add_argument(
        "--save_detail_every",
        type=int,
        default=1,
        help=(
            "When not in lightweight mode, save full per-sample detail artifacts every N samples "
            "(<=0 disables detail artifacts but summary still keeps all samples)."
        ),
    )
    parser.add_argument(
        "--reference_dir",
        type=Path,
        default=None,
        help="Optional reference image directory. Default: <output_dir>/references when lightweight_mode is on.",
    )
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--clf_resize_short", type=int, default=256, help="Classifier pre-resize short side")
    parser.add_argument("--clf_crop_size", type=int, default=224, help="Classifier center-crop size")
    parser.add_argument(
        "--min_clean_conf",
        type=float,
        default=0.05,
        help="Skip pseudo-label evaluation for samples with clean confidence below this threshold",
    )
    parser.add_argument(
        "--archive_dir",
        type=Path,
        default=Path.home() / ".camp_runs" / "CAMP",
        help="Directory for archived run records (default outside repo).",
    )
    parser.add_argument("--archive_tag", type=str, default="", help="Optional tag appended to archive filename.")
    parser.add_argument("--disable_archive", action="store_true", help="Disable extra archived run record.")

    parser.add_argument("--classifier", type=str, default="resnet50", choices=["resnet18", "resnet50", "mobilenet_v3_large"])
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

    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--num_diffusion_steps", type=int, default=1000)
    parser.add_argument("--t_star", type=int, default=40)
    parser.add_argument("--self_correct_k", type=int, default=1)
    parser.add_argument(
        "--self_correct_steps",
        type=str,
        default="",
        help="Comma-separated time steps, e.g. 60,40,20",
    )
    parser.add_argument(
        "--t_bridge",
        type=int,
        default=-1,
        help="Bridge step t1 for interleaved projection. If <=0, auto uses round(t_star*0.25).",
    )
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--mix", type=float, default=0.45)
    parser.add_argument(
        "--predictor_type",
        type=str,
        default="gaussian",
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
        "--predictor_kwargs_json",
        type=str,
        default="{}",
        help="JSON string for predictor constructor kwargs when predictor_type=module",
    )
    parser.add_argument(
        "--predictor_image_size",
        type=int,
        default=0,
        help="If >0, resize input to this square size before predictor, then resize back",
    )
    parser.add_argument(
        "--hf_preserve",
        type=float,
        default=0.35,
        help="How much original HF to preserve in replacement (0=full rewrite, 1=full keep)",
    )
    parser.add_argument(
        "--hf_shrink",
        type=float,
        default=0.6,
        help="Soft-threshold factor for original HF before blending (relative to robust sigma)",
    )
    parser.add_argument(
        "--replacement_mode",
        type=str,
        default="hard",
        choices=["hard", "fused"],
        help="HF replacement strategy: paper-default hard replacement or fused replacement",
    )
    parser.add_argument(
        "--ablation_ll_source",
        type=str,
        default="orig",
        choices=["orig", "hat"],
        help="Ablation only: LL source for IDWT anchor. 'orig'=paper default, 'hat'=predictor LL.",
    )
    parser.add_argument(
        "--ablation_hard_hf_source",
        type=str,
        default="pred",
        choices=["pred", "orig"],
        help="Ablation only under replacement_mode=hard: HF source. 'pred'=paper default, 'orig'=keep original HF.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_loop_steps(self_correct_k: int, self_correct_steps: str, t_star: int, t_bridge: int) -> List[int]:
    if self_correct_k <= 0:
        return []
    if self_correct_steps.strip():
        steps = [int(x.strip()) for x in self_correct_steps.split(",") if x.strip()]
        steps = steps[:self_correct_k]
        if len(steps) < self_correct_k:
            raise ValueError("self_correct_steps count is smaller than self_correct_k")
        return [int(s) for s in steps]

    bridge = t_bridge if t_bridge > 0 else max(2, int(round(t_star * 0.25)))
    if self_correct_k == 1:
        return [int(bridge)]
    tail = np.linspace(bridge, 2, self_correct_k, dtype=int).tolist()
    out: List[int] = []
    for s in tail:
        if not out or out[-1] != s:
            out.append(int(s))
    return out


def setup_weight_cache(cache_dir: Path) -> None:
    cache_dir = cache_dir.expanduser().resolve()
    ensure_dir(cache_dir)
    os.environ["TORCH_HOME"] = str(cache_dir)


def _safe_git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return out or None
    except Exception:
        return None


def _jsonable_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def build_run_meta(args: argparse.Namespace) -> Dict[str, object]:
    now_local = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)
    return {
        "run_id": now_local.strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "timestamp_local": now_local.isoformat(),
        "timestamp_utc": now_utc.isoformat(),
        "script": str(Path(__file__).resolve()),
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "git_commit": _safe_git_commit(),
        "args": _jsonable_args(args),
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
        },
    }


def archive_run_record(args: argparse.Namespace, summary_obj: Dict[str, object], script_name: str) -> Path | None:
    if args.disable_archive:
        return None

    archive_root = args.archive_dir.expanduser().resolve()
    archive_dir = archive_root / script_name
    ensure_dir(archive_dir)

    run_id = str(summary_obj["run_meta"]["run_id"])
    tag = f"_{args.archive_tag.strip()}" if args.archive_tag.strip() else ""
    archive_path = archive_dir / f"{run_id}{tag}.json"
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)
    return archive_path


def load_classifier(name: str, device: torch.device) -> torch.nn.Module:
    from torchvision.models import (
        MobileNet_V3_Large_Weights,
        ResNet18_Weights,
        ResNet50_Weights,
        mobilenet_v3_large,
        resnet18,
        resnet50,
    )

    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif name == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Unsupported classifier: {name}")

    model.eval().to(device)
    return model


def imagenet_normalize(x: Tensor) -> Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def imagenet_preprocess(x: Tensor, resize_short: int = 256, crop_size: int = 224) -> Tensor:
    if x.ndim != 4:
        raise ValueError("Expected x with shape [B, C, H, W]")

    out = x
    h, w = out.shape[-2], out.shape[-1]

    if resize_short > 0:
        short = min(h, w)
        scale = resize_short / float(short)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        out = F.interpolate(out, size=(new_h, new_w), mode="bilinear", align_corners=False)
        h, w = out.shape[-2], out.shape[-1]

    if crop_size > 0:
        if h < crop_size or w < crop_size:
            out = F.interpolate(out, size=(max(h, crop_size), max(w, crop_size)), mode="bilinear", align_corners=False)
            h, w = out.shape[-2], out.shape[-1]
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        out = out[:, :, top : top + crop_size, left : left + crop_size]

    return out


def classifier_logits(
    model: torch.nn.Module,
    x: Tensor,
    resize_short: int,
    crop_size: int,
) -> Tensor:
    x_in = imagenet_preprocess(x, resize_short=resize_short, crop_size=crop_size)
    return model(imagenet_normalize(x_in))


@torch.no_grad()
def classify(
    model: torch.nn.Module,
    x: Tensor,
    resize_short: int,
    crop_size: int,
) -> Tuple[Tensor, Tensor]:
    logits = classifier_logits(model, x, resize_short=resize_short, crop_size=crop_size)
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf


def fgsm_attack(
    model: torch.nn.Module,
    x: Tensor,
    y_ref: Tensor,
    eps: float,
    resize_short: int,
    crop_size: int,
) -> Tensor:
    x_adv = x.detach().clone().requires_grad_(True)
    logits = classifier_logits(model, x_adv, resize_short=resize_short, crop_size=crop_size)
    loss = F.cross_entropy(logits, y_ref)
    model.zero_grad(set_to_none=True)
    loss.backward()
    x_adv = x_adv + eps * x_adv.grad.sign()
    return x_adv.detach().clamp(0.0, 1.0)


def pgd_attack(
    model: torch.nn.Module,
    x: Tensor,
    y_ref: Tensor,
    eps: float,
    alpha: float,
    steps: int,
    resize_short: int,
    crop_size: int,
) -> Tensor:
    x_orig = x.detach().clone()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.detach().clone().requires_grad_(True)
        logits = classifier_logits(model, x_adv, resize_short=resize_short, crop_size=crop_size)
        loss = F.cross_entropy(logits, y_ref)
        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
            x_adv = (x_orig + delta).clamp(0.0, 1.0)

    return x_adv.detach()


def save_purify_trace(sample_dir: Path, t_star: int, trace: Dict[str, object], x_corrected: Tensor, x_final: Tensor) -> None:
    ll_orig = trace["ll_orig"]
    hf_orig = trace["hf_orig"]
    save_tensor_image(ll_orig.clamp(0.0, 1.0), sample_dir / "02_LL_orig.png")
    save_tensor_image(coeff_to_vis(hf_orig["HL"]), sample_dir / "02_HL_orig_vis.png")
    save_tensor_image(coeff_to_vis(hf_orig["LH"]), sample_dir / "02_LH_orig_vis.png")
    save_tensor_image(coeff_to_vis(hf_orig["HH"]), sample_dir / "02_HH_orig_vis.png")

    x_t = trace["x_t"]
    x0_hat = trace["x0_hat"]
    ll_hat = trace["ll_hat"]
    ll_anchor = trace["ll_anchor"]
    hf_hat = trace["hf_hat"]
    hf_selected = trace["hf_selected"]

    save_tensor_image(x_t[0].clamp(0.0, 1.0), sample_dir / f"03_x_t{t_star}.png")
    save_tensor_image(x0_hat[0], sample_dir / "04_x0_hat.png")
    save_tensor_image(ll_hat.clamp(0.0, 1.0), sample_dir / "05_LL_hat.png")
    save_tensor_image(ll_anchor.clamp(0.0, 1.0), sample_dir / "05_LL_anchor.png")
    save_tensor_image(coeff_to_vis(hf_hat["HL"]), sample_dir / "05_HL_hat_vis.png")
    save_tensor_image(coeff_to_vis(hf_hat["LH"]), sample_dir / "05_LH_hat_vis.png")
    save_tensor_image(coeff_to_vis(hf_hat["HH"]), sample_dir / "05_HH_hat_vis.png")
    save_tensor_image(coeff_to_vis(hf_selected["HL"]), sample_dir / "05_HL_selected_vis.png")
    save_tensor_image(coeff_to_vis(hf_selected["LH"]), sample_dir / "05_LH_selected_vis.png")
    save_tensor_image(coeff_to_vis(hf_selected["HH"]), sample_dir / "05_HH_selected_vis.png")

    save_tensor_image(x_corrected[0], sample_dir / "06_x_assembled.png")
    save_tensor_image(x_corrected[0], sample_dir / "06_x_corrected.png")

    for i, record in enumerate(trace["loop_records"], start=1):
        t_step = int(record["t_step"])
        save_tensor_image(record["x_tk"][0].clamp(0.0, 1.0), sample_dir / f"07_loop{i}_x_t{t_step}.png")
        save_tensor_image(record["x_hat_k"][0], sample_dir / f"07_loop{i}_x_hat.png")
        save_tensor_image(record["x_corrected_k"][0], sample_dir / f"07_loop{i}_x_corrected.png")

    save_tensor_image(x_final[0], sample_dir / "08_x_final.png")


def should_save_reference(index_1_based: int, every: int) -> bool:
    return every > 0 and index_1_based % every == 0


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    if args.t_star <= 0 or args.t_star >= args.num_diffusion_steps:
        raise ValueError("t_star must be in (0, num_diffusion_steps)")

    loop_steps = parse_loop_steps(args.self_correct_k, args.self_correct_steps, args.t_star, args.t_bridge)
    for step in loop_steps:
        if step <= 0 or step >= args.num_diffusion_steps:
            raise ValueError("all self-correct steps must be in (0, num_diffusion_steps)")

    setup_weight_cache(args.weights_cache_dir)

    device = torch.device(args.device)
    alpha_bars = build_alpha_bars(args.num_diffusion_steps).to(device)
    predictor = build_predictor(args, device=device)
    classifier = load_classifier(args.classifier, device)
    with torch.no_grad():
        warmup = torch.rand(1, 3, 256, 256, device=device)
        _ = run_predictor(predictor, warmup, args.t_star, predictor_image_size=args.predictor_image_size)
        _ = classify(classifier, warmup, resize_short=args.clf_resize_short, crop_size=args.clf_crop_size)

    image_paths = list_images(args.input_dir, args.glob, args.max_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    per_sample: List[Dict[str, object]] = []
    valid_sample_count = 0
    skipped_low_conf_count = 0
    saved_reference_count = 0
    saved_detail_count = 0

    reference_dir: Path | None = None
    if args.lightweight_mode and args.save_reference_every > 0:
        reference_dir = (args.reference_dir if args.reference_dir is not None else (args.output_dir / "references")).resolve()
        ensure_dir(reference_dir)

    for sample_idx, image_path in enumerate(tqdm(image_paths, desc="WGCP attack+eval"), start=1):
        save_reference = args.lightweight_mode and should_save_reference(sample_idx, args.save_reference_every)
        save_detail = (not args.lightweight_mode) and should_save_reference(sample_idx, args.save_detail_every)
        sample_dir = args.output_dir / image_path.stem
        if save_detail:
            ensure_dir(sample_dir)
            saved_detail_count += 1

        clean_img = Image.open(image_path).convert("RGB")
        if args.resize is not None:
            h, w = args.resize
            clean_img = clean_img.resize((w, h), Image.BICUBIC)

        x_clean = pil_to_tensor(clean_img).unsqueeze(0).to(device)
        if save_detail:
            save_tensor_image(x_clean[0], sample_dir / "00_x_clean.png")

        pred_clean, conf_clean = classify(
            classifier,
            x_clean,
            resize_short=args.clf_resize_short,
            crop_size=args.clf_crop_size,
        )
        if hasattr(predictor, "set_class_label"):
            predictor.set_class_label(int(pred_clean.item()))
        if conf_clean.item() < args.min_clean_conf:
            skipped_low_conf_count += 1
            record = {
                "image": str(image_path),
                "classifier": args.classifier,
                "skipped": True,
                "skip_reason": f"clean_conf<{args.min_clean_conf}",
                "artifacts": {
                    "detail_saved": bool(save_detail),
                    "sample_dir": str(sample_dir) if save_detail else None,
                },
                "pseudo_label_eval": {
                    "clean_pred": int(pred_clean.item()),
                    "clean_conf": float(conf_clean.item()),
                },
            }
            if save_detail:
                with open(sample_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)
            per_sample.append(record)
            continue

        valid_sample_count += 1
        y_ref = pred_clean.detach()

        if args.attack == "fgsm":
            x_adv = fgsm_attack(
                classifier,
                x_clean,
                y_ref,
                eps=args.eps,
                resize_short=args.clf_resize_short,
                crop_size=args.clf_crop_size,
            )
        else:
            x_adv = pgd_attack(
                classifier,
                x_clean,
                y_ref,
                eps=args.eps,
                alpha=args.pgd_alpha,
                steps=args.pgd_steps,
                resize_short=args.clf_resize_short,
                crop_size=args.clf_crop_size,
            )

        if save_detail:
            save_tensor_image(x_adv[0], sample_dir / "01_x_adv.png")
        pred_adv, conf_adv = classify(
            classifier,
            x_adv,
            resize_short=args.clf_resize_short,
            crop_size=args.clf_crop_size,
        )

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
        )
        if save_detail:
            save_purify_trace(sample_dir, args.t_star, trace, x_corrected, x_final)

        pred_final, conf_final = classify(
            classifier,
            x_final,
            resize_short=args.clf_resize_short,
            crop_size=args.clf_crop_size,
        )

        if save_detail:
            panel_clean_adv_final = make_triplet_panel([x_clean[0], x_adv[0], x_final[0]])
            panel_clean_adv_final.save(sample_dir / "09_compare_clean_adv_final.png")

            panel_adv_corrected_final = make_triplet_panel([x_adv[0], x_corrected[0], x_final[0]])
            panel_adv_corrected_final.save(sample_dir / "10_compare_adv_corrected_final.png")
        elif save_reference and reference_dir is not None:
            panel_clean_adv_final = make_triplet_panel([x_clean[0], x_adv[0], x_final[0]])
            ref_name = f"{sample_idx:04d}_{image_path.stem}_clean_adv_final.png"
            panel_clean_adv_final.save(reference_dir / ref_name)
            saved_reference_count += 1

        metrics_clean_adv = compute_metrics(x_clean[0], x_adv[0])
        metrics_clean_purified = compute_metrics(x_clean[0], x_final[0])

        attack_success = int(pred_adv.item() != pred_clean.item())
        recover_success = int(pred_final.item() == pred_clean.item())

        record = {
            "image": str(image_path),
            "classifier": args.classifier,
            "artifacts": {
                "detail_saved": bool(save_detail),
                "sample_dir": str(sample_dir) if save_detail else None,
            },
            "attack": {
                "type": args.attack,
                "eps": args.eps,
                "pgd_steps": args.pgd_steps if args.attack == "pgd" else None,
                "pgd_alpha": args.pgd_alpha if args.attack == "pgd" else None,
            },
            "wgcp": {
                "wavelet": args.wavelet,
                "t_star": args.t_star,
                "t_bridge": args.t_bridge if args.t_bridge > 0 else max(2, int(round(args.t_star * 0.25))),
                "self_correct_steps": loop_steps,
                "NFE": 1 + len(loop_steps),
                "single_step_infer_ms": float(infer_ms),
                "predictor_type": args.predictor_type,
                "predictor_image_size": args.predictor_image_size,
                "replacement_mode": args.replacement_mode,
                "ablation_ll_source": args.ablation_ll_source,
                "ablation_hard_hf_source": args.ablation_hard_hf_source,
                "hf_preserve": args.hf_preserve,
                "hf_shrink": args.hf_shrink,
            },
            "pseudo_label_eval": {
                "clean_pred": int(pred_clean.item()),
                "adv_pred": int(pred_adv.item()),
                "purified_pred": int(pred_final.item()),
                "clean_conf": float(conf_clean.item()),
                "adv_conf": float(conf_adv.item()),
                "purified_conf": float(conf_final.item()),
                "attack_success": bool(attack_success),
                "recover_to_clean_pred": bool(recover_success),
            },
            "image_metrics": {
                "clean_vs_adv": metrics_clean_adv,
                "clean_vs_purified": metrics_clean_purified,
            },
        }

        if save_detail:
            with open(sample_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

        per_sample.append(record)

    valid_samples = [item for item in per_sample if not item.get("skipped", False)]
    attacked = sum(int(item["pseudo_label_eval"]["attack_success"]) for item in valid_samples)
    recovered = sum(
        int(item["pseudo_label_eval"]["recover_to_clean_pred"])
        for item in valid_samples
        if item["pseudo_label_eval"]["attack_success"]
    )
    overall_consistent = sum(int(item["pseudo_label_eval"]["recover_to_clean_pred"]) for item in valid_samples)

    aggregate = {
        "num_samples": len(per_sample),
        "num_valid_samples": valid_sample_count,
        "num_skipped_low_conf": skipped_low_conf_count,
        "min_clean_conf": args.min_clean_conf,
        "num_attack_success": attacked,
        "attack_success_rate": attacked / max(1, valid_sample_count),
        "recover_rate_on_attacked": recovered / max(1, attacked),
        "clean_pred_consistency_rate": overall_consistent / max(1, valid_sample_count),
        "lightweight_mode": bool(args.lightweight_mode),
        "save_detail_every": int(args.save_detail_every),
        "saved_detail_samples": int(saved_detail_count),
        "save_reference_every": int(args.save_reference_every),
        "saved_reference_images": int(saved_reference_count),
        "reference_dir": str(reference_dir) if reference_dir is not None else None,
        "note": "This script uses clean-image predicted label as pseudo-label (no ground-truth required). Low-confidence clean samples can be skipped.",
    }

    summary = {
        "run_meta": build_run_meta(args),
        "aggregate": aggregate,
        "samples": per_sample,
    }

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    archive_path = archive_run_record(args, summary, script_name="wgcp_attack_eval")

    print(f"Done. Results saved to: {args.output_dir}")
    print(f"Summary: {summary_path}")
    if archive_path is not None:
        print(f"Archived summary: {archive_path}")


if __name__ == "__main__":
    main()
