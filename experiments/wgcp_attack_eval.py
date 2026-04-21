import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    from wgcp_small_scale import (
        GaussianCTMPredictor,
        build_alpha_bars,
        coeff_to_vis,
        compute_metrics,
        ensure_dir,
        list_images,
        make_triplet_panel,
        pil_to_tensor,
        purify_adv_tensor,
        save_tensor_image,
        set_seed,
    )
except ImportError:
    from experiments.wgcp_small_scale import (
        GaussianCTMPredictor,
        build_alpha_bars,
        coeff_to_vis,
        compute_metrics,
        ensure_dir,
        list_images,
        make_triplet_panel,
        pil_to_tensor,
        purify_adv_tensor,
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
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"))

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
    parser.add_argument("--t_star", type=int, default=80)
    parser.add_argument("--self_correct_k", type=int, default=0)
    parser.add_argument(
        "--self_correct_steps",
        type=str,
        default="",
        help="Comma-separated time steps, e.g. 60,40,20",
    )
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--mix", type=float, default=0.8)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_loop_steps(self_correct_k: int, self_correct_steps: str, t_star: int) -> List[int]:
    if self_correct_k <= 0:
        return []
    if self_correct_steps.strip():
        steps = [int(x.strip()) for x in self_correct_steps.split(",") if x.strip()]
        steps = steps[:self_correct_k]
        if len(steps) < self_correct_k:
            raise ValueError("self_correct_steps count is smaller than self_correct_k")
        return steps

    top = max(2, t_star)
    return np.linspace(top, 2, self_correct_k + 1, dtype=int).tolist()[:-1]


def setup_weight_cache(cache_dir: Path) -> None:
    cache_dir = cache_dir.expanduser().resolve()
    ensure_dir(cache_dir)
    os.environ["TORCH_HOME"] = str(cache_dir)


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


@torch.no_grad()
def classify(model: torch.nn.Module, x: Tensor) -> Tuple[Tensor, Tensor]:
    logits = model(imagenet_normalize(x))
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf


def fgsm_attack(model: torch.nn.Module, x: Tensor, y_ref: Tensor, eps: float) -> Tensor:
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(imagenet_normalize(x_adv))
    loss = F.cross_entropy(logits, y_ref)
    model.zero_grad(set_to_none=True)
    loss.backward()
    x_adv = x_adv + eps * x_adv.grad.sign()
    return x_adv.detach().clamp(0.0, 1.0)


def pgd_attack(model: torch.nn.Module, x: Tensor, y_ref: Tensor, eps: float, alpha: float, steps: int) -> Tensor:
    x_orig = x.detach().clone()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.detach().clone().requires_grad_(True)
        logits = model(imagenet_normalize(x_adv))
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
    hf_hat = trace["hf_hat"]

    save_tensor_image(x_t[0].clamp(0.0, 1.0), sample_dir / f"03_x_t{t_star}.png")
    save_tensor_image(x0_hat[0], sample_dir / "04_x0_hat.png")
    save_tensor_image(ll_hat.clamp(0.0, 1.0), sample_dir / "05_LL_hat.png")
    save_tensor_image(coeff_to_vis(hf_hat["HL"]), sample_dir / "05_HL_hat_vis.png")
    save_tensor_image(coeff_to_vis(hf_hat["LH"]), sample_dir / "05_LH_hat_vis.png")
    save_tensor_image(coeff_to_vis(hf_hat["HH"]), sample_dir / "05_HH_hat_vis.png")

    save_tensor_image(x_corrected[0], sample_dir / "06_x_corrected.png")

    for i, record in enumerate(trace["loop_records"], start=1):
        t_step = int(record["t_step"])
        save_tensor_image(record["x_tk"][0].clamp(0.0, 1.0), sample_dir / f"07_loop{i}_x_t{t_step}.png")
        save_tensor_image(record["x_hat_k"][0], sample_dir / f"07_loop{i}_x_hat.png")
        save_tensor_image(record["x_corrected_k"][0], sample_dir / f"07_loop{i}_x_corrected.png")

    save_tensor_image(x_final[0], sample_dir / "08_x_final.png")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    if args.t_star <= 0 or args.t_star >= args.num_diffusion_steps:
        raise ValueError("t_star must be in (0, num_diffusion_steps)")

    loop_steps = parse_loop_steps(args.self_correct_k, args.self_correct_steps, args.t_star)
    for step in loop_steps:
        if step <= 0 or step >= args.num_diffusion_steps:
            raise ValueError("all self-correct steps must be in (0, num_diffusion_steps)")

    setup_weight_cache(args.weights_cache_dir)

    device = torch.device(args.device)
    alpha_bars = build_alpha_bars(args.num_diffusion_steps).to(device)
    predictor = GaussianCTMPredictor(kernel_size=args.kernel_size, sigma=args.sigma, mix=args.mix)
    classifier = load_classifier(args.classifier, device)

    image_paths = list_images(args.input_dir, args.glob, args.max_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    per_sample: List[Dict[str, object]] = []

    for image_path in tqdm(image_paths, desc="WGCP attack+eval"):
        sample_dir = args.output_dir / image_path.stem
        ensure_dir(sample_dir)

        clean_img = Image.open(image_path).convert("RGB")
        if args.resize is not None:
            h, w = args.resize
            clean_img = clean_img.resize((w, h), Image.BICUBIC)

        x_clean = pil_to_tensor(clean_img).unsqueeze(0).to(device)
        save_tensor_image(x_clean[0], sample_dir / "00_x_clean.png")

        pred_clean, conf_clean = classify(classifier, x_clean)
        y_ref = pred_clean.detach()

        if args.attack == "fgsm":
            x_adv = fgsm_attack(classifier, x_clean, y_ref, eps=args.eps)
        else:
            x_adv = pgd_attack(classifier, x_clean, y_ref, eps=args.eps, alpha=args.pgd_alpha, steps=args.pgd_steps)

        save_tensor_image(x_adv[0], sample_dir / "01_x_adv.png")
        pred_adv, conf_adv = classify(classifier, x_adv)

        x_corrected, x_final, trace, infer_ms = purify_adv_tensor(
            x_adv=x_adv,
            alpha_bars=alpha_bars,
            predictor=predictor,
            wavelet=args.wavelet,
            t_star=args.t_star,
            loop_steps=loop_steps,
        )
        save_purify_trace(sample_dir, args.t_star, trace, x_corrected, x_final)

        pred_final, conf_final = classify(classifier, x_final)

        panel_clean_adv_final = make_triplet_panel([x_clean[0], x_adv[0], x_final[0]])
        panel_clean_adv_final.save(sample_dir / "09_compare_clean_adv_final.png")

        panel_adv_corrected_final = make_triplet_panel([x_adv[0], x_corrected[0], x_final[0]])
        panel_adv_corrected_final.save(sample_dir / "10_compare_adv_corrected_final.png")

        metrics_clean_adv = compute_metrics(x_clean[0], x_adv[0])
        metrics_clean_purified = compute_metrics(x_clean[0], x_final[0])

        attack_success = int(pred_adv.item() != pred_clean.item())
        recover_success = int(pred_final.item() == pred_clean.item())

        record = {
            "image": str(image_path),
            "classifier": args.classifier,
            "attack": {
                "type": args.attack,
                "eps": args.eps,
                "pgd_steps": args.pgd_steps if args.attack == "pgd" else None,
                "pgd_alpha": args.pgd_alpha if args.attack == "pgd" else None,
            },
            "wgcp": {
                "wavelet": args.wavelet,
                "t_star": args.t_star,
                "self_correct_steps": loop_steps,
                "NFE": 1 + len(loop_steps),
                "single_step_infer_ms": float(infer_ms),
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

        with open(sample_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        per_sample.append(record)

    attacked = sum(int(item["pseudo_label_eval"]["attack_success"]) for item in per_sample)
    recovered = sum(
        int(item["pseudo_label_eval"]["recover_to_clean_pred"]) for item in per_sample if item["pseudo_label_eval"]["attack_success"]
    )
    overall_consistent = sum(int(item["pseudo_label_eval"]["recover_to_clean_pred"]) for item in per_sample)

    aggregate = {
        "num_samples": len(per_sample),
        "num_attack_success": attacked,
        "attack_success_rate": attacked / max(1, len(per_sample)),
        "recover_rate_on_attacked": recovered / max(1, attacked),
        "clean_pred_consistency_rate": overall_consistent / max(1, len(per_sample)),
        "note": "This script uses clean-image predicted label as pseudo-label (no ground-truth required).",
    }

    summary = {
        "aggregate": aggregate,
        "samples": per_sample,
    }

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. Results saved to: {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
