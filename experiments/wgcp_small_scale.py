import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pywt
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wavelet-Guided Consistency Purification (small-scale debug runner)"
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory of adversarial images")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/wgcp_small"))
    parser.add_argument("--clean_dir", type=Path, default=None, help="Optional clean-image directory for metrics")
    parser.add_argument("--max_images", type=int, default=4)
    parser.add_argument("--glob", type=str, default="*.png")
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
    parser.add_argument("--mix", type=float, default=0.8, help="x_t -> denoised blend ratio")
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(input_dir: Path, pattern: str, max_images: int) -> List[Path]:
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
        seen = []
        for ext in exts:
            seen.extend(sorted(input_dir.glob(ext)))
        paths = sorted(set(seen))
    return paths[:max_images]


def pil_to_tensor(img: Image.Image) -> Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def tensor_to_pil(tensor: Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def save_tensor_image(tensor: Tensor, path: Path) -> None:
    tensor_to_pil(tensor).save(path)


def coeff_to_vis(coeff: Tensor) -> Tensor:
    max_abs = torch.max(torch.abs(coeff))
    if max_abs < 1e-8:
        return torch.zeros_like(coeff) + 0.5
    return (coeff / (2 * max_abs) + 0.5).clamp(0.0, 1.0)


def make_gaussian_kernel(kernel_size: int, sigma: float, channels: int, device: str) -> Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel


def gaussian_blur_tensor(x: Tensor, kernel_size: int, sigma: float) -> Tensor:
    b, c, _, _ = x.shape
    kernel = make_gaussian_kernel(kernel_size, sigma, c, str(x.device))
    padding = kernel_size // 2
    return F.conv2d(x, kernel, padding=padding, groups=c)


class GaussianCTMPredictor:
    """Debug predictor used as CTM placeholder for mechanism verification."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, mix: float = 0.8):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.mix = float(np.clip(mix, 0.0, 1.0))

    @torch.no_grad()
    def __call__(self, x_t: Tensor, t_index: int) -> Tensor:
        denoised = gaussian_blur_tensor(x_t, self.kernel_size, self.sigma)
        x0_hat = torch.lerp(x_t, denoised, self.mix)
        return x0_hat.clamp(0.0, 1.0)


def build_alpha_bars(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> Tensor:
    betas = torch.linspace(beta_start, beta_end, steps=num_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alpha_bars


def add_noise(x: Tensor, alpha_bar_t: float) -> Tensor:
    eps = torch.randn_like(x)
    return math.sqrt(alpha_bar_t) * x + math.sqrt(1.0 - alpha_bar_t) * eps


def dwt2_rgb_tensor(x_chw: Tensor, wavelet: str) -> Tuple[Tensor, Dict[str, Tensor]]:
    x_np = x_chw.detach().cpu().numpy()
    ll_list, hl_list, lh_list, hh_list = [], [], [], []
    for c in range(x_np.shape[0]):
        ll, (ch, cv, cd) = pywt.dwt2(x_np[c], wavelet=wavelet, mode="periodization")
        ll_list.append(ll)
        # pywt returns (cH, cV, cD). We map to HL/LH/HH naming for paper notation.
        hl_list.append(ch)
        lh_list.append(cv)
        hh_list.append(cd)
    ll_t = torch.from_numpy(np.stack(ll_list, axis=0)).float()
    hf = {
        "HL": torch.from_numpy(np.stack(hl_list, axis=0)).float(),
        "LH": torch.from_numpy(np.stack(lh_list, axis=0)).float(),
        "HH": torch.from_numpy(np.stack(hh_list, axis=0)).float(),
    }
    return ll_t, hf


def idwt2_rgb_tensor(ll_chw: Tensor, hf: Dict[str, Tensor], wavelet: str) -> Tensor:
    ll_np = ll_chw.detach().cpu().numpy()
    hl_np = hf["HL"].detach().cpu().numpy()
    lh_np = hf["LH"].detach().cpu().numpy()
    hh_np = hf["HH"].detach().cpu().numpy()

    rec_list = []
    for c in range(ll_np.shape[0]):
        rec = pywt.idwt2((ll_np[c], (hl_np[c], lh_np[c], hh_np[c])), wavelet=wavelet, mode="periodization")
        rec_list.append(rec)
    rec_t = torch.from_numpy(np.stack(rec_list, axis=0)).float()
    return rec_t


def parse_steps(args: argparse.Namespace) -> List[int]:
    if args.self_correct_k <= 0:
        return []

    if args.self_correct_steps.strip():
        steps = [int(x.strip()) for x in args.self_correct_steps.split(",") if x.strip()]
        steps = steps[: args.self_correct_k]
        if len(steps) < args.self_correct_k:
            raise ValueError("self_correct_steps count is smaller than self_correct_k")
    else:
        # Auto-generate descending shallow steps from t_star.
        top = max(2, args.t_star)
        steps = np.linspace(top, 2, args.self_correct_k + 1, dtype=int).tolist()[:-1]
    return steps


def compute_metrics(ref_chw: Tensor, pred_chw: Tensor) -> Dict[str, float]:
    ref = ref_chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    pred = pred_chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    ssim_val = float(structural_similarity(ref, pred, channel_axis=2, data_range=1.0))
    psnr_val = float(peak_signal_noise_ratio(ref, pred, data_range=1.0))
    return {"SSIM": ssim_val, "PSNR": psnr_val}


def make_triplet_panel(images: List[Tensor]) -> Image.Image:
    pil_imgs = [tensor_to_pil(x) for x in images]
    widths = [im.width for im in pil_imgs]
    heights = [im.height for im in pil_imgs]
    canvas = Image.new("RGB", (sum(widths), max(heights)))
    x_off = 0
    for im in pil_imgs:
        canvas.paste(im, (x_off, 0))
        x_off += im.width
    return canvas


def process_one_image(
    image_path: Path,
    args: argparse.Namespace,
    alpha_bars: Tensor,
    predictor: GaussianCTMPredictor,
    loop_steps: List[int],
    device: torch.device,
) -> Dict[str, object]:
    sample_dir = args.output_dir / image_path.stem
    ensure_dir(sample_dir)

    img = Image.open(image_path).convert("RGB")
    if args.resize is not None:
        h, w = args.resize
        img = img.resize((w, h), Image.BICUBIC)

    x_adv = pil_to_tensor(img).unsqueeze(0).to(device)

    save_tensor_image(x_adv[0], sample_dir / "00_x_adv.png")

    ll_orig, hf_orig = dwt2_rgb_tensor(x_adv[0], wavelet=args.wavelet)
    save_tensor_image(ll_orig.clamp(0.0, 1.0), sample_dir / "01_LL_orig.png")
    save_tensor_image(coeff_to_vis(hf_orig["HL"]), sample_dir / "01_HL_orig_vis.png")
    save_tensor_image(coeff_to_vis(hf_orig["LH"]), sample_dir / "01_LH_orig_vis.png")
    save_tensor_image(coeff_to_vis(hf_orig["HH"]), sample_dir / "01_HH_orig_vis.png")

    alpha_t = float(alpha_bars[args.t_star].item())
    x_t = add_noise(x_adv, alpha_t)
    save_tensor_image(x_t[0].clamp(0.0, 1.0), sample_dir / f"02_x_t{args.t_star}.png")

    t0 = time.perf_counter()
    x0_hat = predictor(x_t, args.t_star)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    save_tensor_image(x0_hat[0], sample_dir / "03_x0_hat.png")

    ll_hat, hf_hat = dwt2_rgb_tensor(x0_hat[0], wavelet=args.wavelet)
    save_tensor_image(ll_hat.clamp(0.0, 1.0), sample_dir / "04_LL_hat.png")
    save_tensor_image(coeff_to_vis(hf_hat["HL"]), sample_dir / "04_HL_hat_vis.png")
    save_tensor_image(coeff_to_vis(hf_hat["LH"]), sample_dir / "04_LH_hat_vis.png")
    save_tensor_image(coeff_to_vis(hf_hat["HH"]), sample_dir / "04_HH_hat_vis.png")

    x_corrected = idwt2_rgb_tensor(ll_orig, hf_hat, wavelet=args.wavelet).to(device).unsqueeze(0)
    x_corrected = x_corrected.clamp(0.0, 1.0)
    save_tensor_image(x_corrected[0], sample_dir / "05_x_corrected.png")

    current = x_corrected.clone()
    for i, t_step in enumerate(loop_steps, start=1):
        alpha_k = float(alpha_bars[t_step].item())
        x_tk = add_noise(current, alpha_k)
        x_hat_k = predictor(x_tk, t_step)
        _, hf_k = dwt2_rgb_tensor(x_hat_k[0], wavelet=args.wavelet)
        current = idwt2_rgb_tensor(ll_orig, hf_k, wavelet=args.wavelet).to(device).unsqueeze(0).clamp(0.0, 1.0)

        save_tensor_image(x_tk[0].clamp(0.0, 1.0), sample_dir / f"06_loop{i}_x_t{t_step}.png")
        save_tensor_image(x_hat_k[0], sample_dir / f"06_loop{i}_x_hat.png")
        save_tensor_image(current[0], sample_dir / f"06_loop{i}_x_corrected.png")

    x_final = current
    save_tensor_image(x_final[0], sample_dir / "07_x_final.png")

    panel = make_triplet_panel([x_adv[0], x_corrected[0], x_final[0]])
    panel.save(sample_dir / "08_compare_adv_corrected_final.png")

    metrics = compute_metrics(x_adv[0], x_final[0])
    metrics["NFE"] = 1 + len(loop_steps)
    metrics["single_step_infer_ms"] = float(infer_ms)

    if args.clean_dir is not None:
        clean_path = args.clean_dir / image_path.name
        if clean_path.exists():
            clean_img = Image.open(clean_path).convert("RGB")
            if args.resize is not None:
                h, w = args.resize
                clean_img = clean_img.resize((w, h), Image.BICUBIC)
            x_clean = pil_to_tensor(clean_img)
            clean_metrics = compute_metrics(x_clean, x_final[0])
            metrics.update({f"clean_{k}": v for k, v in clean_metrics.items()})
        else:
            metrics["clean_metrics_warning"] = f"missing clean image: {clean_path}"

    result = {
        "image": str(image_path),
        "wavelet": args.wavelet,
        "t_star": args.t_star,
        "self_correct_steps": loop_steps,
        "metrics": metrics,
    }

    with open(sample_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)

    if args.t_star <= 0 or args.t_star >= args.num_diffusion_steps:
        raise ValueError("t_star must be in (0, num_diffusion_steps)")

    device = torch.device(args.device)
    alpha_bars = build_alpha_bars(args.num_diffusion_steps).to(device)

    wavelets = pywt.wavelist(kind="discrete")
    if args.wavelet not in wavelets:
        raise ValueError(f"unknown wavelet: {args.wavelet}")

    loop_steps = parse_steps(args)
    for step in loop_steps:
        if step <= 0 or step >= args.num_diffusion_steps:
            raise ValueError("all self-correct steps must be in (0, num_diffusion_steps)")

    image_paths = list_images(args.input_dir, args.glob, args.max_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    predictor = GaussianCTMPredictor(kernel_size=args.kernel_size, sigma=args.sigma, mix=args.mix)

    results = []
    for path in tqdm(image_paths, desc="WGCP small-scale"):
        result = process_one_image(path, args, alpha_bars, predictor, loop_steps, device)
        results.append(result)

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Results saved to: {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
