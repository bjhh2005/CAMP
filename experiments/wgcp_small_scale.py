import argparse
import importlib
import json
import math
import random
import time
import getpass
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
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
    parser.add_argument("--mix", type=float, default=0.45, help="x_t -> denoised blend ratio")
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
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument(
        "--archive_dir",
        type=Path,
        default=Path.home() / ".camp_runs" / "CAMP",
        help="Directory for archived run records (default outside repo).",
    )
    parser.add_argument("--archive_tag", type=str, default="", help="Optional tag appended to archive filename.")
    parser.add_argument("--disable_archive", action="store_true", help="Disable extra archived run record.")
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


def build_predictor(args: argparse.Namespace, device: torch.device):
    if args.predictor_type == "gaussian":
        return GaussianCTMPredictor(kernel_size=args.kernel_size, sigma=args.sigma, mix=args.mix)

    if args.predictor_type == "module":
        if ":" not in args.predictor_module:
            raise ValueError("predictor_module must be 'package.module:ClassName'")
        module_name, class_name = args.predictor_module.split(":", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        kwargs = json.loads(args.predictor_kwargs_json)
        kwargs.setdefault("device", str(device))
        predictor = cls(**kwargs)
        if not callable(predictor):
            raise TypeError("External predictor must be callable")
        return predictor

    raise ValueError(f"Unsupported predictor_type: {args.predictor_type}")


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


def idwt2_rgb_tensor(
    ll_chw: Tensor,
    hf: Dict[str, Tensor],
    wavelet: str,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tensor:
    ll_np = ll_chw.detach().cpu().numpy()
    hl_np = hf["HL"].detach().cpu().numpy()
    lh_np = hf["LH"].detach().cpu().numpy()
    hh_np = hf["HH"].detach().cpu().numpy()

    rec_list = []
    for c in range(ll_np.shape[0]):
        rec = pywt.idwt2((ll_np[c], (hl_np[c], lh_np[c], hh_np[c])), wavelet=wavelet, mode="periodization")
        rec_list.append(rec)
    rec_t = torch.from_numpy(np.stack(rec_list, axis=0)).float()
    if target_hw is not None:
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        rec_t = rec_t[:, :target_h, :target_w]
        cur_h, cur_w = rec_t.shape[-2], rec_t.shape[-1]
        pad_h = max(0, target_h - cur_h)
        pad_w = max(0, target_w - cur_w)
        if pad_h > 0 or pad_w > 0:
            rec_t = F.pad(rec_t, (0, pad_w, 0, pad_h), mode="replicate")
    return rec_t


def robust_sigma(coeff: Tensor) -> Tensor:
    flat = coeff.abs().flatten(1)
    med = flat.median(dim=1).values
    return med / 0.6745


def soft_shrink(coeff: Tensor, tau: Tensor) -> Tensor:
    while tau.ndim < coeff.ndim:
        tau = tau.unsqueeze(-1)
    return coeff.sign() * torch.relu(coeff.abs() - tau)


def fuse_high_frequency(
    hf_orig: Dict[str, Tensor],
    hf_hat: Dict[str, Tensor],
    hf_preserve: float,
    hf_shrink: float,
) -> Dict[str, Tensor]:
    preserve = float(np.clip(hf_preserve, 0.0, 1.0))
    fused: Dict[str, Tensor] = {}

    for band in ("HL", "LH", "HH"):
        orig = hf_orig[band]
        hat = hf_hat[band]

        sigma = robust_sigma(orig)
        tau = hf_shrink * sigma
        orig_denoised = soft_shrink(orig, tau)
        fused[band] = (1.0 - preserve) * hat + preserve * orig_denoised

    return fused


@torch.no_grad()
def run_predictor(
    predictor,
    x_t: Tensor,
    t_index: int,
    predictor_image_size: int,
) -> Tensor:
    if predictor_image_size and predictor_image_size > 0:
        h, w = x_t.shape[-2:]
        x_small = F.interpolate(
            x_t,
            size=(predictor_image_size, predictor_image_size),
            mode="bilinear",
            align_corners=False,
        )
        x0_small = predictor(x_small, t_index)
        x0_hat = F.interpolate(x0_small, size=(h, w), mode="bilinear", align_corners=False)
        return x0_hat.clamp(0.0, 1.0)
    return predictor(x_t, t_index).clamp(0.0, 1.0)


def parse_steps(args: argparse.Namespace) -> List[int]:
    if args.self_correct_k <= 0:
        return []

    if args.self_correct_steps.strip():
        steps = [int(x.strip()) for x in args.self_correct_steps.split(",") if x.strip()]
        steps = steps[: args.self_correct_k]
        if len(steps) < args.self_correct_k:
            raise ValueError("self_correct_steps count is smaller than self_correct_k")
    else:
        bridge = args.t_bridge if args.t_bridge > 0 else max(2, int(round(args.t_star * 0.25)))
        if args.self_correct_k == 1:
            steps = [bridge]
        else:
            tail = np.linspace(bridge, 2, args.self_correct_k, dtype=int).tolist()
            steps = []
            for s in tail:
                if not steps or steps[-1] != s:
                    steps.append(s)
    return [int(s) for s in steps]


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


def _safe_git_commit() -> Optional[str]:
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


def archive_run_record(args: argparse.Namespace, summary_obj: Dict[str, object], script_name: str) -> Optional[Path]:
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


def purify_adv_tensor(
    x_adv: Tensor,
    alpha_bars: Tensor,
    predictor,
    wavelet: str,
    t_star: int,
    loop_steps: List[int],
    hf_preserve: float = 0.35,
    hf_shrink: float = 0.6,
    replacement_mode: str = "hard",
    predictor_image_size: int = 0,
    ablation_ll_source: str = "orig",
    ablation_hard_hf_source: str = "pred",
) -> Tuple[Tensor, Tensor, Dict[str, object], float]:
    """
    Purify one adversarial tensor with WGCP.
    x_adv shape: [1, C, H, W]
    Returns: (x_corrected, x_final, trace, single_step_infer_ms)
    """
    if x_adv.ndim != 4 or x_adv.shape[0] != 1:
        raise ValueError("x_adv must have shape [1, C, H, W]")
    target_hw = (int(x_adv.shape[-2]), int(x_adv.shape[-1]))

    ll_orig, hf_orig = dwt2_rgb_tensor(x_adv[0], wavelet=wavelet)

    alpha_t = float(alpha_bars[t_star].item())
    x_t = add_noise(x_adv, alpha_t)

    t0 = time.perf_counter()
    x0_hat = run_predictor(predictor, x_t, t_star, predictor_image_size=predictor_image_size)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    ll_hat, hf_hat = dwt2_rgb_tensor(x0_hat[0], wavelet=wavelet)
    ll_anchor = ll_orig if ablation_ll_source == "orig" else ll_hat
    # CAMP default: keep LL from x_adv (ll_orig) and use predictor HF under hard replacement.
    if replacement_mode == "hard":
        hf_selected = hf_hat if ablation_hard_hf_source == "pred" else hf_orig
    elif replacement_mode == "fused":
        hf_selected = fuse_high_frequency(hf_orig, hf_hat, hf_preserve=hf_preserve, hf_shrink=hf_shrink)
    else:
        raise ValueError(f"Unknown replacement_mode: {replacement_mode}")

    x_corrected = idwt2_rgb_tensor(ll_anchor, hf_selected, wavelet=wavelet, target_hw=target_hw).to(x_adv.device).unsqueeze(0)
    x_corrected = x_corrected.clamp(0.0, 1.0)

    current = x_corrected.clone()
    loop_records: List[Dict[str, object]] = []
    for t_step in loop_steps:
        alpha_k = float(alpha_bars[t_step].item())
        x_tk = add_noise(current, alpha_k)
        x_hat_k = run_predictor(predictor, x_tk, t_step, predictor_image_size=predictor_image_size)
        ll_k, hf_k = dwt2_rgb_tensor(x_hat_k[0], wavelet=wavelet)
        ll_anchor_k = ll_orig if ablation_ll_source == "orig" else ll_k
        if replacement_mode == "hard":
            hf_k_selected = hf_k if ablation_hard_hf_source == "pred" else hf_orig
        else:
            hf_k_selected = fuse_high_frequency(hf_orig, hf_k, hf_preserve=hf_preserve, hf_shrink=hf_shrink)
        current = (
            idwt2_rgb_tensor(ll_anchor_k, hf_k_selected, wavelet=wavelet, target_hw=target_hw)
            .to(x_adv.device)
            .unsqueeze(0)
            .clamp(0.0, 1.0)
        )
        loop_records.append(
            {
                "t_step": t_step,
                "x_tk": x_tk.detach().clone(),
                "x_hat_k": x_hat_k.detach().clone(),
                "x_corrected_k": current.detach().clone(),
                "ll_anchor_source": ablation_ll_source,
                "hf_k_selected": {k: v.detach().clone() for k, v in hf_k_selected.items()},
            }
        )

    trace: Dict[str, object] = {
        "ll_orig": ll_orig,
        "hf_orig": hf_orig,
        "x_t": x_t.detach().clone(),
        "x0_hat": x0_hat.detach().clone(),
        "ll_hat": ll_hat,
        "hf_hat": hf_hat,
        "ll_anchor": ll_anchor,
        "hf_selected": hf_selected,
        "x_corrected": x_corrected.detach().clone(),
        "loop_records": loop_records,
        "ablation": {
            "ll_source": ablation_ll_source,
            "hard_hf_source": ablation_hard_hf_source,
        },
    }
    return x_corrected, current, trace, infer_ms


def process_one_image(
    image_path: Path,
    args: argparse.Namespace,
    alpha_bars: Tensor,
    predictor,
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

    ll_orig = trace["ll_orig"]
    hf_orig = trace["hf_orig"]
    save_tensor_image(ll_orig.clamp(0.0, 1.0), sample_dir / "01_LL_orig.png")
    save_tensor_image(coeff_to_vis(hf_orig["HL"]), sample_dir / "01_HL_orig_vis.png")
    save_tensor_image(coeff_to_vis(hf_orig["LH"]), sample_dir / "01_LH_orig_vis.png")
    save_tensor_image(coeff_to_vis(hf_orig["HH"]), sample_dir / "01_HH_orig_vis.png")

    x_t = trace["x_t"]
    save_tensor_image(x_t[0].clamp(0.0, 1.0), sample_dir / f"02_x_t{args.t_star}.png")

    x0_hat = trace["x0_hat"]
    save_tensor_image(x0_hat[0], sample_dir / "03_x0_hat.png")

    ll_hat = trace["ll_hat"]
    ll_anchor = trace["ll_anchor"]
    hf_hat = trace["hf_hat"]
    hf_selected = trace["hf_selected"]
    save_tensor_image(ll_hat.clamp(0.0, 1.0), sample_dir / "04_LL_hat.png")
    save_tensor_image(ll_anchor.clamp(0.0, 1.0), sample_dir / "04_LL_anchor.png")
    save_tensor_image(coeff_to_vis(hf_hat["HL"]), sample_dir / "04_HL_hat_vis.png")
    save_tensor_image(coeff_to_vis(hf_hat["LH"]), sample_dir / "04_LH_hat_vis.png")
    save_tensor_image(coeff_to_vis(hf_hat["HH"]), sample_dir / "04_HH_hat_vis.png")
    save_tensor_image(coeff_to_vis(hf_selected["HL"]), sample_dir / "04_HL_selected_vis.png")
    save_tensor_image(coeff_to_vis(hf_selected["LH"]), sample_dir / "04_LH_selected_vis.png")
    save_tensor_image(coeff_to_vis(hf_selected["HH"]), sample_dir / "04_HH_selected_vis.png")

    save_tensor_image(x_corrected[0], sample_dir / "05_x_assembled.png")
    save_tensor_image(x_corrected[0], sample_dir / "05_x_corrected.png")

    for i, record in enumerate(trace["loop_records"], start=1):
        t_step = int(record["t_step"])
        x_tk = record["x_tk"]
        x_hat_k = record["x_hat_k"]
        x_corr_k = record["x_corrected_k"]
        save_tensor_image(x_tk[0].clamp(0.0, 1.0), sample_dir / f"06_loop{i}_x_t{t_step}.png")
        save_tensor_image(x_hat_k[0], sample_dir / f"06_loop{i}_x_hat.png")
        save_tensor_image(x_corr_k[0], sample_dir / f"06_loop{i}_x_corrected.png")

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
        "t_bridge": args.t_bridge if args.t_bridge > 0 else max(2, int(round(args.t_star * 0.25))),
        "self_correct_steps": loop_steps,
        "predictor_type": args.predictor_type,
        "predictor_image_size": args.predictor_image_size,
        "replacement_mode": args.replacement_mode,
        "ablation_ll_source": args.ablation_ll_source,
        "ablation_hard_hf_source": args.ablation_hard_hf_source,
        "hf_preserve": args.hf_preserve,
        "hf_shrink": args.hf_shrink,
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

    predictor = build_predictor(args, device=device)

    results = []
    for path in tqdm(image_paths, desc="WGCP small-scale"):
        result = process_one_image(path, args, alpha_bars, predictor, loop_steps, device)
        results.append(result)

    run_meta = build_run_meta(args)
    summary_obj: Dict[str, object] = {
        "run_meta": run_meta,
        "samples": results,
    }

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)

    archive_path = archive_run_record(args, summary_obj, script_name="wgcp_small_scale")

    print(f"Done. Results saved to: {args.output_dir}")
    print(f"Summary: {summary_path}")
    if archive_path is not None:
        print(f"Archived summary: {archive_path}")


if __name__ == "__main__":
    main()
