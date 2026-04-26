import argparse
import json
import getpass
import platform
import random
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


Tensor = torch.Tensor


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
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_pil(tensor: Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def save_tensor_image(tensor: Tensor, path: Path) -> None:
    tensor_to_pil(tensor).save(path)


def normalize_spatial_map(x: Tensor, eps: float = 1e-6) -> Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError("normalize_spatial_map expects [H,W] or [1,H,W] or [C,H,W]")
    if x.shape[0] > 1:
        x = x.mean(dim=0, keepdim=True)
    x = x.detach().float()
    x = x - x.min()
    denom = x.max().clamp_min(eps)
    return (x / denom).clamp(0.0, 1.0)


def spatial_map_to_rgb(x: Tensor) -> Tensor:
    heat = normalize_spatial_map(x)
    red = heat
    green = (1.0 - (2.0 * heat - 1.0).abs()).clamp(0.0, 1.0)
    blue = 1.0 - heat
    return torch.cat([red, green, blue], dim=0).clamp(0.0, 1.0)


def blend_heatmap_on_image(image_chw: Tensor, heatmap: Tensor, alpha: float = 0.45) -> Tensor:
    base = image_chw.detach().float().clamp(0.0, 1.0)
    heat_rgb = spatial_map_to_rgb(heatmap)
    mix = float(np.clip(alpha, 0.0, 1.0))
    return ((1.0 - mix) * base + mix * heat_rgb).clamp(0.0, 1.0)


def coeff_to_vis(coeff: Tensor) -> Tensor:
    max_abs = torch.max(torch.abs(coeff))
    if max_abs < 1e-8:
        return torch.zeros_like(coeff) + 0.5
    return (coeff / (2 * max_abs) + 0.5).clamp(0.0, 1.0)


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
    for img in pil_imgs:
        canvas.paste(img, (x_off, 0))
        x_off += img.width
    return canvas


def _safe_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return out or None
    except Exception:
        return None


def _jsonable_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def build_run_meta(
    args: argparse.Namespace,
    script_path: Optional[Path] = None,
) -> Dict[str, object]:
    now_local = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)
    resolved_script = script_path or Path(__file__).resolve()
    return {
        "run_id": now_local.strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "timestamp_local": now_local.isoformat(),
        "timestamp_utc": now_utc.isoformat(),
        "script": str(resolved_script),
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
    with open(archive_path, "w", encoding="utf-8") as handle:
        json.dump(summary_obj, handle, ensure_ascii=False, indent=2)
    return archive_path
