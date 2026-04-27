import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pywt
import torch
import torch.nn.functional as F

try:
    from .wgcp_wavelet import (
        adaptive_multiscale_fusion,
        dwt2_rgb_tensor,
        resolve_level_schedule,
        wavedec2_rgb_tensor,
        waverec2_rgb_tensor,
    )
except ImportError:
    from wgcp_wavelet import (
        adaptive_multiscale_fusion,
        dwt2_rgb_tensor,
        resolve_level_schedule,
        wavedec2_rgb_tensor,
        waverec2_rgb_tensor,
    )


Tensor = torch.Tensor


@lru_cache(maxsize=16)
def _wavelet_filters_2d(wavelet: str) -> Tuple[Tensor, int]:
    wave = pywt.Wavelet(wavelet)
    dec_lo = torch.tensor(list(reversed(wave.dec_lo)), dtype=torch.float32)
    dec_hi = torch.tensor(list(reversed(wave.dec_hi)), dtype=torch.float32)
    ll = torch.outer(dec_lo, dec_lo)
    lh = torch.outer(dec_lo, dec_hi)
    hl = torch.outer(dec_hi, dec_lo)
    hh = torch.outer(dec_hi, dec_hi)
    filters = torch.stack([ll, lh, hl, hh], dim=0)
    return filters, int(dec_lo.numel())


def _pad_for_stride2(x: Tensor, filter_len: int) -> Tensor:
    total = max(0, filter_len - 2)
    pad_l = total // 2
    pad_r = total - pad_l
    pad_t = total // 2
    pad_b = total - pad_t
    mode = "reflect"
    if x.shape[-2] <= pad_t or x.shape[-2] <= pad_b or x.shape[-1] <= pad_l or x.shape[-1] <= pad_r:
        mode = "replicate"
    return F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)


def dwt2_torch(x: Tensor, wavelet: str) -> Dict[str, Tensor]:
    if x.ndim != 4:
        raise ValueError("dwt2_torch expects x with shape [B, C, H, W]")
    base_filters, filter_len = _wavelet_filters_2d(wavelet)
    filters = base_filters.to(device=x.device, dtype=x.dtype).unsqueeze(1)
    channels = int(x.shape[1])
    filters = filters.repeat(channels, 1, 1, 1)
    x_pad = _pad_for_stride2(x, filter_len=filter_len)
    coeff = F.conv2d(x_pad, filters, stride=2, groups=channels)
    coeff = coeff.view(x.shape[0], channels, 4, coeff.shape[-2], coeff.shape[-1])
    return {
        "LL": coeff[:, :, 0],
        "LH": coeff[:, :, 1],
        "HL": coeff[:, :, 2],
        "HH": coeff[:, :, 3],
    }


def wavedec2_torch(x: Tensor, wavelet: str, levels: int) -> List[Dict[str, Tensor]]:
    coeff_levels: List[Dict[str, Tensor]] = []
    cur = x
    for _ in range(max(1, int(levels))):
        coeff = dwt2_torch(cur, wavelet=wavelet)
        coeff_levels.append(coeff)
        cur = coeff["LL"]
        if min(int(cur.shape[-2]), int(cur.shape[-1])) < 2:
            break
    return coeff_levels


def wavelet_guided_loss(
    x: Tensor,
    ref: Tensor,
    wavelet: str,
    levels: int,
    lambda_ll_text: str,
    lambda_h_text: str,
    lambda_hh_text: str,
) -> Tuple[Tensor, Dict[str, object]]:
    x_coeffs = wavedec2_torch(x, wavelet=wavelet, levels=levels)
    ref_coeffs = wavedec2_torch(ref, wavelet=wavelet, levels=levels)
    use_levels = min(len(x_coeffs), len(ref_coeffs))
    ll_sched = resolve_level_schedule(use_levels, lambda_ll_text, fallback=10.0)
    h_sched = resolve_level_schedule(use_levels, lambda_h_text, fallback=1.0)
    hh_sched = resolve_level_schedule(use_levels, lambda_hh_text, fallback=0.5)

    loss = x.new_tensor(0.0)
    level_stats: Dict[str, Dict[str, float]] = {}
    for level in range(1, use_levels + 1):
        x_level = x_coeffs[level - 1]
        ref_level = ref_coeffs[level - 1]
        ll_term = (x_level["LL"] - ref_level["LL"]).abs().mean()
        lh_term = (x_level["LH"] - ref_level["LH"]).abs().mean()
        hl_term = (x_level["HL"] - ref_level["HL"]).abs().mean()
        hh_term = (x_level["HH"] - ref_level["HH"]).abs().mean()
        weighted = (
            float(ll_sched[level]) * ll_term
            + float(h_sched[level]) * (lh_term + hl_term)
            + float(hh_sched[level]) * hh_term
        )
        loss = loss + weighted
        level_stats[f"L{level}"] = {
            "lambda_ll": float(ll_sched[level]),
            "lambda_h": float(h_sched[level]),
            "lambda_hh": float(hh_sched[level]),
            "ll_l1": float(ll_term.detach().item()),
            "lh_l1": float(lh_term.detach().item()),
            "hl_l1": float(hl_term.detach().item()),
            "hh_l1": float(hh_term.detach().item()),
        }

    meta: Dict[str, object] = {
        "levels": int(use_levels),
        "lambda_ll_schedule": {f"L{k}": float(v) for k, v in ll_sched.items()},
        "lambda_h_schedule": {f"L{k}": float(v) for k, v in h_sched.items()},
        "lambda_hh_schedule": {f"L{k}": float(v) for k, v in hh_sched.items()},
        "level_stats": level_stats,
    }
    return loss, meta


def _build_c1_reference(
    x_adv_chw: Tensor,
    x_cm_chw: Tensor,
    wavelet: str,
    levels: int,
    gamma_text: str,
    ll_alpha: float,
    eps: float,
    target_hw: Optional[Tuple[int, int]],
) -> Tuple[Tensor, Dict[str, object]]:
    return adaptive_multiscale_fusion(
        x_orig_chw=x_adv_chw,
        x_pred_chw=x_cm_chw,
        wavelet=wavelet,
        levels=levels,
        gamma_text=gamma_text,
        w_min=0.0,
        w_max=1.0,
        ll_alpha=ll_alpha,
        eps=eps,
        target_hw=target_hw,
    )


def wgcp_v2_cm_output(
    x_adv_chw: Tensor,
    x_cm_chw: Tensor,
    wavelet: str,
    levels: int,
    c1_gamma_text: str,
    c1_ll_alpha: float,
    eps: float = 1e-6,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Dict[str, object]]:
    x_c1, c1_meta = _build_c1_reference(
        x_adv_chw=x_adv_chw,
        x_cm_chw=x_cm_chw,
        wavelet=wavelet,
        levels=levels,
        gamma_text=c1_gamma_text,
        ll_alpha=c1_ll_alpha,
        eps=eps,
        target_hw=target_hw,
    )
    x_out = x_cm_chw.clamp(0.0, 1.0)
    _, hf_out = dwt2_rgb_tensor(x_out, wavelet=wavelet)
    ll_out, _, use_levels = wavedec2_rgb_tensor(x_out, wavelet=wavelet, levels=levels)
    return x_out, {
        "variant": "cm",
        "x_cm": x_cm_chw.detach().clone(),
        "x_c1": x_c1.detach().clone(),
        "c1_meta": c1_meta,
        "hf_policy": "pure_consistency_model_output",
        "ll_final": ll_out,
        "hf_final": hf_out,
        "levels": int(use_levels),
    }


def wgcp_v2_lowfreq_fusion(
    x_adv_chw: Tensor,
    x_cm_chw: Tensor,
    wavelet: str,
    levels: int,
    c1_gamma_text: str,
    c1_ll_alpha: float,
    eps: float = 1e-6,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Dict[str, object]]:
    x_c1, c1_meta = _build_c1_reference(
        x_adv_chw=x_adv_chw,
        x_cm_chw=x_cm_chw,
        wavelet=wavelet,
        levels=levels,
        gamma_text=c1_gamma_text,
        ll_alpha=c1_ll_alpha,
        eps=eps,
        target_hw=target_hw,
    )
    ll_c1, _, use_levels = wavedec2_rgb_tensor(x_c1, wavelet=wavelet, levels=levels)
    _, hf_cm_levels, _ = wavedec2_rgb_tensor(x_cm_chw, wavelet=wavelet, levels=use_levels)
    x_out = waverec2_rgb_tensor(ll_c1, hf_cm_levels, wavelet=wavelet, target_hw=target_hw).clamp(0.0, 1.0)
    _, hf_out = dwt2_rgb_tensor(x_out, wavelet=wavelet)
    return x_out, {
        "variant": "fuse",
        "x_cm": x_cm_chw.detach().clone(),
        "x_c1": x_c1.detach().clone(),
        "c1_meta": c1_meta,
        "hf_policy": "c1_lowfreq_plus_cm_highfreq",
        "ll_final": ll_c1,
        "hf_final": hf_out,
        "levels": int(use_levels),
    }


def wgcp_v2_optimize(
    x_adv_chw: Tensor,
    x_cm_chw: Tensor,
    wavelet: str,
    levels: int,
    c1_gamma_text: str,
    c1_ll_alpha: float,
    lambda_ll_text: str,
    lambda_h_text: str,
    lambda_hh_text: str,
    pixel_gamma: float,
    steps: int,
    lr: float,
    eps: float = 1e-6,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Dict[str, object]]:
    x_c1, c1_meta = _build_c1_reference(
        x_adv_chw=x_adv_chw,
        x_cm_chw=x_cm_chw,
        wavelet=wavelet,
        levels=levels,
        gamma_text=c1_gamma_text,
        ll_alpha=c1_ll_alpha,
        eps=eps,
        target_hw=target_hw,
    )

    x_cm = x_cm_chw.unsqueeze(0).detach()
    x_ref = x_c1.to(device=x_cm.device, dtype=x_cm.dtype).unsqueeze(0).detach()
    x_var = x_cm.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x_var], lr=float(lr))

    history: List[Dict[str, float]] = []
    wave_meta: Dict[str, object] = {}
    for step_idx in range(max(0, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        loss_wave, wave_meta = wavelet_guided_loss(
            x=x_var,
            ref=x_ref,
            wavelet=wavelet,
            levels=levels,
            lambda_ll_text=lambda_ll_text,
            lambda_h_text=lambda_h_text,
            lambda_hh_text=lambda_hh_text,
        )
        loss_pixel = F.mse_loss(x_var, x_cm)
        loss_total = loss_wave + float(pixel_gamma) * loss_pixel
        loss_total.backward()
        optimizer.step()
        with torch.no_grad():
            x_var.clamp_(0.0, 1.0)
        history.append(
            {
                "step": float(step_idx + 1),
                "loss_total": float(loss_total.detach().item()),
                "loss_wavelet": float(loss_wave.detach().item()),
                "loss_pixel": float(loss_pixel.detach().item()),
            }
        )

    x_out = x_var.detach().squeeze(0)
    ll_out, _, use_levels = wavedec2_rgb_tensor(x_out, wavelet=wavelet, levels=levels)
    _, hf_out = dwt2_rgb_tensor(x_out, wavelet=wavelet)
    return x_out, {
        "variant": "opt",
        "x_cm": x_cm_chw.detach().clone(),
        "x_c1": x_c1.detach().clone(),
        "c1_meta": c1_meta,
        "wavelet_meta": wave_meta,
        "loss_history": history,
        "pixel_gamma": float(pixel_gamma),
        "steps": int(steps),
        "lr": float(lr),
        "hf_policy": "cm_init_wavelet_guided_optimization",
        "ll_final": ll_out,
        "hf_final": hf_out,
        "levels": int(use_levels),
    }
