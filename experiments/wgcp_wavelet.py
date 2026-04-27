import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pywt
import torch
import torch.nn.functional as F


Tensor = torch.Tensor
WAVELET_MODE = "reflect"


def dwt2_rgb_tensor(x_chw: Tensor, wavelet: str) -> Tuple[Tensor, Dict[str, Tensor]]:
    x_np = x_chw.detach().cpu().numpy()
    ll_list, hl_list, lh_list, hh_list = [], [], [], []
    for channel in range(x_np.shape[0]):
        ll, (ch, cv, cd) = pywt.dwt2(x_np[channel], wavelet=wavelet, mode=WAVELET_MODE)
        ll_list.append(ll)
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
    for channel in range(ll_np.shape[0]):
        rec = pywt.idwt2((ll_np[channel], (hl_np[channel], lh_np[channel], hh_np[channel])), wavelet=wavelet, mode=WAVELET_MODE)
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


def resolve_level_schedule(levels: int, values_text: str, fallback: float) -> Dict[int, float]:
    parts = [item.strip() for item in values_text.split(",") if item.strip()]
    vals = [float(fallback)] if not parts else [float(item) for item in parts]
    out: Dict[int, float] = {}
    for level in range(1, levels + 1):
        idx = min(level - 1, len(vals) - 1)
        out[level] = float(vals[idx])
    return out


def resolve_ms_gamma_schedule(levels: int, gamma_text: str) -> Dict[int, float]:
    return resolve_level_schedule(levels, gamma_text, fallback=1.0)


def wavedec2_rgb_tensor(
    x_chw: Tensor,
    wavelet: str,
    levels: int,
) -> Tuple[Tensor, Dict[int, Dict[str, Tensor]], int]:
    x_np = x_chw.detach().cpu().numpy()
    wave = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(min(int(x_chw.shape[-2]), int(x_chw.shape[-1])), wave.dec_len)
    use_levels = max(1, min(int(levels), int(max_level)))

    ll_list = []
    hf_collect: Dict[int, Dict[str, List[np.ndarray]]] = {
        level: {"HL": [], "LH": [], "HH": []} for level in range(1, use_levels + 1)
    }
    for channel in range(x_np.shape[0]):
        coeffs = pywt.wavedec2(x_np[channel], wavelet=wavelet, mode=WAVELET_MODE, level=use_levels)
        ll_list.append(coeffs[0])
        for level in range(1, use_levels + 1):
            idx = use_levels - level + 1
            ch, cv, cd = coeffs[idx]
            hf_collect[level]["HL"].append(ch)
            hf_collect[level]["LH"].append(cv)
            hf_collect[level]["HH"].append(cd)

    ll_t = torch.from_numpy(np.stack(ll_list, axis=0)).float()
    hf_levels: Dict[int, Dict[str, Tensor]] = {}
    for level in range(1, use_levels + 1):
        hf_levels[level] = {
            band: torch.from_numpy(np.stack(hf_collect[level][band], axis=0)).float()
            for band in ("HL", "LH", "HH")
        }
    return ll_t, hf_levels, use_levels


def waverec2_rgb_tensor(
    ll_chw: Tensor,
    hf_levels: Dict[int, Dict[str, Tensor]],
    wavelet: str,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tensor:
    levels = len(hf_levels)
    ll_np = ll_chw.detach().cpu().numpy()
    hf_np_levels: Dict[int, Dict[str, np.ndarray]] = {
        level: {band: hf_levels[level][band].detach().cpu().numpy() for band in ("HL", "LH", "HH")}
        for level in range(1, levels + 1)
    }
    rec_list = []
    for channel in range(ll_np.shape[0]):
        coeffs = [ll_np[channel]]
        for level in range(levels, 0, -1):
            bands = hf_np_levels[level]
            coeffs.append((bands["HL"][channel], bands["LH"][channel], bands["HH"][channel]))
        rec = pywt.waverec2(coeffs, wavelet=wavelet, mode=WAVELET_MODE)
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


def median_abs_deviation(coeff: Tensor, eps: float = 1e-6) -> Tensor:
    flat = coeff.flatten(1)
    med = flat.median(dim=1).values.unsqueeze(-1).unsqueeze(-1)
    mad = (coeff - med).abs().flatten(1).median(dim=1).values
    return mad.clamp_min(eps)


def expand_channel_stat(stat: Tensor, coeff: Tensor) -> Tensor:
    view = stat
    while view.ndim < coeff.ndim:
        view = view.unsqueeze(-1)
    return view


def normalized_gate_map(delta: Tensor, scale: Tensor, gate_tau: float, gate_gain: float) -> Tensor:
    scale_map = expand_channel_stat(scale.clamp_min(1e-6), delta)
    score = delta.abs() / scale_map
    return torch.sigmoid(float(gate_gain) * (score - float(gate_tau)))


def robust_sigma(coeff: Tensor) -> Tensor:
    flat = coeff.abs().flatten(1)
    med = flat.median(dim=1).values
    return med / 0.6745


def soft_shrink(coeff: Tensor, tau: Tensor) -> Tensor:
    while tau.ndim < coeff.ndim:
        tau = tau.unsqueeze(-1)
    return coeff.sign() * torch.relu(coeff.abs() - tau)


def _make_gaussian_kernel2d(sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    sigma = max(float(sigma), 0.5)
    kernel_size = max(3, int(math.ceil(6.0 * sigma)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g = g / g.sum().clamp_min(1e-8)
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum().clamp_min(1e-8)
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


def _gaussian_blur_map(x: Tensor, sigma: float) -> Tensor:
    if x.ndim != 4:
        raise ValueError("_gaussian_blur_map expects x with shape [B, C, H, W]")
    kernel = _make_gaussian_kernel2d(sigma=sigma, device=x.device, dtype=x.dtype)
    padding = kernel.shape[-1] // 2
    return F.conv2d(x, kernel.expand(x.shape[1], 1, -1, -1), padding=padding, groups=x.shape[1])


def _resolve_edge_sigma(coeff: Tensor, sigma_divisor: float) -> float:
    short_side = max(1, min(int(coeff.shape[-2]), int(coeff.shape[-1])))
    divisor = max(float(sigma_divisor), 1.0)
    return max(0.5, short_side / divisor)


def _normalize_spatial_map(x: Tensor, eps: float = 1e-6) -> Tensor:
    if x.ndim != 3:
        raise ValueError("_normalize_spatial_map expects x with shape [C, H, W] or [1, H, W]")
    flat = x.flatten(1)
    x_max = flat.max(dim=1).values.view(-1, 1, 1)
    return x / x_max.clamp_min(eps)


def _compute_edge_activity_map(
    coeff: Tensor,
    sigma_divisor: float,
    use_channel_aggregation: bool = True,
) -> Tensor:
    sigma = _resolve_edge_sigma(coeff, sigma_divisor=sigma_divisor)
    energy = coeff.pow(2)
    if use_channel_aggregation:
        energy = energy.sum(dim=0, keepdim=True)
    blurred = _gaussian_blur_map(energy.unsqueeze(0), sigma=sigma).squeeze(0)
    return torch.sqrt(blurred.clamp_min(0.0) + 1e-12)


def _edge_alpha_from_activity(
    activity: Tensor,
    eta: float,
    alpha_min: float,
) -> Tensor:
    activity_norm = _normalize_spatial_map(activity)
    alpha = 1.0 - float(np.clip(eta, 0.0, 1.0)) * activity_norm
    return alpha.clamp(min=float(alpha_min), max=1.0)


def _compute_modulus_maxima_map(
    hl_coeff: Tensor,
    lh_coeff: Tensor,
    sigma_divisor: float,
    threshold: float,
) -> Tensor:
    sigma = _resolve_edge_sigma(hl_coeff, sigma_divisor=sigma_divisor)
    modulus = torch.sqrt((hl_coeff.pow(2) + lh_coeff.pow(2)).sum(dim=0, keepdim=True).clamp_min(0.0) + 1e-12)
    smooth = _gaussian_blur_map(modulus.unsqueeze(0), sigma=sigma).squeeze(0)
    smooth_norm = _normalize_spatial_map(smooth)
    local_max = F.max_pool2d(smooth_norm.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
    maxima_mask = (smooth_norm >= (local_max - 1e-6)).float()
    strong_mask = (smooth_norm >= float(np.clip(threshold, 0.0, 1.0))).float()
    return (smooth_norm * maxima_mask * strong_mask).clamp(0.0, 1.0)


def adaptive_multiscale_fusion(
    x_orig_chw: Tensor,
    x_pred_chw: Tensor,
    wavelet: str,
    levels: int,
    gamma_text: str,
    w_min: float,
    w_max: float,
    ll_alpha: float,
    eps: float = 1e-6,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Dict[str, object]]:
    ll_orig, hf_orig_levels, use_levels = wavedec2_rgb_tensor(x_orig_chw, wavelet=wavelet, levels=levels)
    ll_pred, _, _ = wavedec2_rgb_tensor(x_pred_chw, wavelet=wavelet, levels=use_levels)
    gamma_sched = resolve_ms_gamma_schedule(use_levels, gamma_text)

    ll_a = float(np.clip(ll_alpha, 0.0, 1.0))
    ll_final = (1.0 - ll_a) * ll_pred + ll_a * ll_orig

    hf_final_levels: Dict[int, Dict[str, Tensor]] = {}
    level_stats: Dict[str, Dict[str, float]] = {}
    for level in range(1, use_levels + 1):
        gamma = float(gamma_sched[level])
        lvl_stats: Dict[str, float] = {"gamma": gamma}
        hf_final_levels[level] = {}
        for band in ("HL", "LH", "HH"):
            orig = hf_orig_levels[level][band]
            mad = median_abs_deviation(orig, eps=eps)
            sigma = mad / 0.6745
            n_coeff = max(2.0, float(orig.shape[-2] * orig.shape[-1]))
            tau = gamma * sigma * math.sqrt(2.0 * math.log(n_coeff))
            hf_final_levels[level][band] = soft_shrink(orig, tau)
            lvl_stats[f"sigma_mean_{band}"] = float(sigma.mean().item())
            lvl_stats[f"tau_mean_{band}"] = float(tau.mean().item())
        level_stats[f"L{level}"] = lvl_stats

    x_rec = waverec2_rgb_tensor(ll_final, hf_final_levels, wavelet=wavelet, target_hw=target_hw)
    meta: Dict[str, object] = {
        "levels": int(use_levels),
        "gamma_schedule": {f"L{k}": float(v) for k, v in gamma_sched.items()},
        "legacy_w_min": float(min(w_min, w_max)),
        "legacy_w_max": float(max(w_min, w_max)),
        "ll_alpha": float(ll_a),
        "level_stats": level_stats,
        "ll_final": ll_final,
        "hf_final_levels": hf_final_levels,
        "ll_pred": ll_pred,
        "hf_policy": "orig_soft_shrink_only",
    }
    return x_rec, meta


def adaptive_multiscale_edge_fusion(
    x_orig_chw: Tensor,
    x_pred_chw: Tensor,
    wavelet: str,
    levels: int,
    gamma_text: str,
    edge_eta_text: str,
    edge_eta_hh_text: str,
    edge_sigma_divisor: float,
    edge_alpha_min: float,
    use_channel_aggregation: bool = True,
    eps: float = 1e-6,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Dict[str, object]]:
    ll_orig, hf_orig_levels, use_levels = wavedec2_rgb_tensor(x_orig_chw, wavelet=wavelet, levels=levels)
    ll_pred, _, _ = wavedec2_rgb_tensor(x_pred_chw, wavelet=wavelet, levels=use_levels)
    del ll_pred
    gamma_sched = resolve_ms_gamma_schedule(use_levels, gamma_text)
    edge_eta_sched = resolve_level_schedule(use_levels, edge_eta_text, fallback=0.5)
    edge_eta_hh_sched = resolve_level_schedule(use_levels, edge_eta_hh_text, fallback=0.3)

    ll_final = ll_orig
    hf_final_levels: Dict[int, Dict[str, Tensor]] = {}
    level_stats: Dict[str, Dict[str, float]] = {}
    for level in range(1, use_levels + 1):
        gamma = float(gamma_sched[level])
        eta_main = float(np.clip(edge_eta_sched[level], 0.0, 1.0))
        eta_hh = float(np.clip(edge_eta_hh_sched[level], 0.0, 1.0))
        lvl_stats: Dict[str, float] = {
            "gamma": gamma,
            "edge_eta": eta_main,
            "edge_eta_hh": eta_hh,
        }
        hf_final_levels[level] = {}
        for band in ("HL", "LH", "HH"):
            orig = hf_orig_levels[level][band]
            mad = median_abs_deviation(orig, eps=eps)
            sigma = mad / 0.6745
            n_coeff = max(2.0, float(orig.shape[-2] * orig.shape[-1]))
            tau = gamma * sigma * math.sqrt(2.0 * math.log(n_coeff))
            activity = _compute_edge_activity_map(
                orig,
                sigma_divisor=edge_sigma_divisor,
                use_channel_aggregation=use_channel_aggregation,
            )
            alpha = _edge_alpha_from_activity(
                activity,
                eta=eta_hh if band == "HH" else eta_main,
                alpha_min=edge_alpha_min,
            )
            tau_map = expand_channel_stat(tau, orig) * alpha
            hf_final_levels[level][band] = soft_shrink(orig, tau_map)
            lvl_stats[f"sigma_mean_{band}"] = float(sigma.mean().item())
            lvl_stats[f"tau_mean_{band}"] = float(tau.mean().item())
            lvl_stats[f"alpha_mean_{band}"] = float(alpha.mean().item())
            lvl_stats[f"activity_mean_{band}"] = float(activity.mean().item())
        level_stats[f"L{level}"] = lvl_stats

    x_rec = waverec2_rgb_tensor(ll_final, hf_final_levels, wavelet=wavelet, target_hw=target_hw)
    meta: Dict[str, object] = {
        "levels": int(use_levels),
        "gamma_schedule": {f"L{k}": float(v) for k, v in gamma_sched.items()},
        "edge_eta_schedule": {f"L{k}": float(v) for k, v in edge_eta_sched.items()},
        "edge_eta_hh_schedule": {f"L{k}": float(v) for k, v in edge_eta_hh_sched.items()},
        "edge_sigma_divisor": float(edge_sigma_divisor),
        "edge_alpha_min": float(edge_alpha_min),
        "use_channel_aggregation": bool(use_channel_aggregation),
        "level_stats": level_stats,
        "ll_final": ll_final,
        "hf_final_levels": hf_final_levels,
        "ll_pred": ll_orig,
        "hf_policy": "edge_aware_directional_soft_shrink",
    }
    return x_rec, meta


def adaptive_multiscale_modulus_fusion(
    x_orig_chw: Tensor,
    x_pred_chw: Tensor,
    wavelet: str,
    levels: int,
    gamma_text: str,
    edge_eta_text: str,
    edge_eta_hh_text: str,
    edge_sigma_divisor: float,
    edge_alpha_min: float,
    modmax_threshold: float,
    modmax_boost_text: str,
    use_channel_aggregation: bool = True,
    eps: float = 1e-6,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Dict[str, object]]:
    ll_orig, hf_orig_levels, use_levels = wavedec2_rgb_tensor(x_orig_chw, wavelet=wavelet, levels=levels)
    ll_pred, _, _ = wavedec2_rgb_tensor(x_pred_chw, wavelet=wavelet, levels=use_levels)
    del ll_pred
    gamma_sched = resolve_ms_gamma_schedule(use_levels, gamma_text)
    edge_eta_sched = resolve_level_schedule(use_levels, edge_eta_text, fallback=0.5)
    edge_eta_hh_sched = resolve_level_schedule(use_levels, edge_eta_hh_text, fallback=0.3)
    modmax_boost_sched = resolve_level_schedule(use_levels, modmax_boost_text, fallback=0.1)

    ll_final = ll_orig
    hf_final_levels: Dict[int, Dict[str, Tensor]] = {}
    level_stats: Dict[str, Dict[str, float]] = {}
    for level in range(1, use_levels + 1):
        gamma = float(gamma_sched[level])
        eta_main = float(np.clip(edge_eta_sched[level], 0.0, 1.0))
        eta_hh = float(np.clip(edge_eta_hh_sched[level], 0.0, 1.0))
        boost = max(0.0, float(modmax_boost_sched[level]))
        modulus_map = _compute_modulus_maxima_map(
            hf_orig_levels[level]["HL"],
            hf_orig_levels[level]["LH"],
            sigma_divisor=edge_sigma_divisor,
            threshold=modmax_threshold,
        )
        lvl_stats: Dict[str, float] = {
            "gamma": gamma,
            "edge_eta": eta_main,
            "edge_eta_hh": eta_hh,
            "modmax_boost": boost,
            "modmax_threshold": float(modmax_threshold),
            "modulus_maxima_mean": float(modulus_map.mean().item()),
        }
        hf_final_levels[level] = {}
        for band in ("HL", "LH", "HH"):
            orig = hf_orig_levels[level][band]
            mad = median_abs_deviation(orig, eps=eps)
            sigma = mad / 0.6745
            n_coeff = max(2.0, float(orig.shape[-2] * orig.shape[-1]))
            tau = gamma * sigma * math.sqrt(2.0 * math.log(n_coeff))

            if band == "HH":
                activity = _compute_edge_activity_map(
                    orig,
                    sigma_divisor=edge_sigma_divisor,
                    use_channel_aggregation=use_channel_aggregation,
                )
                alpha = _edge_alpha_from_activity(activity, eta=eta_hh, alpha_min=edge_alpha_min)
                tau_map = expand_channel_stat(tau, orig) * alpha
                coeff_final = soft_shrink(orig, tau_map)
                lvl_stats[f"activity_mean_{band}"] = float(activity.mean().item())
            else:
                alpha = (1.0 - eta_main * modulus_map).clamp(min=float(edge_alpha_min), max=1.0)
                tau_map = expand_channel_stat(tau, orig) * alpha
                coeff_shrunk = soft_shrink(orig, tau_map)
                boost_map = 1.0 + boost * modulus_map
                coeff_final = coeff_shrunk * boost_map
                lvl_stats[f"boost_mean_{band}"] = float(boost_map.mean().item())

            hf_final_levels[level][band] = coeff_final
            lvl_stats[f"sigma_mean_{band}"] = float(sigma.mean().item())
            lvl_stats[f"tau_mean_{band}"] = float(tau.mean().item())
            lvl_stats[f"alpha_mean_{band}"] = float(alpha.mean().item())
        level_stats[f"L{level}"] = lvl_stats

    x_rec = waverec2_rgb_tensor(ll_final, hf_final_levels, wavelet=wavelet, target_hw=target_hw)
    meta: Dict[str, object] = {
        "levels": int(use_levels),
        "gamma_schedule": {f"L{k}": float(v) for k, v in gamma_sched.items()},
        "edge_eta_schedule": {f"L{k}": float(v) for k, v in edge_eta_sched.items()},
        "edge_eta_hh_schedule": {f"L{k}": float(v) for k, v in edge_eta_hh_sched.items()},
        "modmax_boost_schedule": {f"L{k}": float(v) for k, v in modmax_boost_sched.items()},
        "edge_sigma_divisor": float(edge_sigma_divisor),
        "edge_alpha_min": float(edge_alpha_min),
        "modmax_threshold": float(modmax_threshold),
        "use_channel_aggregation": bool(use_channel_aggregation),
        "level_stats": level_stats,
        "ll_final": ll_final,
        "hf_final_levels": hf_final_levels,
        "ll_pred": ll_orig,
        "hf_policy": "wavelet_modulus_maxima_edge_preserve_boost",
    }
    return x_rec, meta


def adaptive_multiscale_guided_fusion(
    x_orig_chw: Tensor,
    x_pred_chw: Tensor,
    wavelet: str,
    levels: int,
    gamma_text: str,
    w_min: float,
    w_max: float,
    ll_gate_tau: float,
    ll_gate_gain: float,
    hf_pred_levels: str,
    hf_gate_tau: float,
    hf_gate_gain: float,
    eps: float = 1e-6,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Dict[str, object]]:
    ll_orig, hf_orig_levels, use_levels = wavedec2_rgb_tensor(x_orig_chw, wavelet=wavelet, levels=levels)
    ll_pred, hf_pred_levels_map, _ = wavedec2_rgb_tensor(x_pred_chw, wavelet=wavelet, levels=use_levels)
    gamma_sched = resolve_ms_gamma_schedule(use_levels, gamma_text)
    pred_mix_sched = resolve_level_schedule(use_levels, hf_pred_levels, fallback=0.0)

    w_lo = float(np.clip(min(w_min, w_max), 0.0, 1.0))
    w_hi = float(np.clip(max(w_min, w_max), 0.0, 1.0))
    ll_delta = ll_pred - ll_orig
    ll_scale = median_abs_deviation(ll_delta, eps=eps) / 0.6745
    ll_gate = normalized_gate_map(ll_delta, ll_scale, gate_tau=ll_gate_tau, gate_gain=ll_gate_gain)
    ll_weight = w_lo + (w_hi - w_lo) * ll_gate
    ll_final = (1.0 - ll_weight) * ll_orig + ll_weight * ll_pred

    hf_final_levels: Dict[int, Dict[str, Tensor]] = {}
    level_stats: Dict[str, Dict[str, float]] = {}
    for level in range(1, use_levels + 1):
        gamma = float(gamma_sched[level])
        pred_mix_base = float(np.clip(pred_mix_sched[level], 0.0, 1.0))
        lvl_stats: Dict[str, float] = {"gamma": gamma, "pred_mix_base": pred_mix_base}
        hf_final_levels[level] = {}
        for band in ("HL", "LH", "HH"):
            orig = hf_orig_levels[level][band]
            pred = hf_pred_levels_map[level][band]

            mad_orig = median_abs_deviation(orig, eps=eps)
            sigma_orig = mad_orig / 0.6745
            n_coeff = max(2.0, float(orig.shape[-2] * orig.shape[-1]))
            tau = gamma * sigma_orig * math.sqrt(2.0 * math.log(n_coeff))
            orig_denoised = soft_shrink(orig, tau)
            pred_denoised = soft_shrink(pred, tau)

            delta_scale = median_abs_deviation(pred - orig, eps=eps) / 0.6745
            hf_gate = normalized_gate_map(
                pred_denoised - orig_denoised,
                delta_scale,
                gate_tau=hf_gate_tau,
                gate_gain=hf_gate_gain,
            )
            sign_agree = (orig * pred > 0).float()
            stronger_pred = (pred_denoised.abs() > orig_denoised.abs()).float()
            mix_map = pred_mix_base * hf_gate * sign_agree * stronger_pred
            hf_final = orig_denoised + mix_map * (pred_denoised - orig_denoised)
            hf_final_levels[level][band] = hf_final

            lvl_stats[f"sigma_mean_{band}"] = float(sigma_orig.mean().item())
            lvl_stats[f"tau_mean_{band}"] = float(tau.mean().item())
            lvl_stats[f"gate_mean_{band}"] = float(hf_gate.mean().item())
            lvl_stats[f"mix_mean_{band}"] = float(mix_map.mean().item())
            lvl_stats[f"sign_agree_ratio_{band}"] = float(sign_agree.mean().item())
        level_stats[f"L{level}"] = lvl_stats

    x_rec = waverec2_rgb_tensor(ll_final, hf_final_levels, wavelet=wavelet, target_hw=target_hw)
    meta: Dict[str, object] = {
        "levels": int(use_levels),
        "gamma_schedule": {f"L{k}": float(v) for k, v in gamma_sched.items()},
        "pred_mix_schedule": {f"L{k}": float(v) for k, v in pred_mix_sched.items()},
        "ll_weight_bounds": {"min": w_lo, "max": w_hi},
        "ll_gate_tau": float(ll_gate_tau),
        "ll_gate_gain": float(ll_gate_gain),
        "hf_gate_tau": float(hf_gate_tau),
        "hf_gate_gain": float(hf_gate_gain),
        "ll_weight_mean": float(ll_weight.mean().item()),
        "ll_gate_mean": float(ll_gate.mean().item()),
        "level_stats": level_stats,
        "ll_final": ll_final,
        "hf_final_levels": hf_final_levels,
        "ll_pred": ll_pred,
        "hf_policy": "orig_soft_shrink_plus_guided_pred_residual",
    }
    return x_rec, meta


def adaptive_multiscale_w2lite_fusion(
    x_orig_chw: Tensor,
    x_pred_chw: Tensor,
    wavelet: str,
    levels: int,
    gamma_text: str,
    ll_alpha: float,
    hf_mix_text: str,
    eps: float = 1e-6,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Dict[str, object]]:
    ll_orig, hf_orig_levels, use_levels = wavedec2_rgb_tensor(x_orig_chw, wavelet=wavelet, levels=levels)
    ll_pred, hf_pred_levels_map, _ = wavedec2_rgb_tensor(x_pred_chw, wavelet=wavelet, levels=use_levels)
    gamma_sched = resolve_ms_gamma_schedule(use_levels, gamma_text)
    pred_mix_sched = resolve_level_schedule(use_levels, hf_mix_text, fallback=1.0)

    ll_a = float(np.clip(ll_alpha, 0.0, 1.0))
    ll_final = (1.0 - ll_a) * ll_pred + ll_a * ll_orig

    hf_final_levels: Dict[int, Dict[str, Tensor]] = {}
    level_stats: Dict[str, Dict[str, float]] = {}
    for level in range(1, use_levels + 1):
        gamma = float(gamma_sched[level])
        pred_mix = float(np.clip(pred_mix_sched[level], 0.0, 1.0))
        lvl_stats: Dict[str, float] = {"gamma": gamma, "pred_mix": pred_mix}
        hf_final_levels[level] = {}
        for band in ("HL", "LH", "HH"):
            orig = hf_orig_levels[level][band]
            pred = hf_pred_levels_map[level][band]

            sigma_orig = median_abs_deviation(orig, eps=eps) / 0.6745
            sigma_pred = median_abs_deviation(pred, eps=eps) / 0.6745
            n_coeff = max(2.0, float(orig.shape[-2] * orig.shape[-1]))
            base = math.sqrt(2.0 * math.log(n_coeff))
            tau_orig = gamma * sigma_orig * base
            tau_pred = gamma * sigma_pred * base

            orig_denoised = soft_shrink(orig, tau_orig)
            pred_denoised = soft_shrink(pred, tau_pred)
            hf_final_levels[level][band] = (1.0 - pred_mix) * orig_denoised + pred_mix * pred_denoised

            lvl_stats[f"sigma_mean_orig_{band}"] = float(sigma_orig.mean().item())
            lvl_stats[f"sigma_mean_pred_{band}"] = float(sigma_pred.mean().item())
            lvl_stats[f"tau_mean_orig_{band}"] = float(tau_orig.mean().item())
            lvl_stats[f"tau_mean_pred_{band}"] = float(tau_pred.mean().item())
        level_stats[f"L{level}"] = lvl_stats

    x_rec = waverec2_rgb_tensor(ll_final, hf_final_levels, wavelet=wavelet, target_hw=target_hw)
    meta: Dict[str, object] = {
        "levels": int(use_levels),
        "gamma_schedule": {f"L{k}": float(v) for k, v in gamma_sched.items()},
        "pred_mix_schedule": {f"L{k}": float(v) for k, v in pred_mix_sched.items()},
        "ll_alpha": float(ll_a),
        "level_stats": level_stats,
        "ll_final": ll_final,
        "hf_final_levels": hf_final_levels,
        "ll_pred": ll_pred,
        "hf_policy": "c1_ll_plus_predictor_hf_multiscale_blend",
    }
    return x_rec, meta


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
