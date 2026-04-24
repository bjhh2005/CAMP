import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from .wgcp_predictor import add_noise, run_predictor
    from .wgcp_utils import (
        coeff_to_vis,
        compute_metrics,
        ensure_dir,
        make_triplet_panel,
        pil_to_tensor,
        save_tensor_image,
    )
    from .wgcp_wavelet import (
        adaptive_multiscale_edge_fusion,
        adaptive_multiscale_fusion,
        adaptive_multiscale_guided_fusion,
        adaptive_multiscale_modulus_fusion,
        dwt2_rgb_tensor,
        fuse_high_frequency,
        idwt2_rgb_tensor,
    )
except ImportError:
    from wgcp_predictor import add_noise, run_predictor
    from wgcp_utils import (
        coeff_to_vis,
        compute_metrics,
        ensure_dir,
        make_triplet_panel,
        pil_to_tensor,
        save_tensor_image,
    )
    from wgcp_wavelet import (
        adaptive_multiscale_edge_fusion,
        adaptive_multiscale_fusion,
        adaptive_multiscale_guided_fusion,
        adaptive_multiscale_modulus_fusion,
        dwt2_rgb_tensor,
        fuse_high_frequency,
        idwt2_rgb_tensor,
    )


Tensor = torch.Tensor


def _reconstruct_from_prediction(
    x_orig_chw: Tensor,
    x_pred_chw: Tensor,
    wavelet: str,
    target_hw: Tuple[int, int],
    replacement_mode: str,
    hf_preserve: float,
    hf_shrink: float,
    ablation_ll_source: str,
    ablation_hard_hf_source: str,
    ms_levels: int,
    ms_gamma_levels: str,
    ms_w_min: float,
    ms_w_max: float,
    ms_ll_alpha: float,
    ms_eps: float,
    ms_ll_gate_tau: float,
    ms_ll_gate_gain: float,
    ms_hf_pred_levels: str,
    ms_hf_gate_tau: float,
    ms_hf_gate_gain: float,
    ms_edge_eta_levels: str,
    ms_edge_eta_hh_levels: str,
    ms_edge_sigma_divisor: float,
    ms_edge_alpha_min: float,
    ms_edge_channel_agg: bool,
    ms_modmax_threshold: float,
    ms_modmax_boost_levels: str,
    orig_components: Optional[Tuple[Tensor, Dict[str, Tensor]]] = None,
    ll_source_override: Optional[str] = None,
    forced_ll_anchor: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, object]]:
    if orig_components is None:
        ll_orig, hf_orig = dwt2_rgb_tensor(x_orig_chw, wavelet=wavelet)
    else:
        ll_orig, hf_orig = orig_components

    ll_hat, hf_hat = dwt2_rgb_tensor(x_pred_chw, wavelet=wavelet)
    adaptive_meta: Optional[Dict[str, object]] = None
    ll_source = ll_source_override or ablation_ll_source

    if replacement_mode == "hard":
        ll_anchor = forced_ll_anchor if forced_ll_anchor is not None else (ll_orig if ll_source == "orig" else ll_hat)
        hf_selected = hf_hat if ablation_hard_hf_source == "pred" else hf_orig
        x_rec = idwt2_rgb_tensor(ll_anchor, hf_selected, wavelet=wavelet, target_hw=target_hw)
    elif replacement_mode == "fused":
        ll_anchor = forced_ll_anchor if forced_ll_anchor is not None else (ll_orig if ll_source == "orig" else ll_hat)
        hf_selected = fuse_high_frequency(hf_orig, hf_hat, hf_preserve=hf_preserve, hf_shrink=hf_shrink)
        x_rec = idwt2_rgb_tensor(ll_anchor, hf_selected, wavelet=wavelet, target_hw=target_hw)
    elif replacement_mode == "adaptive_ms":
        x_rec, adaptive_meta = adaptive_multiscale_fusion(
            x_orig_chw=x_orig_chw,
            x_pred_chw=x_pred_chw,
            wavelet=wavelet,
            levels=ms_levels,
            gamma_text=ms_gamma_levels,
            w_min=ms_w_min,
            w_max=ms_w_max,
            ll_alpha=ms_ll_alpha,
            eps=ms_eps,
            target_hw=target_hw,
        )
        ll_anchor = adaptive_meta["ll_final"]
        hf_selected = dwt2_rgb_tensor(x_rec, wavelet=wavelet)[1]
    elif replacement_mode == "adaptive_ms_guided":
        x_rec, adaptive_meta = adaptive_multiscale_guided_fusion(
            x_orig_chw=x_orig_chw,
            x_pred_chw=x_pred_chw,
            wavelet=wavelet,
            levels=ms_levels,
            gamma_text=ms_gamma_levels,
            w_min=ms_w_min,
            w_max=ms_w_max,
            ll_gate_tau=ms_ll_gate_tau,
            ll_gate_gain=ms_ll_gate_gain,
            hf_pred_levels=ms_hf_pred_levels,
            hf_gate_tau=ms_hf_gate_tau,
            hf_gate_gain=ms_hf_gate_gain,
            eps=ms_eps,
            target_hw=target_hw,
        )
        ll_anchor = adaptive_meta["ll_final"]
        hf_selected = dwt2_rgb_tensor(x_rec, wavelet=wavelet)[1]
    elif replacement_mode == "adaptive_ms_edge":
        x_rec, adaptive_meta = adaptive_multiscale_edge_fusion(
            x_orig_chw=x_orig_chw,
            x_pred_chw=x_pred_chw,
            wavelet=wavelet,
            levels=ms_levels,
            gamma_text=ms_gamma_levels,
            edge_eta_text=ms_edge_eta_levels,
            edge_eta_hh_text=ms_edge_eta_hh_levels,
            edge_sigma_divisor=ms_edge_sigma_divisor,
            edge_alpha_min=ms_edge_alpha_min,
            use_channel_aggregation=ms_edge_channel_agg,
            eps=ms_eps,
            target_hw=target_hw,
        )
        ll_anchor = adaptive_meta["ll_final"]
        hf_selected = dwt2_rgb_tensor(x_rec, wavelet=wavelet)[1]
    elif replacement_mode == "adaptive_ms_modmax":
        x_rec, adaptive_meta = adaptive_multiscale_modulus_fusion(
            x_orig_chw=x_orig_chw,
            x_pred_chw=x_pred_chw,
            wavelet=wavelet,
            levels=ms_levels,
            gamma_text=ms_gamma_levels,
            edge_eta_text=ms_edge_eta_levels,
            edge_eta_hh_text=ms_edge_eta_hh_levels,
            edge_sigma_divisor=ms_edge_sigma_divisor,
            edge_alpha_min=ms_edge_alpha_min,
            modmax_threshold=ms_modmax_threshold,
            modmax_boost_text=ms_modmax_boost_levels,
            use_channel_aggregation=ms_edge_channel_agg,
            eps=ms_eps,
            target_hw=target_hw,
        )
        ll_anchor = adaptive_meta["ll_final"]
        hf_selected = dwt2_rgb_tensor(x_rec, wavelet=wavelet)[1]
    else:
        raise ValueError(f"Unknown replacement_mode: {replacement_mode}")

    meta: Dict[str, object] = {
        "ll_orig": ll_orig,
        "hf_orig": hf_orig,
        "ll_hat": ll_hat,
        "hf_hat": hf_hat,
        "ll_anchor": ll_anchor,
        "hf_selected": hf_selected,
        "adaptive_ms": adaptive_meta,
    }
    return x_rec, meta


def compute_patch_padding(length: int, patch: int, stride: int) -> int:
    if length <= patch:
        return patch - length
    rem = (length - patch) % stride
    return 0 if rem == 0 else (stride - rem)


def make_patch_gaussian_weight(
    patch_size: int,
    sigma_pixels: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    sigma = float(sigma_pixels) if sigma_pixels > 0 else max(1.0, float(patch_size) / 6.0)
    coords = torch.arange(patch_size, device=device, dtype=dtype) - (patch_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g2d = torch.outer(g, g)
    g2d = g2d / g2d.max().clamp_min(1e-8)
    return g2d.view(1, 1, patch_size, patch_size).clamp_min(1e-6)


def unfold_patches(
    x: Tensor,
    patch_size: int,
    stride: int,
    pad_mode: str,
) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("unfold_patches expects x with shape [1, C, H, W]")
    height, width = int(x.shape[-2]), int(x.shape[-1])
    pad_h = compute_patch_padding(height, patch_size, stride)
    pad_w = compute_patch_padding(width, patch_size, stride)
    mode = pad_mode
    if mode == "reflect" and (height <= 1 or width <= 1 or pad_h >= height or pad_w >= width):
        mode = "replicate"
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    h_pad, w_pad = int(x_pad.shape[-2]), int(x_pad.shape[-1])
    cols = F.unfold(x_pad, kernel_size=patch_size, stride=stride)
    num_patches = int(cols.shape[-1])
    channels = int(x.shape[1])
    patches = cols.squeeze(0).transpose(0, 1).contiguous().view(num_patches, channels, patch_size, patch_size)
    return patches, (h_pad, w_pad), (pad_h, pad_w)


def fold_patches_weighted(
    patches: Tensor,
    output_hw: Tuple[int, int],
    patch_size: int,
    stride: int,
    weight_patch: Tensor,
    target_hw: Tuple[int, int],
) -> Tensor:
    if patches.ndim != 4:
        raise ValueError("fold_patches_weighted expects patches with shape [L, C, K, K]")
    num_patches, channels, _, _ = patches.shape
    weighted = patches * weight_patch
    cols = weighted.reshape(num_patches, channels * patch_size * patch_size).transpose(0, 1).unsqueeze(0)
    recon = F.fold(cols, output_size=output_hw, kernel_size=patch_size, stride=stride)

    weight_cols = (
        weight_patch.expand(num_patches, 1, patch_size, patch_size)
        .reshape(num_patches, patch_size * patch_size)
        .transpose(0, 1)
        .unsqueeze(0)
    )
    weight_sum = F.fold(weight_cols, output_size=output_hw, kernel_size=patch_size, stride=stride).clamp_min(1e-6)
    recon = recon / weight_sum.expand(1, channels, output_hw[0], output_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    return recon[:, :, :target_h, :target_w].contiguous()


def _purify_adv_tensor_global(
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
    ms_levels: int = 3,
    ms_gamma_levels: str = "1.6,1.2,0.9",
    ms_w_min: float = 0.05,
    ms_w_max: float = 0.95,
    ms_ll_alpha: float = 0.08,
    ms_eps: float = 1e-6,
    ms_ll_gate_tau: float = 0.75,
    ms_ll_gate_gain: float = 4.0,
    ms_hf_pred_levels: str = "0.20,0.12,0.08",
    ms_hf_gate_tau: float = 0.5,
    ms_hf_gate_gain: float = 4.0,
    ms_edge_eta_levels: str = "0.5,0.5,0.5",
    ms_edge_eta_hh_levels: str = "0.3,0.3,0.3",
    ms_edge_sigma_divisor: float = 30.0,
    ms_edge_alpha_min: float = 0.1,
    ms_edge_channel_agg: bool = True,
    ms_modmax_threshold: float = 0.15,
    ms_modmax_boost_levels: str = "0.10,0.08,0.05",
) -> Tuple[Tensor, Tensor, Dict[str, object], float]:
    if x_adv.ndim != 4 or x_adv.shape[0] != 1:
        raise ValueError("x_adv must have shape [1, C, H, W]")
    target_hw = (int(x_adv.shape[-2]), int(x_adv.shape[-1]))
    orig_components = dwt2_rgb_tensor(x_adv[0], wavelet=wavelet)

    alpha_t = float(alpha_bars[t_star].item())
    x_t = add_noise(x_adv, alpha_t)

    t0 = time.perf_counter()
    x0_hat = run_predictor(predictor, x_t, t_star, predictor_image_size=predictor_image_size)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    x_first, first_meta = _reconstruct_from_prediction(
        x_orig_chw=x_adv[0],
        x_pred_chw=x0_hat[0],
        wavelet=wavelet,
        target_hw=target_hw,
        replacement_mode=replacement_mode,
        hf_preserve=hf_preserve,
        hf_shrink=hf_shrink,
        ablation_ll_source=ablation_ll_source,
        ablation_hard_hf_source=ablation_hard_hf_source,
        ms_levels=ms_levels,
        ms_gamma_levels=ms_gamma_levels,
        ms_w_min=ms_w_min,
        ms_w_max=ms_w_max,
        ms_ll_alpha=ms_ll_alpha,
        ms_eps=ms_eps,
        ms_ll_gate_tau=ms_ll_gate_tau,
        ms_ll_gate_gain=ms_ll_gate_gain,
        ms_hf_pred_levels=ms_hf_pred_levels,
        ms_hf_gate_tau=ms_hf_gate_tau,
        ms_hf_gate_gain=ms_hf_gate_gain,
        ms_edge_eta_levels=ms_edge_eta_levels,
        ms_edge_eta_hh_levels=ms_edge_eta_hh_levels,
        ms_edge_sigma_divisor=ms_edge_sigma_divisor,
        ms_edge_alpha_min=ms_edge_alpha_min,
        ms_edge_channel_agg=ms_edge_channel_agg,
        ms_modmax_threshold=ms_modmax_threshold,
        ms_modmax_boost_levels=ms_modmax_boost_levels,
        orig_components=orig_components,
    )

    x_corrected = x_first.to(x_adv.device).unsqueeze(0).clamp(0.0, 1.0)
    current = x_corrected.clone()
    loop_records: List[Dict[str, object]] = []
    for t_step in loop_steps:
        alpha_k = float(alpha_bars[t_step].item())
        x_tk = add_noise(current, alpha_k)
        x_hat_k = run_predictor(predictor, x_tk, t_step, predictor_image_size=predictor_image_size)
        x_k, step_meta = _reconstruct_from_prediction(
            x_orig_chw=x_adv[0],
            x_pred_chw=x_hat_k[0],
            wavelet=wavelet,
            target_hw=target_hw,
            replacement_mode=replacement_mode,
            hf_preserve=hf_preserve,
            hf_shrink=hf_shrink,
            ablation_ll_source=ablation_ll_source,
            ablation_hard_hf_source=ablation_hard_hf_source,
            ms_levels=ms_levels,
            ms_gamma_levels=ms_gamma_levels,
            ms_w_min=ms_w_min,
            ms_w_max=ms_w_max,
            ms_ll_alpha=ms_ll_alpha,
            ms_eps=ms_eps,
            ms_ll_gate_tau=ms_ll_gate_tau,
            ms_ll_gate_gain=ms_ll_gate_gain,
            ms_hf_pred_levels=ms_hf_pred_levels,
            ms_hf_gate_tau=ms_hf_gate_tau,
            ms_hf_gate_gain=ms_hf_gate_gain,
            ms_edge_eta_levels=ms_edge_eta_levels,
            ms_edge_eta_hh_levels=ms_edge_eta_hh_levels,
            ms_edge_sigma_divisor=ms_edge_sigma_divisor,
            ms_edge_alpha_min=ms_edge_alpha_min,
            ms_edge_channel_agg=ms_edge_channel_agg,
            ms_modmax_threshold=ms_modmax_threshold,
            ms_modmax_boost_levels=ms_modmax_boost_levels,
            orig_components=orig_components,
        )
        current = x_k.to(x_adv.device).unsqueeze(0).clamp(0.0, 1.0)
        loop_records.append(
            {
                "t_step": t_step,
                "x_tk": x_tk.detach().clone(),
                "x_hat_k": x_hat_k.detach().clone(),
                "x_corrected_k": current.detach().clone(),
                "ll_anchor_source": ablation_ll_source,
                "hf_k_selected": {key: value.detach().clone() for key, value in step_meta["hf_selected"].items()},
            }
        )

    trace: Dict[str, object] = {
        "ll_orig": first_meta["ll_orig"],
        "hf_orig": first_meta["hf_orig"],
        "x_t": x_t.detach().clone(),
        "x0_hat": x0_hat.detach().clone(),
        "ll_hat": first_meta["ll_hat"],
        "hf_hat": first_meta["hf_hat"],
        "ll_anchor": first_meta["ll_anchor"],
        "hf_selected": first_meta["hf_selected"],
        "x_corrected": x_corrected.detach().clone(),
        "loop_records": loop_records,
        "ablation": {
            "ll_source": ablation_ll_source,
            "hard_hf_source": ablation_hard_hf_source,
        },
        "adaptive_ms": first_meta["adaptive_ms"],
        "predictor_calls": 1 + len(loop_steps),
    }
    return x_corrected, current, trace, infer_ms


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
    patch_mode: bool = False,
    patch_size: int = 64,
    patch_stride: int = 32,
    patch_batch_size: int = 64,
    patch_weight_sigma: float = 0.0,
    patch_lowfreq_alpha: float = 0.1,
    patch_ll_source: str = "hat",
    patch_pad_mode: str = "reflect",
    ms_levels: int = 3,
    ms_gamma_levels: str = "1.6,1.2,0.9",
    ms_w_min: float = 0.05,
    ms_w_max: float = 0.95,
    ms_ll_alpha: float = 0.08,
    ms_eps: float = 1e-6,
    ms_ll_gate_tau: float = 0.75,
    ms_ll_gate_gain: float = 4.0,
    ms_hf_pred_levels: str = "0.20,0.12,0.08",
    ms_hf_gate_tau: float = 0.5,
    ms_hf_gate_gain: float = 4.0,
    ms_edge_eta_levels: str = "0.5,0.5,0.5",
    ms_edge_eta_hh_levels: str = "0.3,0.3,0.3",
    ms_edge_sigma_divisor: float = 30.0,
    ms_edge_alpha_min: float = 0.1,
    ms_edge_channel_agg: bool = True,
    ms_modmax_threshold: float = 0.15,
    ms_modmax_boost_levels: str = "0.10,0.08,0.05",
) -> Tuple[Tensor, Tensor, Dict[str, object], float]:
    if not patch_mode:
        return _purify_adv_tensor_global(
            x_adv=x_adv,
            alpha_bars=alpha_bars,
            predictor=predictor,
            wavelet=wavelet,
            t_star=t_star,
            loop_steps=loop_steps,
            hf_preserve=hf_preserve,
            hf_shrink=hf_shrink,
            replacement_mode=replacement_mode,
            predictor_image_size=predictor_image_size,
            ablation_ll_source=ablation_ll_source,
            ablation_hard_hf_source=ablation_hard_hf_source,
            ms_levels=ms_levels,
            ms_gamma_levels=ms_gamma_levels,
            ms_w_min=ms_w_min,
            ms_w_max=ms_w_max,
            ms_ll_alpha=ms_ll_alpha,
            ms_eps=ms_eps,
            ms_ll_gate_tau=ms_ll_gate_tau,
            ms_ll_gate_gain=ms_ll_gate_gain,
            ms_hf_pred_levels=ms_hf_pred_levels,
            ms_hf_gate_tau=ms_hf_gate_tau,
            ms_hf_gate_gain=ms_hf_gate_gain,
            ms_edge_eta_levels=ms_edge_eta_levels,
            ms_edge_eta_hh_levels=ms_edge_eta_hh_levels,
            ms_edge_sigma_divisor=ms_edge_sigma_divisor,
            ms_edge_alpha_min=ms_edge_alpha_min,
            ms_edge_channel_agg=ms_edge_channel_agg,
            ms_modmax_threshold=ms_modmax_threshold,
            ms_modmax_boost_levels=ms_modmax_boost_levels,
        )

    if x_adv.ndim != 4 or x_adv.shape[0] != 1:
        raise ValueError("x_adv must have shape [1, C, H, W]")
    if patch_size <= 1 or patch_stride <= 0 or patch_batch_size <= 0:
        raise ValueError("patch_size/patch_stride/patch_batch_size must be positive")

    target_hw = (int(x_adv.shape[-2]), int(x_adv.shape[-1]))
    ll_orig, hf_orig = dwt2_rgb_tensor(x_adv[0], wavelet=wavelet)

    x_base, _, base_trace, infer_base_ms = _purify_adv_tensor_global(
        x_adv=x_adv,
        alpha_bars=alpha_bars,
        predictor=predictor,
        wavelet=wavelet,
        t_star=t_star,
        loop_steps=[],
        hf_preserve=hf_preserve,
        hf_shrink=hf_shrink,
        replacement_mode=replacement_mode,
        predictor_image_size=predictor_image_size,
        ablation_ll_source=ablation_ll_source,
        ablation_hard_hf_source=ablation_hard_hf_source,
        ms_levels=ms_levels,
        ms_gamma_levels=ms_gamma_levels,
        ms_w_min=ms_w_min,
        ms_w_max=ms_w_max,
        ms_ll_alpha=ms_ll_alpha,
        ms_eps=ms_eps,
        ms_ll_gate_tau=ms_ll_gate_tau,
        ms_ll_gate_gain=ms_ll_gate_gain,
        ms_hf_pred_levels=ms_hf_pred_levels,
        ms_hf_gate_tau=ms_hf_gate_tau,
        ms_hf_gate_gain=ms_hf_gate_gain,
        ms_edge_eta_levels=ms_edge_eta_levels,
        ms_edge_eta_hh_levels=ms_edge_eta_hh_levels,
        ms_edge_sigma_divisor=ms_edge_sigma_divisor,
        ms_edge_alpha_min=ms_edge_alpha_min,
        ms_edge_channel_agg=ms_edge_channel_agg,
        ms_modmax_threshold=ms_modmax_threshold,
        ms_modmax_boost_levels=ms_modmax_boost_levels,
    )

    alpha_t = float(alpha_bars[t_star].item())
    patches_adv, output_hw, _ = unfold_patches(
        x_adv,
        patch_size=patch_size,
        stride=patch_stride,
        pad_mode=patch_pad_mode,
    )
    num_patches = int(patches_adv.shape[0])

    patch_outputs: List[Tensor] = []
    infer_patch_ms = 0.0
    for start in range(0, num_patches, patch_batch_size):
        end = min(num_patches, start + patch_batch_size)
        batch_adv = patches_adv[start:end].to(x_adv.device)
        batch_t = add_noise(batch_adv, alpha_t)
        t0 = time.perf_counter()
        batch_hat = run_predictor(predictor, batch_t, t_star, predictor_image_size=predictor_image_size)
        infer_patch_ms += (time.perf_counter() - t0) * 1000.0

        rec_batch: List[Tensor] = []
        for idx in range(int(batch_adv.shape[0])):
            rec_p, _ = _reconstruct_from_prediction(
                x_orig_chw=batch_adv[idx],
                x_pred_chw=batch_hat[idx],
                wavelet=wavelet,
                target_hw=(patch_size, patch_size),
                replacement_mode=replacement_mode,
                hf_preserve=hf_preserve,
                hf_shrink=hf_shrink,
                ablation_ll_source=ablation_ll_source,
                ablation_hard_hf_source=ablation_hard_hf_source,
                ms_levels=ms_levels,
                ms_gamma_levels=ms_gamma_levels,
                ms_w_min=ms_w_min,
                ms_w_max=ms_w_max,
                ms_ll_alpha=ms_ll_alpha,
                ms_eps=ms_eps,
                ms_ll_gate_tau=ms_ll_gate_tau,
                ms_ll_gate_gain=ms_ll_gate_gain,
                ms_hf_pred_levels=ms_hf_pred_levels,
                ms_hf_gate_tau=ms_hf_gate_tau,
                ms_hf_gate_gain=ms_hf_gate_gain,
                ms_edge_eta_levels=ms_edge_eta_levels,
                ms_edge_eta_hh_levels=ms_edge_eta_hh_levels,
                ms_edge_sigma_divisor=ms_edge_sigma_divisor,
                ms_edge_alpha_min=ms_edge_alpha_min,
                ms_edge_channel_agg=ms_edge_channel_agg,
                ms_modmax_threshold=ms_modmax_threshold,
                ms_modmax_boost_levels=ms_modmax_boost_levels,
                ll_source_override=patch_ll_source,
            )
            rec_batch.append(rec_p.to(x_adv.device))
        patch_outputs.append(torch.stack(rec_batch, dim=0))

    patches_clean = torch.cat(patch_outputs, dim=0)
    weight_patch = make_patch_gaussian_weight(
        patch_size=patch_size,
        sigma_pixels=patch_weight_sigma,
        device=x_adv.device,
        dtype=x_adv.dtype,
    )
    x_patch = fold_patches_weighted(
        patches=patches_clean,
        output_hw=output_hw,
        patch_size=patch_size,
        stride=patch_stride,
        weight_patch=weight_patch,
        target_hw=target_hw,
    ).clamp(0.0, 1.0)

    ll_base, _ = dwt2_rgb_tensor(x_base[0], wavelet=wavelet)
    ll_patch, hf_patch = dwt2_rgb_tensor(x_patch[0], wavelet=wavelet)
    alpha_mix = float(np.clip(patch_lowfreq_alpha, 0.0, 1.0))
    ll_anchor = (1.0 - alpha_mix) * ll_base + alpha_mix * ll_patch
    x_corrected = (
        idwt2_rgb_tensor(ll_anchor, hf_patch, wavelet=wavelet, target_hw=target_hw)
        .to(x_adv.device)
        .unsqueeze(0)
        .clamp(0.0, 1.0)
    )

    current = x_corrected.clone()
    loop_records: List[Dict[str, object]] = []
    for t_step in loop_steps:
        alpha_k = float(alpha_bars[t_step].item())
        x_tk = add_noise(current, alpha_k)
        x_hat_k = run_predictor(predictor, x_tk, t_step, predictor_image_size=predictor_image_size)
        if replacement_mode in {"hard", "fused"}:
            x_k, step_meta = _reconstruct_from_prediction(
                x_orig_chw=x_adv[0],
                x_pred_chw=x_hat_k[0],
                wavelet=wavelet,
                target_hw=target_hw,
                replacement_mode=replacement_mode,
                hf_preserve=hf_preserve,
                hf_shrink=hf_shrink,
                ablation_ll_source=ablation_ll_source,
                ablation_hard_hf_source=ablation_hard_hf_source,
                ms_levels=ms_levels,
                ms_gamma_levels=ms_gamma_levels,
                ms_w_min=ms_w_min,
                ms_w_max=ms_w_max,
                ms_ll_alpha=ms_ll_alpha,
                ms_eps=ms_eps,
                ms_ll_gate_tau=ms_ll_gate_tau,
                ms_ll_gate_gain=ms_ll_gate_gain,
                ms_hf_pred_levels=ms_hf_pred_levels,
                ms_hf_gate_tau=ms_hf_gate_tau,
                ms_hf_gate_gain=ms_hf_gate_gain,
                ms_edge_eta_levels=ms_edge_eta_levels,
                ms_edge_eta_hh_levels=ms_edge_eta_hh_levels,
                ms_edge_sigma_divisor=ms_edge_sigma_divisor,
                ms_edge_alpha_min=ms_edge_alpha_min,
                ms_edge_channel_agg=ms_edge_channel_agg,
                ms_modmax_threshold=ms_modmax_threshold,
                ms_modmax_boost_levels=ms_modmax_boost_levels,
                orig_components=(ll_orig, hf_orig),
                forced_ll_anchor=ll_anchor,
            )
            current = x_k.to(x_adv.device).unsqueeze(0).clamp(0.0, 1.0)
            hf_k_selected = step_meta["hf_selected"]
        else:
            x_k, _ = _reconstruct_from_prediction(
                x_orig_chw=x_adv[0],
                x_pred_chw=x_hat_k[0],
                wavelet=wavelet,
                target_hw=target_hw,
                replacement_mode=replacement_mode,
                hf_preserve=hf_preserve,
                hf_shrink=hf_shrink,
                ablation_ll_source=ablation_ll_source,
                ablation_hard_hf_source=ablation_hard_hf_source,
                ms_levels=ms_levels,
                ms_gamma_levels=ms_gamma_levels,
                ms_w_min=ms_w_min,
                ms_w_max=ms_w_max,
                ms_ll_alpha=ms_ll_alpha,
                ms_eps=ms_eps,
                ms_ll_gate_tau=ms_ll_gate_tau,
                ms_ll_gate_gain=ms_ll_gate_gain,
                ms_hf_pred_levels=ms_hf_pred_levels,
                ms_hf_gate_tau=ms_hf_gate_tau,
                ms_hf_gate_gain=ms_hf_gate_gain,
                ms_edge_eta_levels=ms_edge_eta_levels,
                ms_edge_eta_hh_levels=ms_edge_eta_hh_levels,
                ms_edge_sigma_divisor=ms_edge_sigma_divisor,
                ms_edge_alpha_min=ms_edge_alpha_min,
                ms_edge_channel_agg=ms_edge_channel_agg,
                ms_modmax_threshold=ms_modmax_threshold,
                ms_modmax_boost_levels=ms_modmax_boost_levels,
                orig_components=(ll_orig, hf_orig),
            )
            hf_k_selected = dwt2_rgb_tensor(x_k, wavelet=wavelet)[1]
            current = x_k.to(x_adv.device).unsqueeze(0).clamp(0.0, 1.0)

        loop_records.append(
            {
                "t_step": t_step,
                "x_tk": x_tk.detach().clone(),
                "x_hat_k": x_hat_k.detach().clone(),
                "x_corrected_k": current.detach().clone(),
                "ll_anchor_source": "patch_mix",
                "hf_k_selected": {key: value.detach().clone() for key, value in hf_k_selected.items()},
            }
        )

    patch_predictor_calls = int(math.ceil(num_patches / float(patch_batch_size)))
    infer_ms_total = float(infer_base_ms + infer_patch_ms)
    trace: Dict[str, object] = {
        "ll_orig": ll_orig,
        "hf_orig": hf_orig,
        "x_t": base_trace["x_t"],
        "x0_hat": x_patch.detach().clone(),
        "ll_hat": ll_patch,
        "hf_hat": hf_patch,
        "ll_anchor": ll_anchor,
        "hf_selected": hf_patch,
        "x_corrected": x_corrected.detach().clone(),
        "loop_records": loop_records,
        "ablation": {
            "ll_source": ablation_ll_source,
            "hard_hf_source": ablation_hard_hf_source,
        },
        "patch": {
            "enabled": True,
            "patch_size": int(patch_size),
            "patch_stride": int(patch_stride),
            "patch_batch_size": int(patch_batch_size),
            "patch_weight_sigma": float(patch_weight_sigma),
            "patch_lowfreq_alpha": float(alpha_mix),
            "patch_ll_source": patch_ll_source,
            "num_patches": num_patches,
            "predictor_calls_patch": patch_predictor_calls,
            "predictor_ms_global": float(infer_base_ms),
            "predictor_ms_patch": float(infer_patch_ms),
        },
        "predictor_calls": int(1 + patch_predictor_calls + len(loop_steps)),
    }
    return x_corrected, current, trace, infer_ms_total


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
        height, width = args.resize
        img = img.resize((width, height), Image.BICUBIC)

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
        patch_mode=args.patch_mode,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        patch_batch_size=args.patch_batch_size,
        patch_weight_sigma=args.patch_weight_sigma,
        patch_lowfreq_alpha=args.patch_lowfreq_alpha,
        patch_ll_source=args.patch_ll_source,
        patch_pad_mode=args.patch_pad_mode,
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

    for idx, record in enumerate(trace["loop_records"], start=1):
        t_step = int(record["t_step"])
        x_tk = record["x_tk"]
        x_hat_k = record["x_hat_k"]
        x_corr_k = record["x_corrected_k"]
        save_tensor_image(x_tk[0].clamp(0.0, 1.0), sample_dir / f"06_loop{idx}_x_t{t_step}.png")
        save_tensor_image(x_hat_k[0], sample_dir / f"06_loop{idx}_x_hat.png")
        save_tensor_image(x_corr_k[0], sample_dir / f"06_loop{idx}_x_corrected.png")

    save_tensor_image(x_final[0], sample_dir / "07_x_final.png")
    panel = make_triplet_panel([x_adv[0], x_corrected[0], x_final[0]])
    panel.save(sample_dir / "08_compare_adv_corrected_final.png")

    metrics = compute_metrics(x_adv[0], x_final[0])
    metrics["NFE"] = int(trace.get("predictor_calls", 1 + len(loop_steps)))
    metrics["single_step_infer_ms"] = float(infer_ms)

    if args.clean_dir is not None:
        clean_path = args.clean_dir / image_path.name
        if clean_path.exists():
            clean_img = Image.open(clean_path).convert("RGB")
            if args.resize is not None:
                height, width = args.resize
                clean_img = clean_img.resize((width, height), Image.BICUBIC)
            x_clean = pil_to_tensor(clean_img)
            clean_metrics = compute_metrics(x_clean, x_final[0])
            metrics.update({f"clean_{key}": value for key, value in clean_metrics.items()})
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
        "patch_mode": bool(args.patch_mode),
        "patch_size": args.patch_size,
        "patch_stride": args.patch_stride,
        "patch_batch_size": args.patch_batch_size,
        "patch_weight_sigma": args.patch_weight_sigma,
        "patch_lowfreq_alpha": args.patch_lowfreq_alpha,
        "patch_ll_source": args.patch_ll_source,
        "patch_pad_mode": args.patch_pad_mode,
        "ms_levels": args.ms_levels,
        "ms_gamma_levels": args.ms_gamma_levels,
        "ms_w_min": args.ms_w_min,
        "ms_w_max": args.ms_w_max,
        "ms_ll_alpha": args.ms_ll_alpha,
        "ms_eps": args.ms_eps,
        "ms_ll_gate_tau": args.ms_ll_gate_tau,
        "ms_ll_gate_gain": args.ms_ll_gate_gain,
        "ms_hf_pred_levels": args.ms_hf_pred_levels,
        "ms_hf_gate_tau": args.ms_hf_gate_tau,
        "ms_hf_gate_gain": args.ms_hf_gate_gain,
        "ms_edge_eta_levels": args.ms_edge_eta_levels,
        "ms_edge_eta_hh_levels": args.ms_edge_eta_hh_levels,
        "ms_edge_sigma_divisor": args.ms_edge_sigma_divisor,
        "ms_edge_alpha_min": args.ms_edge_alpha_min,
        "ms_edge_channel_agg": bool(args.ms_edge_channel_agg),
        "ms_modmax_threshold": args.ms_modmax_threshold,
        "ms_modmax_boost_levels": args.ms_modmax_boost_levels,
        "replacement_mode": args.replacement_mode,
        "ablation_ll_source": args.ablation_ll_source,
        "ablation_hard_hf_source": args.ablation_hard_hf_source,
        "hf_preserve": args.hf_preserve,
        "hf_shrink": args.hf_shrink,
        "metrics": metrics,
    }

    with open(sample_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    return result
