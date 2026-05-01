from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pywt
import torch
import torch.nn.functional as F


Tensor = torch.Tensor
WAVELET_MODE = "reflect"
BANDS = ("HL", "LH", "HH")


def _level_value(values: List[float], level: int, fallback: float) -> float:
    if not values:
        return float(fallback)
    idx = min(max(level - 1, 0), len(values) - 1)
    return float(values[idx])


def _to_numpy_chw(x: Tensor) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError("Expected CHW tensor")
    return x.detach().cpu().float().numpy()


def wavedec2_chw(x_chw: Tensor, wavelet: str, levels: int):
    x_np = _to_numpy_chw(x_chw)
    wave = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(min(int(x_chw.shape[-2]), int(x_chw.shape[-1])), wave.dec_len)
    use_levels = max(1, min(int(levels), int(max_level)))
    channel_coeffs = [
        pywt.wavedec2(x_np[channel], wavelet=wavelet, mode=WAVELET_MODE, level=use_levels)
        for channel in range(x_np.shape[0])
    ]
    return channel_coeffs, use_levels


def waverec2_chw(channel_coeffs, wavelet: str, target_hw: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> Tensor:
    rec_list = []
    for coeffs in channel_coeffs:
        rec = pywt.waverec2(coeffs, wavelet=wavelet, mode=WAVELET_MODE)
        rec_list.append(rec)
    rec_t = torch.from_numpy(np.stack(rec_list, axis=0)).to(device=device, dtype=dtype)
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    rec_t = rec_t[:, :target_h, :target_w]
    pad_h = max(0, target_h - rec_t.shape[-2])
    pad_w = max(0, target_w - rec_t.shape[-1])
    if pad_h > 0 or pad_w > 0:
        rec_t = F.pad(rec_t, (0, pad_w, 0, pad_h), mode="replicate")
    return rec_t


def reweight_high_frequency(
    x: Tensor,
    wavelet: str,
    levels: int,
    highfreq_scales: List[float],
    bands: Optional[Iterable[str]] = None,
    lowfreq_scale: float = 1.0,
) -> Tensor:
    """Apply per-level wavelet gains to a BCHW tensor and reconstruct it."""

    if x.ndim != 4:
        raise ValueError("reweight_high_frequency expects BCHW tensor")
    active_bands = set(bands or BANDS)
    out = []
    for sample in x:
        coeffs, use_levels = wavedec2_chw(sample, wavelet=wavelet, levels=levels)
        new_coeffs = []
        for channel_coeffs in coeffs:
            current = [channel_coeffs[0] * float(lowfreq_scale)]
            for idx in range(1, len(channel_coeffs)):
                # pywt order after cA is coarse->fine. Map to level 1=fine.
                level = use_levels - idx + 1
                scale = _level_value(highfreq_scales, level=level, fallback=1.0)
                bands_tuple = []
                for band_name, band_coeff in zip(BANDS, channel_coeffs[idx]):
                    if band_name in active_bands:
                        bands_tuple.append(band_coeff * scale)
                    else:
                        bands_tuple.append(band_coeff)
                current.append(tuple(bands_tuple))
            new_coeffs.append(current)
        out.append(waverec2_chw(new_coeffs, wavelet=wavelet, target_hw=x.shape[-2:], device=x.device, dtype=x.dtype))
    return torch.stack(out, dim=0)


def enhance_noise_estimate(
    z_hat_minus: Tensor,
    wavelet: str,
    levels: int,
    gains: List[float],
    bands: Optional[Iterable[str]] = None,
) -> Tensor:
    return reweight_high_frequency(
        z_hat_minus,
        wavelet=wavelet,
        levels=levels,
        highfreq_scales=gains,
        bands=bands,
        lowfreq_scale=1.0,
    )


def attenuate_bp_guidance(
    bp_guidance: Tensor,
    wavelet: str,
    levels: int,
    highfreq_scales: List[float],
    lowfreq_scale: float,
    bands: Optional[Iterable[str]] = None,
) -> Tensor:
    return reweight_high_frequency(
        bp_guidance,
        wavelet=wavelet,
        levels=levels,
        highfreq_scales=highfreq_scales,
        bands=bands,
        lowfreq_scale=lowfreq_scale,
    )

