"""
Lightweight fallback for `flash_attn.flash_attn_interface`.

This module implements a subset of flash-attn API used by sony/ctm so we can
run without compiling CUDA flash-attn wheels.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _to_int(x: torch.Tensor | int) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fallback implementation using torch scaled_dot_product_attention.

    Expected input:
    - qkv: [total_tokens, 3, num_heads, head_dim]
    - cu_seqlens: [batch + 1], cumulative sequence offsets
    """
    if qkv.ndim != 4 or qkv.shape[1] != 3:
        raise ValueError(f"Expected qkv shape [T,3,H,D], got {tuple(qkv.shape)}")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"Expected cu_seqlens shape [B+1], got {tuple(cu_seqlens.shape)}")

    bsz = cu_seqlens.shape[0] - 1
    outs = []
    scale = softmax_scale

    for i in range(bsz):
        start = _to_int(cu_seqlens[i])
        end = _to_int(cu_seqlens[i + 1])
        if end <= start:
            continue

        # [S, H, D] -> [1, H, S, D]
        q = qkv[start:end, 0].permute(1, 0, 2).unsqueeze(0)
        k = qkv[start:end, 1].permute(1, 0, 2).unsqueeze(0)
        v = qkv[start:end, 2].permute(1, 0, 2).unsqueeze(0)

        if scale is not None:
            q = q * float(scale)

        # dropout disabled in inference fallback.
        o = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0 if not q.requires_grad else float(dropout_p),
            is_causal=causal,
        )
        # [1, H, S, D] -> [S, H, D]
        o = o.squeeze(0).permute(1, 0, 2).contiguous()
        outs.append(o)

    if outs:
        out = torch.cat(outs, dim=0)
    else:
        out = torch.empty((0, qkv.shape[2], qkv.shape[3]), device=qkv.device, dtype=qkv.dtype)

    if return_attn_probs:
        # Keep API compatibility; probabilities not exposed in fallback.
        empty = torch.empty(0, device=qkv.device, dtype=qkv.dtype)
        return out, empty, empty
    return out

