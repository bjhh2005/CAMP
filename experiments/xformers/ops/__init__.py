"""
Minimal fallback for `xformers.ops.memory_efficient_attention`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias=None,
    p: float = 0.0,
    scale=None,
) -> torch.Tensor:
    # Expect [B, N, H, D] or [B, H, N, D]-like tensors in CTM usage.
    # We conservatively map to [B, H, N, D] for SDPA.
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("memory_efficient_attention expects 4D tensors")

    def to_bhnd(x: torch.Tensor) -> torch.Tensor:
        # Heuristic: if dim1 equals key dim1 sequence length, assume [B,N,H,D].
        if x.shape[1] == key.shape[1]:
            return x.permute(0, 2, 1, 3).contiguous()
        return x

    q = to_bhnd(query)
    k = to_bhnd(key)
    v = to_bhnd(value)

    if scale is not None:
        q = q * float(scale)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=float(p), is_causal=False)
    # Return in [B,N,H,D] form expected by most xformers callsites.
    return out.permute(0, 2, 1, 3).contiguous()

