"""
Compatibility shim for legacy import:
    from xformers.components.attention import ScaledDotProduct
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProduct(nn.Module):
    def __init__(
        self,
        dropout: float = 0.0,
        causal: bool = False,
        scale: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.causal = bool(causal)
        self.scale = scale

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, att_mask=None, **kwargs) -> torch.Tensor:
        # Accept [B,N,H,D] and [B,H,N,D]-like inputs.
        def to_bhnd(x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 4:
                raise ValueError(f"ScaledDotProduct expects 4D tensors, got {x.shape}")
            # Heuristic: if dim1 looks like sequence, assume [B,N,H,D].
            if x.shape[1] >= x.shape[2]:
                return x.permute(0, 2, 1, 3).contiguous()
            return x

        q_bhnd = to_bhnd(q)
        k_bhnd = to_bhnd(k)
        v_bhnd = to_bhnd(v)

        scale = self.scale
        if scale is None:
            scale = 1.0 / math.sqrt(float(q_bhnd.shape[-1]))

        q_bhnd = q_bhnd * float(scale)
        out = F.scaled_dot_product_attention(
            q_bhnd,
            k_bhnd,
            v_bhnd,
            attn_mask=att_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        # Return [B,N,H,D] to match xformers-style conventions.
        return out.permute(0, 2, 1, 3).contiguous()

