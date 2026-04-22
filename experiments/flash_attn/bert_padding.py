"""
Fallback implementation for `flash_attn.bert_padding`.

Provides:
- unpad_input
- pad_input
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Args:
      hidden_states: [B, S, ...]
      attention_mask: [B, S], 1/True means keep token.

    Returns:
      hidden_states_unpad: [N_valid, ...]
      indices: [N_valid] flattened token indices
      cu_seqlens: [B+1] int32 prefix sums
      max_seqlen_in_batch: int
    """
    if hidden_states.ndim < 2:
        raise ValueError(f"hidden_states must be at least 2D, got {hidden_states.shape}")
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be [B,S], got {attention_mask.shape}")
    if hidden_states.shape[0] != attention_mask.shape[0] or hidden_states.shape[1] != attention_mask.shape[1]:
        raise ValueError(
            "hidden_states and attention_mask batch/sequence dims must match, "
            f"got {hidden_states.shape[:2]} vs {attention_mask.shape}"
        )

    bsz, seqlen = attention_mask.shape
    mask = attention_mask.to(dtype=torch.bool)
    seqlens_in_batch = mask.sum(dim=-1, dtype=torch.int32)
    max_seqlen_in_batch = int(seqlens_in_batch.max().item()) if bsz > 0 else 0
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    flat_mask = mask.reshape(-1)
    indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)

    flat_states = hidden_states.reshape(bsz * seqlen, *hidden_states.shape[2:])
    hidden_states_unpad = flat_states.index_select(0, indices)
    return hidden_states_unpad, indices, cu_seqlens, max_seqlen_in_batch


def pad_input(
    hidden_states_unpad: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    seqlen: int,
) -> torch.Tensor:
    """
    Inverse op for `unpad_input`.
    """
    total = int(batch_size) * int(seqlen)
    out_shape = (total,) + tuple(hidden_states_unpad.shape[1:])
    hidden_states = torch.zeros(
        out_shape,
        dtype=hidden_states_unpad.dtype,
        device=hidden_states_unpad.device,
    )
    hidden_states.index_copy_(0, indices, hidden_states_unpad)
    return hidden_states.view(int(batch_size), int(seqlen), *hidden_states_unpad.shape[1:])

