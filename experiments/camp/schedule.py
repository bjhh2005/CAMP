from __future__ import annotations

from typing import List

import torch

from .config import ScheduleConfig


def build_betas(config: ScheduleConfig, device: torch.device) -> torch.Tensor:
    return torch.linspace(
        float(config.beta_start),
        float(config.beta_end),
        steps=int(config.num_diffusion_steps),
        device=device,
        dtype=torch.float32,
    )


def compute_alpha_from_betas(betas: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    beta = torch.cat([torch.zeros(1, device=betas.device, dtype=betas.dtype), betas], dim=0)
    return (1.0 - beta).cumprod(dim=0).index_select(0, t.long() + 1).view(1)


def get_next_alpha(prev_alpha: torch.Tensor, gamma: float) -> torch.Tensor:
    return torch.clamp(prev_alpha * (1.0 + float(gamma)), 0.0, 0.9999)


def build_alpha_schedule(config: ScheduleConfig, betas: torch.Tensor, device: torch.device) -> List[torch.Tensor]:
    t0 = torch.ones(1, device=device, dtype=torch.long) * int(config.iN + 1)
    alphas = [compute_alpha_from_betas(betas, t0)]
    for _ in range(int(config.sampling_steps) - 1):
        alphas.append(get_next_alpha(alphas[-1], config.gamma).reshape(1))
    return alphas


def alpha_to_nearest_t_index(betas: torch.Tensor, alpha: torch.Tensor) -> int:
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    idx = torch.argmin(torch.abs(alpha_bars - alpha.reshape(()))).item()
    return int(idx)
