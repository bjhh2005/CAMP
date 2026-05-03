from __future__ import annotations

from typing import List

import torch

from .config import ScheduleConfig


def validate_schedule_config(config: ScheduleConfig) -> None:
    num_steps = int(config.num_diffusion_steps)
    sampling_steps = int(config.sampling_steps)
    iN = int(config.iN)
    beta_start = float(config.beta_start)
    beta_end = float(config.beta_end)
    gamma = float(config.gamma)
    eta = float(config.eta)
    sigma_min = float(config.sigma_min)
    sigma_max = float(config.sigma_max)

    if num_steps <= 0:
        raise ValueError("schedule.num_diffusion_steps must be positive")
    if sampling_steps <= 0:
        raise ValueError("schedule.sampling_steps must be positive")
    if iN < 0 or iN + 1 >= num_steps:
        raise ValueError(
            "schedule.iN must satisfy 0 <= iN + 1 < num_diffusion_steps; "
            f"got iN={iN}, num_diffusion_steps={num_steps}"
        )
    if not (0.0 < beta_start < 1.0 and 0.0 < beta_end < 1.0):
        raise ValueError("schedule.beta_start and beta_end must be in (0, 1)")
    if beta_start > beta_end:
        raise ValueError("schedule.beta_start must be <= beta_end")
    if gamma < 0.0:
        raise ValueError("schedule.gamma must be non-negative")
    if not (0.0 <= eta <= 1.0):
        raise ValueError("schedule.eta must be in [0, 1]")
    if sigma_min <= 0.0 or sigma_max <= 0.0:
        raise ValueError("schedule.sigma_min and sigma_max must be positive")
    if sigma_min > sigma_max:
        raise ValueError("schedule.sigma_min must be <= sigma_max")


def build_betas(config: ScheduleConfig, device: torch.device) -> torch.Tensor:
    validate_schedule_config(config)
    return torch.linspace(
        float(config.beta_start),
        float(config.beta_end),
        steps=int(config.num_diffusion_steps),
        device=device,
        dtype=torch.float32,
    )


def compute_alpha_from_betas(betas: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if betas.ndim != 1 or betas.numel() == 0:
        raise ValueError("betas must be a non-empty 1D tensor")
    if torch.any((t.long() + 1) < 0) or torch.any((t.long() + 1) > betas.numel()):
        raise IndexError(f"t index out of range for betas length {betas.numel()}: {t}")
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
