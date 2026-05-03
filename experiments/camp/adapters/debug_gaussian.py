from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import PredictionContext


class DebugGaussianBackend:
    name = "debug_gaussian"

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, mix: float = 0.6, device: str = "cpu", **_) -> None:
        self.kernel_size = int(kernel_size) if int(kernel_size) % 2 == 1 else int(kernel_size) + 1
        self.sigma = float(sigma)
        self.mix = float(max(0.0, min(1.0, mix)))
        self.device = torch.device(device)

    def _kernel(self, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = torch.arange(self.kernel_size, device=device, dtype=dtype) - (self.kernel_size - 1) / 2.0
        g = torch.exp(-(coords**2) / (2.0 * self.sigma * self.sigma))
        g = g / g.sum().clamp_min(1e-8)
        kernel = torch.outer(g, g)
        kernel = kernel / kernel.sum().clamp_min(1e-8)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(channels, 1, 1, 1)

    @torch.no_grad()
    def predict_x0(self, x_t: torch.Tensor, context: PredictionContext) -> torch.Tensor:
        del context
        x_t = x_t.to(self.device)
        x01 = ((x_t + 1.0) / 2.0).clamp(0.0, 1.0)
        kernel = self._kernel(x01.shape[1], device=x01.device, dtype=x01.dtype)
        denoised = F.conv2d(x01, kernel, padding=self.kernel_size // 2, groups=x01.shape[1])
        x0 = torch.lerp(x01, denoised, self.mix).clamp(0.0, 1.0)
        return x0 * 2.0 - 1.0

