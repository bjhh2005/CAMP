from __future__ import annotations

import torch
import torch.nn.functional as F


class GaussianX0Predictor:
    """Dependency-free debug purifier backend.

    It is not a research model. Use it only to verify dataset/config/CLI wiring
    before switching to a real consistency model checkpoint on the server.
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, mix: float = 0.6, device: str = "cpu") -> None:
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
    def __call__(self, x_t: torch.Tensor, _time) -> torch.Tensor:
        x_t = x_t.to(self.device)
        kernel = self._kernel(x_t.shape[1], device=x_t.device, dtype=x_t.dtype)
        denoised = F.conv2d(x_t, kernel, padding=self.kernel_size // 2, groups=x_t.shape[1])
        return torch.lerp(x_t, denoised, self.mix).clamp(0.0, 1.0)

