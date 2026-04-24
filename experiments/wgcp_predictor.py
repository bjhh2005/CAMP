import argparse
import importlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def make_gaussian_kernel(kernel_size: int, sigma: float, channels: int, device: str) -> Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def gaussian_blur_tensor(x: Tensor, kernel_size: int, sigma: float) -> Tensor:
    _, channels, _, _ = x.shape
    kernel = make_gaussian_kernel(kernel_size, sigma, channels, str(x.device))
    padding = kernel_size // 2
    return F.conv2d(x, kernel, padding=padding, groups=channels)


class GaussianCTMPredictor:
    """Debug predictor used as CTM placeholder for mechanism verification."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, mix: float = 0.8):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.mix = float(np.clip(mix, 0.0, 1.0))

    @torch.no_grad()
    def __call__(self, x_t: Tensor, t_index: int) -> Tensor:
        del t_index
        denoised = gaussian_blur_tensor(x_t, self.kernel_size, self.sigma)
        x0_hat = torch.lerp(x_t, denoised, self.mix)
        return x0_hat.clamp(0.0, 1.0)


def build_predictor(args: argparse.Namespace, device: torch.device):
    if args.predictor_type == "gaussian":
        return GaussianCTMPredictor(kernel_size=args.kernel_size, sigma=args.sigma, mix=args.mix)

    if args.predictor_type == "module":
        if ":" not in args.predictor_module:
            raise ValueError("predictor_module must be 'package.module:ClassName'")
        module_name, class_name = args.predictor_module.split(":", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        kwargs = json.loads(args.predictor_kwargs_json)
        kwargs.setdefault("device", str(device))
        predictor = cls(**kwargs)
        if not callable(predictor):
            raise TypeError("External predictor must be callable")
        return predictor

    raise ValueError(f"Unsupported predictor_type: {args.predictor_type}")


def build_alpha_bars(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> Tensor:
    betas = torch.linspace(beta_start, beta_end, steps=num_steps)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def add_noise(x: Tensor, alpha_bar_t: float) -> Tensor:
    eps = torch.randn_like(x)
    return math.sqrt(alpha_bar_t) * x + math.sqrt(1.0 - alpha_bar_t) * eps


@torch.no_grad()
def run_predictor(
    predictor,
    x_t: Tensor,
    t_index: int,
    predictor_image_size: int,
) -> Tensor:
    if predictor_image_size and predictor_image_size > 0:
        height, width = x_t.shape[-2:]
        x_small = F.interpolate(
            x_t,
            size=(predictor_image_size, predictor_image_size),
            mode="bilinear",
            align_corners=False,
        )
        x0_small = predictor(x_small, t_index)
        x0_hat = F.interpolate(x0_small, size=(height, width), mode="bilinear", align_corners=False)
        return x0_hat.clamp(0.0, 1.0)
    return predictor(x_t, t_index).clamp(0.0, 1.0)
