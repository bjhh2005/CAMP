from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .base import PredictionContext


class DiffusersConsistencyBackend:
    """Adapter for diffusers ConsistencyModelPipeline checkpoints.

    Expected checkpoint layout:
        model_index.json
        unet/
        scheduler/

    This uses CMStochasticIterativeScheduler semantics: scale the model input,
    call the consistency UNet, and apply scheduler.step(...). With a single
    timestep, the scheduler returns the denoised x0-style sample without adding
    the next-step stochastic noise.
    """

    name = "diffusers_consistency"

    def __init__(
        self,
        model_dir: str = "",
        unet_subdir: str = "unet",
        scheduler_subdir: str = "scheduler",
        input_range: str = "minus_one_one",
        output_range: str = "minus_one_one",
        class_cond: bool = False,
        default_class_label: int = 0,
        torch_dtype: str = "float32",
        device: str = "cuda",
        **_: Any,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else None
        self.input_range = input_range
        self.output_range = output_range
        self.class_cond = bool(class_cond)
        self.default_class_label = int(default_class_label)
        self.device = torch.device(device)
        self.torch_dtype = self._resolve_dtype(torch_dtype)

        if self.model_dir is None:
            raise ValueError("DiffusersConsistencyBackend requires model_kwargs.model_dir")
        if not self.model_dir.exists():
            raise FileNotFoundError(f"diffusers model_dir not found: {self.model_dir}")
        if not (self.model_dir / "model_index.json").exists():
            raise FileNotFoundError(f"diffusers model_index.json not found in: {self.model_dir}")

        try:
            from diffusers import CMStochasticIterativeScheduler, UNet2DModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DiffusersConsistencyBackend requires diffusers. "
                "Install it in camp_torch with: python -m pip install diffusers safetensors accelerate"
            ) from exc

        self.scheduler = CMStochasticIterativeScheduler.from_pretrained(
            str(self.model_dir),
            subfolder=scheduler_subdir,
        )
        self.unet = UNet2DModel.from_pretrained(str(self.model_dir), subfolder=unet_subdir)
        self.unet.to(device=self.device, dtype=self.torch_dtype)
        self.unet.eval()

        # The scheduler's initial sigmas are CPU tensors by design in diffusers.
        self.train_sigmas = self.scheduler.sigmas.detach().float().cpu()

    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        name = str(name or "float32").lower()
        if name in {"fp16", "float16", "half"}:
            return torch.float16
        if name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if name in {"fp32", "float32", "float"}:
            return torch.float32
        raise ValueError(f"Unsupported torch_dtype for diffusers_consistency: {name}")

    def _to_backend_input(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.input_range == "minus_one_one":
            return x_t
        if self.input_range == "zero_one":
            return ((x_t + 1.0) / 2.0).clamp(0.0, 1.0)
        raise ValueError(f"Unsupported diffusers consistency input_range: {self.input_range}")

    def _to_minus_one_one(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_range == "minus_one_one":
            return x.clamp(-1.0, 1.0)
        if self.output_range == "zero_one":
            return (x.clamp(0.0, 1.0) * 2.0 - 1.0).clamp(-1.0, 1.0)
        raise ValueError(f"Unsupported diffusers consistency output_range: {self.output_range}")

    def _timestep_index_from_sigma(self, context: PredictionContext) -> int:
        sigma = float(context.sigma_t.detach().float().cpu().reshape(-1)[0].item())
        idx = torch.argmin(torch.abs(self.train_sigmas - sigma)).item()
        max_idx = int(getattr(self.scheduler.config, "num_train_timesteps", len(self.train_sigmas)) - 1)
        return max(0, min(max_idx, int(idx)))

    def _class_labels(self, sample: torch.Tensor, context: PredictionContext) -> torch.Tensor | None:
        if not self.class_cond:
            return None
        if context.class_labels is None:
            return torch.full(
                (sample.shape[0],),
                self.default_class_label,
                device=self.device,
                dtype=torch.long,
            )
        labels = context.class_labels.to(device=self.device, dtype=torch.long).reshape(-1)
        if labels.numel() == 1 and sample.shape[0] > 1:
            labels = labels.repeat(sample.shape[0])
        if labels.numel() != sample.shape[0]:
            raise ValueError(f"class_labels batch mismatch: got {labels.numel()}, expected {sample.shape[0]}")
        return labels

    @torch.no_grad()
    def predict_x0(self, x_t: torch.Tensor, context: PredictionContext) -> torch.Tensor:
        sample = self._to_backend_input(x_t).to(device=self.device, dtype=self.torch_dtype)
        timestep_index = self._timestep_index_from_sigma(context)

        # One-step denoising: with a single timestep the scheduler does not add
        # stochastic next-step noise, so prev_sample is the denoised estimate.
        self.scheduler.set_timesteps(timesteps=[timestep_index], device=self.device)
        timestep = self.scheduler.timesteps[0]
        model_input = self.scheduler.scale_model_input(sample, timestep)
        model_output = self.unet(
            model_input,
            timestep,
            class_labels=self._class_labels(sample, context),
        ).sample
        x0 = self.scheduler.step(model_output, timestep, sample).prev_sample
        return self._to_minus_one_one(x0.float()).to(device=x_t.device, dtype=x_t.dtype)
