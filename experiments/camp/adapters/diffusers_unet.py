from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .base import PredictionContext


class DiffusersUNetBackend:
    """Adapter for diffusers DDPM-style UNet checkpoints.

    The backend returns an x0 estimate in CAMP's BCHW [-1, 1] convention. For
    epsilon-prediction DDPM checkpoints, x0 is reconstructed from the scheduler's
    alpha cumulative product.
    """

    name = "diffusers_unet"

    def __init__(
        self,
        model_dir: str = "",
        unet_subdir: str = "unet",
        scheduler_subdir: str = "scheduler",
        input_range: str = "minus_one_one",
        output_range: str = "minus_one_one",
        prediction_type: str = "",
        scheduler_prediction_type: str = "",
        torch_dtype: str = "float32",
        device: str = "cuda",
        **_: Any,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else None
        self.input_range = input_range
        self.output_range = output_range
        self.prediction_type_override = str(scheduler_prediction_type or prediction_type or "").strip().lower()
        self.device = torch.device(device)
        self.torch_dtype = self._resolve_dtype(torch_dtype)

        if self.model_dir is None:
            raise ValueError("DiffusersUNetBackend requires model_kwargs.model_dir")
        if not self.model_dir.exists():
            raise FileNotFoundError(f"diffusers model_dir not found: {self.model_dir}")
        if not (self.model_dir / "model_index.json").exists():
            raise FileNotFoundError(f"diffusers model_index.json not found in: {self.model_dir}")

        try:
            from diffusers import DDPMScheduler, UNet2DModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DiffusersUNetBackend requires the 'diffusers' package. "
                "Install it in camp_torch with: python -m pip install diffusers safetensors"
            ) from exc

        self.scheduler = DDPMScheduler.from_pretrained(str(self.model_dir), subfolder=scheduler_subdir)
        self.unet = UNet2DModel.from_pretrained(str(self.model_dir), subfolder=unet_subdir)
        self.unet.to(device=self.device, dtype=self.torch_dtype)
        self.unet.eval()
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=self.device, dtype=torch.float32)
        self.prediction_type = self._resolve_prediction_type()

    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        name = str(name or "float32").lower()
        if name in {"fp16", "float16", "half"}:
            return torch.float16
        if name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if name in {"fp32", "float32", "float"}:
            return torch.float32
        raise ValueError(f"Unsupported torch_dtype for diffusers_unet: {name}")

    def _resolve_prediction_type(self) -> str:
        # registry.py passes CAMP's generic model_prediction_type as
        # prediction_type. Values such as network_output/x0 describe CAMP's old
        # adapter contract, not diffusers scheduler prediction semantics.
        if self.prediction_type_override in {"network_output", "x0"}:
            pred = str(getattr(self.scheduler.config, "prediction_type", "epsilon")).lower()
        elif self.prediction_type_override:
            pred = self.prediction_type_override
        else:
            pred = str(getattr(self.scheduler.config, "prediction_type", "epsilon")).lower()
        if pred not in {"epsilon", "sample", "v_prediction"}:
            raise ValueError(f"Unsupported diffusers prediction_type: {pred}")
        return pred

    def _to_backend_input(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.input_range == "minus_one_one":
            return x_t
        if self.input_range == "zero_one":
            return ((x_t + 1.0) / 2.0).clamp(0.0, 1.0)
        raise ValueError(f"Unsupported diffusers input_range: {self.input_range}")

    def _to_minus_one_one(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_range == "minus_one_one":
            return x.clamp(-1.0, 1.0)
        if self.output_range == "zero_one":
            return (x.clamp(0.0, 1.0) * 2.0 - 1.0).clamp(-1.0, 1.0)
        raise ValueError(f"Unsupported diffusers output_range: {self.output_range}")

    def _timestep(self, context: PredictionContext) -> int:
        max_t = int(self.alphas_cumprod.numel() - 1)
        return max(0, min(max_t, int(context.t_index)))

    def _predict_x0_from_model_output(self, x_t: torch.Tensor, model_output: torch.Tensor, timestep: int) -> torch.Tensor:
        alpha_prod_t = self.alphas_cumprod[timestep].to(device=x_t.device, dtype=x_t.dtype)
        beta_prod_t = (1.0 - alpha_prod_t).clamp_min(1e-12)
        sqrt_alpha = alpha_prod_t.sqrt()
        sqrt_beta = beta_prod_t.sqrt()

        if self.prediction_type == "sample":
            return model_output
        if self.prediction_type == "epsilon":
            return (x_t - sqrt_beta * model_output) / sqrt_alpha.clamp_min(1e-12)
        # v_prediction, matching diffusers' DDPM conversion.
        return sqrt_alpha * x_t - sqrt_beta * model_output

    @torch.no_grad()
    def predict_x0(self, x_t: torch.Tensor, context: PredictionContext) -> torch.Tensor:
        if context.class_labels is not None:
            raise ValueError("DiffusersUNetBackend currently supports unconditional UNet checkpoints only")
        timestep = self._timestep(context)
        sample = self._to_backend_input(x_t).to(device=self.device, dtype=self.torch_dtype)
        t = torch.full((sample.shape[0],), timestep, device=self.device, dtype=torch.long)
        model_output = self.unet(sample, t).sample
        x0 = self._predict_x0_from_model_output(sample, model_output, timestep)
        return self._to_minus_one_one(x0.float()).to(device=x_t.device, dtype=x_t.dtype)
