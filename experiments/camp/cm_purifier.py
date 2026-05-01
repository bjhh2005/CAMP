from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from .config import PurificationConfig
from .operators import IdentityOperator
from .schedule import alpha_to_nearest_t_index, build_alpha_schedule, build_betas
from .wavelet_ops import attenuate_bp_guidance, enhance_noise_estimate


Tensor = torch.Tensor


def append_dims(x: Tensor, target_dims: int) -> Tensor:
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError("input has more dimensions than target")
    return x[(...,) + (None,) * dims_to_append]


def get_scalings_for_boundary_condition(
    sigma: Tensor,
    sigma_data: float = 0.5,
    sigma_min: float = 0.002,
):
    c_skip = sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)
    c_out = (sigma - sigma_min) * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
    c_in = 1.0 / (sigma**2 + sigma_data**2) ** 0.5
    return c_skip, c_out, c_in


@dataclass
class PurificationResult:
    x_purified: Tensor
    trace: Dict[str, Any]


class CMPurifier:
    """CM4IR-style few-step purifier with optional wavelet noise/BP guidance."""

    def __init__(self, config: PurificationConfig, model, device: torch.device) -> None:
        self.config = config
        self.model = model
        self.device = device
        self.operator = IdentityOperator()
        self.betas = build_betas(config.schedule, device=device)

    @torch.no_grad()
    def purify(
        self,
        x_adv: Tensor,
        classes: Optional[Tensor] = None,
        return_trace: bool = False,
    ) -> PurificationResult:
        if x_adv.ndim != 4:
            raise ValueError("x_adv must be BCHW")
        x = x_adv.to(self.device).float().clamp(-1.0, 1.0)
        y_0 = x.reshape(x.shape[0], -1)
        alphas = build_alpha_schedule(self.config.schedule, self.betas, device=self.device)
        alphas_next: List[Any] = alphas[1:] + [-1]
        sigma_t0 = torch.sqrt(1.0 - alphas[0])
        xt = x + torch.randn_like(x) * sigma_t0

        trace: Dict[str, Any] = {
            "sigma": [],
            "wavelet_noise_enabled": bool(self.config.wavelet_noise.enabled),
            "bp_enabled": bool(self.config.bp.enabled),
            "wavelet_bp_enabled": bool(self.config.bp.wavelet.enabled),
        }

        x0_t = x
        for iter_ind, (at, at_next) in enumerate(zip(alphas, alphas_next)):
            sigma_t = torch.sqrt(1.0 - at)
            sigma_t = torch.clamp(
                sigma_t,
                min=float(self.config.schedule.sigma_min),
                max=float(self.config.schedule.sigma_max),
            )
            trace["sigma"].append(float(sigma_t.item()))

            c_skip, c_out, c_in = [
                append_dims(s, xt.ndim)
                for s in get_scalings_for_boundary_condition(sigma_t, sigma_min=self.config.schedule.sigma_min)
            ]
            rescaled_sigma_t = 1000.0 * 0.25 * torch.log(sigma_t + 1e-44)
            t_index = alpha_to_nearest_t_index(self.betas, at)
            if self.config.model_input_kind == "cm_scaled":
                model_input = c_in * xt
            elif self.config.model_input_kind == "xt":
                model_input = xt
            else:
                raise ValueError(f"Unsupported model_input_kind: {self.config.model_input_kind}")

            if classes is None:
                model_pred = self.model(model_input, sigma_t, rescaled_sigma_t, t_index=t_index)
            else:
                model_pred = self.model(model_input, sigma_t, rescaled_sigma_t, t_index=t_index, classes=classes)

            if self.config.model_prediction_type == "network_output":
                x0_t = (c_out * model_pred + c_skip * xt).clamp(-1.0, 1.0)
            elif self.config.model_prediction_type == "x0":
                x0_t = model_pred.clamp(-1.0, 1.0)
            else:
                raise ValueError(f"Unsupported model_prediction_type: {self.config.model_prediction_type}")

            z_hat_minus = (x0_t - xt) / sigma_t
            if self.config.wavelet_noise.enabled:
                z_hat_minus = enhance_noise_estimate(
                    z_hat_minus,
                    wavelet=self.config.wavelet_noise.wavelet,
                    levels=self.config.wavelet_noise.levels,
                    gains=self.config.wavelet_noise.gains,
                    bands=self.config.wavelet_noise.bands,
                )

            if self.config.bp.enabled and float(self.config.bp.mu) != 0.0:
                bp_eta_reg = float(self.config.bp.sigma_y) ** 2 * float(self.config.bp.zeta)
                residual = self.operator.A(x0_t.reshape(x0_t.size(0), -1)) - y_0
                bp_guidance = self.operator.A_pinv_add_eta(residual, eta=bp_eta_reg).reshape_as(x)
                if self.config.bp.wavelet.enabled:
                    bp_guidance = attenuate_bp_guidance(
                        bp_guidance,
                        wavelet=self.config.bp.wavelet.wavelet,
                        levels=self.config.bp.wavelet.levels,
                        highfreq_scales=self.config.bp.wavelet.highfreq_scales,
                        lowfreq_scale=self.config.bp.wavelet.lowfreq_scale,
                        bands=self.config.bp.wavelet.bands,
                    )
                xt = x0_t - (float(self.config.bp.mu) * float(self.config.bp.mu_factor) ** iter_ind) * bp_guidance
            else:
                xt = x0_t

            if at_next != -1:
                next_sigma_t = torch.sqrt(1.0 - at_next)
                next_sigma_t = torch.clamp(
                    next_sigma_t,
                    min=float(self.config.schedule.sigma_min),
                    max=float(self.config.schedule.sigma_max),
                )
                c1 = next_sigma_t * float(self.config.schedule.eta)
                c2 = next_sigma_t * ((1.0 - float(self.config.schedule.eta) ** 2) ** 0.5)
                xt = xt + c1 * torch.randn_like(xt) + c2 * z_hat_minus

        return PurificationResult(
            x_purified=xt.clamp(-1.0, 1.0),
            trace=trace if return_trace else {},
        )
