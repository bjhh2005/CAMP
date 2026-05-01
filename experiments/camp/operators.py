from __future__ import annotations

import torch


class IdentityOperator:
    """Observation model A=I for pure adversarial purification."""

    def __init__(self, eta: float = 0.0) -> None:
        self.eta = float(eta)

    def A(self, vec: torch.Tensor) -> torch.Tensor:
        return vec.reshape(vec.shape[0], -1)

    def A_pinv_add_eta(self, vec: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        denom = 1.0 + float(eta)
        return vec.reshape(vec.shape[0], -1) / denom

