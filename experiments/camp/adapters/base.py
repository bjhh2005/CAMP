from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch


@dataclass
class PredictionContext:
    sigma_t: torch.Tensor
    rescaled_sigma_t: torch.Tensor
    t_index: int
    class_labels: Optional[torch.Tensor] = None


class PurifierBackend(Protocol):
    name: str

    def predict_x0(self, x_t: torch.Tensor, context: PredictionContext) -> torch.Tensor:
        """Return x0 prediction in [-1, 1] BCHW format."""

