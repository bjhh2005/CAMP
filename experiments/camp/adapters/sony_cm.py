from __future__ import annotations

from typing import Optional

import torch

from experiments.cm_adapter_sony import CMRepoPredictor

from .base import PredictionContext


class SonyCMBackend:
    name = "sony_cm"

    def __init__(
        self,
        input_range: str = "minus_one_one",
        output_range: str = "zero_one",
        input_kind: str = "xt",
        prediction_type: str = "x0",
        **kwargs,
    ) -> None:
        self.predictor = CMRepoPredictor(**kwargs)
        self.input_range = input_range
        self.output_range = output_range
        self.input_kind = input_kind
        self.prediction_type = prediction_type

    def _to_backend_input(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.input_range == "minus_one_one":
            return x_t
        if self.input_range == "zero_one":
            return ((x_t + 1.0) / 2.0).clamp(0.0, 1.0)
        raise ValueError(f"Unsupported input_range: {self.input_range}")

    def _to_minus_one_one(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_range == "minus_one_one":
            return x.clamp(-1.0, 1.0)
        if self.output_range == "zero_one":
            return (x.clamp(0.0, 1.0) * 2.0 - 1.0).clamp(-1.0, 1.0)
        raise ValueError(f"Unsupported output_range: {self.output_range}")

    @torch.no_grad()
    def predict_x0(self, x_t: torch.Tensor, context: PredictionContext) -> torch.Tensor:
        if context.class_labels is not None and hasattr(self.predictor, "set_class_label"):
            if context.class_labels.numel() != 1:
                raise ValueError("SonyCMBackend set_class_label path currently requires batch_size=1")
            self.predictor.set_class_label(int(context.class_labels.flatten()[0].item()))
        x_backend = self._to_backend_input(x_t)
        y = self.predictor(x_backend, int(context.t_index))
        return self._to_minus_one_one(y)

