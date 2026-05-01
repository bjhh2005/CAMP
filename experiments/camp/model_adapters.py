from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import PurificationConfig


def _import_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        root = Path(__file__).resolve().parents[2]
        experiments_dir = root / "experiments"
        for path in (str(root), str(experiments_dir)):
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            raise exc


class CallableGenerativeModel:
    """Thin range adapter around a callable consistency-model style backend."""

    def __init__(self, backend: Any, input_range: str, output_range: str, time_mode: str) -> None:
        if not callable(backend):
            raise TypeError("Generative model backend must be callable")
        self.backend = backend
        self.input_range = input_range
        self.output_range = output_range
        self.time_mode = time_mode

    def _to_backend_range(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_range == "minus_one_one":
            return x
        if self.input_range == "zero_one":
            return (x + 1.0) / 2.0
        raise ValueError(f"Unsupported model_input_range: {self.input_range}")

    def _from_backend_range(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_range == "minus_one_one":
            return x
        if self.output_range == "zero_one":
            return x * 2.0 - 1.0
        raise ValueError(f"Unsupported model_output_range: {self.output_range}")

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        sigma_t: torch.Tensor,
        rescaled_sigma_t: torch.Tensor,
        t_index: Optional[int] = None,
        classes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_backend = self._to_backend_range(x)
        call_classes = classes
        if classes is not None and hasattr(self.backend, "set_class_label"):
            if classes.numel() != 1:
                raise ValueError(
                    "Backend uses set_class_label(...) and only supports batch_size=1 for class-conditional calls"
                )
            self.backend.set_class_label(int(classes.flatten()[0].item()))
            call_classes = None

        if self.time_mode == "sigma":
            time_arg = sigma_t
        elif self.time_mode == "rescaled_sigma":
            time_arg = rescaled_sigma_t
        elif self.time_mode == "t_index":
            if t_index is None:
                raise ValueError("model_time_mode='t_index' requires t_index from the purifier schedule")
            time_arg = int(t_index)
        else:
            raise ValueError(f"Unsupported model_time_mode: {self.time_mode}")

        if hasattr(self.backend, "predict_from_sigma"):
            y = self.backend.predict_from_sigma(
                x_backend,
                sigma_t=sigma_t,
                rescaled_sigma_t=rescaled_sigma_t,
                t_index=t_index,
                classes=call_classes,
            )
            return self._from_backend_range(y).clamp(-1.0, 1.0)

        try:
            if call_classes is None:
                y = self.backend(x_backend, time_arg)
            else:
                y = self.backend(x_backend, time_arg, call_classes)
        except TypeError:
            if call_classes is None:
                y = self.backend(x_backend, t_index=int(float(time_arg)))
            else:
                y = self.backend(x_backend, t_index=int(float(time_arg)), classes=call_classes)
        return self._from_backend_range(y).clamp(-1.0, 1.0)


def build_generative_model(config: PurificationConfig, device: torch.device) -> CallableGenerativeModel:
    if config.model_type != "module":
        raise ValueError("Only model_type='module' is implemented in CAMP core")
    if ":" not in config.model_module:
        raise ValueError("purification.model_module must use 'package.module:ClassName'")
    module_name, class_name = config.model_module.split(":", 1)
    module = _import_module(module_name)
    cls = getattr(module, class_name)
    kwargs: Dict[str, Any] = dict(config.model_kwargs)
    kwargs.setdefault("device", str(device))
    backend = cls(**kwargs)
    return CallableGenerativeModel(
        backend=backend,
        input_range=config.model_input_range,
        output_range=config.model_output_range,
        time_mode=config.model_time_mode,
    )
