from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from experiments.camp.config import PurificationConfig

from .base import PurifierBackend


def _import_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        root = Path(__file__).resolve().parents[3]
        experiments_dir = root / "experiments"
        for path in (str(root), str(experiments_dir)):
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            raise exc


def _build_from_class_path(class_path: str, kwargs: Dict[str, Any], device: torch.device):
    if ":" not in class_path:
        raise ValueError(f"Expected class path in package.module:ClassName format, got: {class_path}")
    module_name, class_name = class_path.split(":", 1)
    module = _import_module(module_name)
    cls = getattr(module, class_name)
    kwargs = dict(kwargs)
    kwargs.setdefault("device", str(device))
    return cls(**kwargs)


def build_backend(config: PurificationConfig, device: torch.device) -> PurifierBackend:
    backend_name = str(config.backend or "").strip()
    kwargs = dict(config.model_kwargs)
    kwargs.setdefault("input_range", config.model_input_range)
    kwargs.setdefault("output_range", config.model_output_range)
    kwargs.setdefault("time_mode", config.model_time_mode)
    kwargs.setdefault("input_kind", config.model_input_kind)
    kwargs.setdefault("prediction_type", config.model_prediction_type)
    kwargs.setdefault("class_cond", config.class_cond)
    kwargs.setdefault("default_class_label", config.default_class_label)

    if backend_name in {"", "module"}:
        if not config.model_module:
            raise ValueError("purification.model_module is required when backend is empty/module")
        return _build_from_class_path(config.model_module, kwargs, device=device)

    if backend_name == "debug_gaussian":
        from .debug_gaussian import DebugGaussianBackend

        return DebugGaussianBackend(device=str(device), **config.model_kwargs)

    if backend_name == "sony_cm":
        from .sony_cm import SonyCMBackend

        return SonyCMBackend(device=str(device), **kwargs)

    if backend_name == "openai_cifar_jax":
        from .openai_cifar_jax import OpenAICIFARJAXBackend

        return OpenAICIFARJAXBackend(device=str(device), **kwargs)

    if backend_name == "diffusers_unet":
        from .diffusers_unet import DiffusersUNetBackend

        return DiffusersUNetBackend(device=str(device), **kwargs)

    if backend_name == "class_path":
        class_path = str(config.model_module).strip()
        return _build_from_class_path(class_path, kwargs, device=device)

    raise ValueError(f"Unsupported purification.backend: {backend_name}")
