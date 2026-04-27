"""
Sony CM adapter for CAMP/WGCP.

This adapter loads official sony/ctm code + a consistency-model checkpoint and
exposes a callable predictor interface:

    __call__(x_t: Tensor[B,C,H,W], t_index: int) -> Tensor[B,C,H,W]

Expected x_t range: [0, 1]
Output range: [0, 1]

Notes:
- The official sony/ctm repo serves both CTM and CM-style checkpoints.
- Different checkpoints may prefer slightly different inference knobs
  (`training_mode`, `ctm_inference`, `output_head`), so those are configurable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class CMRepoPredictor:
    def __init__(
        self,
        ctm_repo: str,
        checkpoint: str,
        device: str = "cuda",
        data_name: str = "imagenet64",
        num_diffusion_steps: int = 1000,
        predictor_image_size: int = 64,
        class_cond: bool = True,
        class_label: int = 0,
        use_fp16: bool = True,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        start_scales: int = 40,
        inner_parametrization: str = "edm",
        outer_parametrization: str = "euler",
        attention_type: str = "flash",
        strict_load: bool = False,
        training_mode: str = "consistency_distillation",
        ctm_inference: bool = False,
        output_head: str = "g_theta",
        teacher_model_path: str = "",
        **_: Any,
    ) -> None:
        self.ctm_repo = Path(ctm_repo).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device)
        self.data_name = data_name
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.predictor_image_size = int(predictor_image_size)
        self.class_cond = bool(class_cond)
        self.class_label = int(class_label)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.strict_load = bool(strict_load)
        self.training_mode = str(training_mode).strip() or "consistency_distillation"
        self.ctm_inference = bool(ctm_inference)
        self.output_head = str(output_head).strip().lower() or "g_theta"
        self.teacher_model_path = str(teacher_model_path).strip() or str(self.checkpoint)

        if self.output_head not in {"g_theta", "denoised"}:
            raise ValueError("output_head must be one of: g_theta, denoised")

        if not self.ctm_repo.exists():
            raise FileNotFoundError(f"CM repo not found: {self.ctm_repo}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")

        code_dir = self.ctm_repo / "code"
        if not code_dir.exists():
            raise FileNotFoundError(f"CM code dir not found: {code_dir}")
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))

        from cm.script_util import (  # type: ignore
            cm_train_defaults,
            create_model_and_diffusion,
            ctm_data_defaults,
            ctm_eval_defaults,
            ctm_loss_defaults,
            ctm_train_defaults,
            model_and_diffusion_defaults,
            train_defaults,
        )

        defaults: Dict[str, Any] = {}
        defaults.update(train_defaults(data_name))
        defaults.update(model_and_diffusion_defaults(data_name))
        defaults.update(cm_train_defaults(data_name))
        defaults.update(ctm_train_defaults(data_name))
        defaults.update(ctm_eval_defaults(data_name))
        defaults.update(ctm_loss_defaults(data_name))
        defaults.update(ctm_data_defaults(data_name))

        defaults.update(
            {
                "data_name": data_name,
                "model_path": str(self.checkpoint),
                "teacher_model_path": self.teacher_model_path,
                "training_mode": self.training_mode,
                "use_MPI": False,
                "device_id": 0,
                "class_cond": self.class_cond,
                "class_cond_true": self.class_cond,
                "num_classes": 1000 if data_name.lower() == "imagenet64" else defaults.get("num_classes", 10),
                "use_fp16": bool(use_fp16 and self.device.type == "cuda"),
                "image_size": int(self.predictor_image_size),
                "sigma_min": self.sigma_min,
                "sigma_max": self.sigma_max,
                "rho": self.rho,
                "start_scales": int(start_scales),
                "inner_parametrization": inner_parametrization,
                "outer_parametrization": outer_parametrization,
                "attention_type": (
                    attention_type
                    if bool(use_fp16 and self.device.type == "cuda") or attention_type != "flash"
                    else "xformer"
                ),
                "target_subtract": False,
                "rescaling": False,
            }
        )

        self.args = SimpleNamespace(**defaults)
        self.model, self.diffusion = create_model_and_diffusion(self.args)

        state = torch.load(str(self.checkpoint), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if not isinstance(state, dict):
            raise RuntimeError("Unexpected checkpoint format. Expecting state_dict-like object.")

        load_info = self.model.load_state_dict(state, strict=self.strict_load)
        if not self.strict_load:
            missing = getattr(load_info, "missing_keys", [])
            unexpected = getattr(load_info, "unexpected_keys", [])
            if missing or unexpected:
                print(
                    "[CMRepoPredictor] non-strict checkpoint load summary:",
                    json.dumps(
                        {
                            "missing_keys": len(missing),
                            "unexpected_keys": len(unexpected),
                        }
                    ),
                )

        self.model.to(self.device)
        self.use_fp16 = bool(getattr(self.args, "use_fp16", False) and self.device.type == "cuda")
        self.compute_dtype = torch.float32
        self.model.eval()

    def set_class_label(self, class_label: int) -> None:
        self.class_label = int(class_label)

    def list_named_modules(self, leaf_only: bool = True) -> List[str]:
        names: List[str] = []
        for name, module in self.model.named_modules():
            if not name:
                continue
            if leaf_only and any(True for _ in module.children()):
                continue
            names.append(name)
        return names

    def _resolve_named_module(self, layer: str) -> Tuple[str, torch.nn.Module]:
        layer_text = str(layer).strip()
        if not layer_text:
            raise ValueError("layer must be a non-empty string")

        named_modules = [(name, module) for name, module in self.model.named_modules() if name]
        exact = [(name, module) for name, module in named_modules if name == layer_text]
        if len(exact) == 1:
            return exact[0]

        if layer_text.isdigit():
            idx = int(layer_text)
            leaf_names = self.list_named_modules(leaf_only=True)
            if idx < 0 or idx >= len(leaf_names):
                raise IndexError(f"layer index out of range: {idx} not in [0, {len(leaf_names) - 1}]")
            target_name = leaf_names[idx]
            for name, module in named_modules:
                if name == target_name:
                    return name, module

        partial = [(name, module) for name, module in named_modules if layer_text in name]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            names = ", ".join(name for name, _ in partial[:8])
            raise ValueError(f"layer '{layer_text}' matched multiple modules: {names}")

        raise KeyError(f"layer '{layer_text}' not found in CM model modules")

    def _index_to_sigma(self, t_index: int) -> float:
        if self.num_diffusion_steps <= 1:
            return self.sigma_min
        frac = float(np.clip(t_index / float(self.num_diffusion_steps - 1), 0.0, 1.0))
        sigma = self.sigma_max ** (1.0 / self.rho) + frac * (
            self.sigma_min ** (1.0 / self.rho) - self.sigma_max ** (1.0 / self.rho)
        )
        return float(sigma**self.rho)

    def _build_model_inputs(
        self,
        x_t: torch.Tensor,
        t_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if x_t.ndim != 4:
            raise ValueError("x_t must be [B,C,H,W]")
        x_t = x_t.to(device=self.device, dtype=self.compute_dtype)
        bsz = x_t.shape[0]
        x_model = x_t * 2.0 - 1.0

        sigma_t = self._index_to_sigma(int(t_index))
        t = torch.full((bsz,), sigma_t, device=self.device, dtype=x_model.dtype)

        model_kwargs: Dict[str, torch.Tensor] = {}
        if self.class_cond:
            y = torch.full((bsz,), int(self.class_label), device=self.device, dtype=torch.long)
            model_kwargs["y"] = y
        return x_t, x_model, t, model_kwargs

    def _run_diffusion(self, x_t: torch.Tensor, t_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _, x_model, t, model_kwargs = self._build_model_inputs(x_t, t_index)
        s = torch.full((x_model.shape[0],), self.sigma_min, device=self.device, dtype=x_model.dtype)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.use_fp16 and self.device.type == "cuda",
        ):
            denoised, g_theta = self.diffusion.get_denoised_and_G(
                self.model,
                x_model,
                t=t,
                s=s,
                ctm=self.ctm_inference,
                teacher=False,
                **model_kwargs,
            )
        return denoised, g_theta

    def _to_output_image(self, tensor: torch.Tensor) -> torch.Tensor:
        x0 = (tensor + 1.0) / 2.0
        return x0.float().clamp(0.0, 1.0)

    def _extract_first_tensor(self, obj: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (list, tuple)):
            for item in obj:
                tensor = self._extract_first_tensor(item)
                if tensor is not None:
                    return tensor
            return None
        if isinstance(obj, dict):
            for item in obj.values():
                tensor = self._extract_first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _collect_spatial_features(
        self,
        x_t: torch.Tensor,
        t_index: int,
        leaf_only: bool = True,
    ) -> List[Tuple[str, torch.Tensor]]:
        captures: List[Tuple[str, torch.Tensor]] = []
        hooks = []
        try:
            for name, module in self.model.named_modules():
                if not name:
                    continue
                if leaf_only and any(True for _ in module.children()):
                    continue

                def _make_hook(module_name: str):
                    def _hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                        tensor = self._extract_first_tensor(output)
                        if tensor is None or tensor.ndim != 4:
                            return
                        captures.append((module_name, tensor.detach()))

                    return _hook

                hooks.append(module.register_forward_hook(_make_hook(name)))
            self._run_diffusion(x_t, t_index)
        finally:
            for hook in hooks:
                hook.remove()
        return captures

    def _select_auto_feature_capture(
        self,
        captures: List[Tuple[str, torch.Tensor]],
    ) -> Tuple[str, torch.Tensor]:
        if not captures:
            raise RuntimeError("Could not auto-select a spatial 4D feature map from the CM model")

        strong_candidates = [
            (name, feat)
            for name, feat in captures
            if feat.ndim == 4 and int(feat.shape[1]) >= 8 and min(int(feat.shape[-2]), int(feat.shape[-1])) >= 8
        ]
        if strong_candidates:
            return strong_candidates[-1]

        weak_candidates = [(name, feat) for name, feat in captures if feat.ndim == 4 and int(feat.shape[1]) > 3]
        if weak_candidates:
            return weak_candidates[-1]

        return captures[-1]

    def _choose_output_tensor(self, denoised: torch.Tensor, g_theta: torch.Tensor) -> torch.Tensor:
        return g_theta if self.output_head == "g_theta" else denoised

    def extract_feature_map(
        self,
        x_t: torch.Tensor,
        t_index: int,
        layer: str = "",
        leaf_only: bool = True,
    ) -> Dict[str, Any]:
        feature_capture: Dict[str, torch.Tensor] = {}

        if layer.strip():
            module_name, module = self._resolve_named_module(layer)

            def _hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                tensor = self._extract_first_tensor(output)
                if tensor is None:
                    raise RuntimeError(f"Module '{module_name}' did not emit a tensor output")
                feature_capture["feature"] = tensor.detach()

            hook = module.register_forward_hook(_hook)
            try:
                denoised, g_theta = self._run_diffusion(x_t, t_index)
            finally:
                hook.remove()
            feature = feature_capture.get("feature")
            if feature is None:
                raise RuntimeError(f"Failed to capture feature map from module '{module_name}'")
            if feature.ndim != 4:
                raise ValueError(f"Captured feature from '{module_name}' is not 4D: shape={tuple(feature.shape)}")
            return {
                "layer": module_name,
                "feature": feature.float(),
                "x0_hat": self._to_output_image(self._choose_output_tensor(denoised, g_theta)),
            }

        captures = self._collect_spatial_features(x_t=x_t, t_index=t_index, leaf_only=leaf_only)
        module_name, feature = self._select_auto_feature_capture(captures)
        denoised, g_theta = self._run_diffusion(x_t, t_index)
        return {
            "layer": module_name,
            "feature": feature.float(),
            "x0_hat": self._to_output_image(self._choose_output_tensor(denoised, g_theta)),
        }

    @torch.no_grad()
    def __call__(self, x_t: torch.Tensor, t_index: int) -> torch.Tensor:
        denoised, g_theta = self._run_diffusion(x_t, t_index)
        return self._to_output_image(self._choose_output_tensor(denoised, g_theta))
