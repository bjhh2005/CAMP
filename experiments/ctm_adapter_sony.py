"""
Sony CTM adapter for CAMP/WGCP.

This adapter loads official sony/ctm code + checkpoint and exposes a callable
predictor interface:

    __call__(x_t: Tensor[B,C,H,W], t_index: int) -> Tensor[B,C,H,W]

Expected x_t range: [0, 1]
Output range: [0, 1]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch


class CTMRepoPredictor:
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

        if not self.ctm_repo.exists():
            raise FileNotFoundError(f"CTM repo not found: {self.ctm_repo}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")

        code_dir = self.ctm_repo / "code"
        if not code_dir.exists():
            raise FileNotFoundError(f"CTM code dir not found: {code_dir}")
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

        # Minimal overrides to run in inference mode for CAMP.
        defaults.update(
            {
                "data_name": data_name,
                "model_path": str(self.checkpoint),
                "teacher_model_path": str(self.checkpoint),
                "training_mode": "ctm",
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
                "attention_type": attention_type,
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
                    "[CTMRepoPredictor] non-strict checkpoint load summary:",
                    json.dumps(
                        {
                            "missing_keys": len(missing),
                            "unexpected_keys": len(unexpected),
                        }
                    ),
                )

        self.model.to(self.device)
        if getattr(self.args, "use_fp16", False) and hasattr(self.model, "convert_to_fp16"):
            self.model.convert_to_fp16()
        self.model.eval()

    def set_class_label(self, class_label: int) -> None:
        self.class_label = int(class_label)

    def _index_to_sigma(self, t_index: int) -> float:
        if self.num_diffusion_steps <= 1:
            return self.sigma_min
        frac = float(np.clip(t_index / float(self.num_diffusion_steps - 1), 0.0, 1.0))
        sigma = self.sigma_max ** (1.0 / self.rho) + frac * (
            self.sigma_min ** (1.0 / self.rho) - self.sigma_max ** (1.0 / self.rho)
        )
        return float(sigma**self.rho)

    @torch.no_grad()
    def __call__(self, x_t: torch.Tensor, t_index: int) -> torch.Tensor:
        if x_t.ndim != 4:
            raise ValueError("x_t must be [B,C,H,W]")
        x_t = x_t.to(self.device)
        bsz = x_t.shape[0]

        # CAMP code uses [0,1]; CTM code expects [-1,1].
        x_ctm = x_t * 2.0 - 1.0

        sigma_t = self._index_to_sigma(int(t_index))
        t = torch.full((bsz,), sigma_t, device=self.device, dtype=x_ctm.dtype)
        s = torch.full((bsz,), self.sigma_min, device=self.device, dtype=x_ctm.dtype)

        model_kwargs: Dict[str, torch.Tensor] = {}
        if self.class_cond:
            y = torch.full((bsz,), int(self.class_label), device=self.device, dtype=torch.long)
            model_kwargs["y"] = y

        denoised, g_theta = self.diffusion.get_denoised_and_G(
            self.model,
            x_ctm,
            t=t,
            s=s,
            ctm=True,
            teacher=False,
            **model_kwargs,
        )

        # For CTM transition t->s, G_theta corresponds to the destination state.
        x0 = g_theta
        x0 = (x0 + 1.0) / 2.0
        return x0.clamp(0.0, 1.0)

