from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from .base import PredictionContext


class OpenAICIFARJAXBackend:
    """Adapter for OpenAI CIFAR-10 JAX/Flax consistency models.

    This adapter keeps all OpenAI repo-specific code in the external repo. CAMP
    only handles path setup, checkpoint restore, and torch<->JAX tensor
    conversion around the repo's distiller function.
    """

    name = "openai_cifar_jax"

    def __init__(
        self,
        repo: str = "",
        checkpoint: str = "",
        config_module: str = "configs.cifar10_ve_cd",
        config_factory: str = "get_config",
        workdir: str = "",
        image_size: int = 32,
        batch_size: Optional[int] = None,
        rng_seed: int = 42,
        jit: bool = True,
        backend_device: str = "",
        input_range: str = "minus_one_one",
        output_range: str = "minus_one_one",
        torch_device: str = "cuda",
        device: str = "cuda",
        **_,
    ) -> None:
        self.repo = Path(repo).expanduser().resolve() if repo else None
        self.checkpoint = Path(checkpoint).expanduser().resolve() if checkpoint else None
        self.workdir = Path(workdir).expanduser().resolve() if workdir else None
        self.config_module = config_module
        self.config_factory = config_factory
        self.image_size = int(image_size)
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.rng_seed = int(rng_seed)
        self.use_jit = bool(jit)
        self.backend_device = str(backend_device or "").strip()
        self.input_range = input_range
        self.output_range = output_range
        self.torch_device = torch.device(torch_device or device)

        if self.repo is None:
            raise ValueError("OpenAICIFARJAXBackend requires model_kwargs.repo pointing to the OpenAI CIFAR-10 repo")
        if not self.repo.exists():
            raise FileNotFoundError(f"OpenAI CIFAR-10 repo not found: {self.repo}")
        if self.checkpoint is None:
            raise ValueError("OpenAICIFARJAXBackend requires model_kwargs.checkpoint")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"OpenAI CIFAR-10 checkpoint not found: {self.checkpoint}")

        self._add_repo_to_path(self.repo)
        self._load_jax_stack()
        self.config = self._load_config()
        self.model, self.state = self._restore_state()
        self.distiller_fn = self._build_distiller_fn()

    @staticmethod
    def _add_repo_to_path(repo: Path) -> None:
        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def _load_jax_stack(self) -> None:
        try:
            self.jax = importlib.import_module("jax")
            self.jnp = importlib.import_module("jax.numpy")
            self.flax_checkpoints = importlib.import_module("flax.training.checkpoints")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "OpenAICIFARJAXBackend requires JAX, Flax, and the OpenAI CIFAR-10 consistency-model "
                "dependencies installed in the active Python environment."
            ) from exc

        if self.backend_device:
            devices = [d for d in self.jax.devices() if d.platform == self.backend_device]
            if not devices:
                available = ", ".join(sorted({d.platform for d in self.jax.devices()}))
                raise RuntimeError(
                    f"Requested JAX backend_device='{self.backend_device}', but available platforms are: {available}"
                )
            self.jax_device = devices[0]
        else:
            self.jax_device = self.jax.devices()[0]

    def _load_config(self) -> Any:
        if importlib.util.find_spec(self.config_module) is None:
            raise RuntimeError(
                f"Could not find OpenAI CIFAR-10 config module '{self.config_module}' from repo {self.repo}"
            )
        try:
            module = importlib.import_module(self.config_module)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"OpenAI CIFAR-10 config module '{self.config_module}' was found, but one of its dependencies "
                f"is missing: {exc.name}"
            ) from exc
        try:
            factory = getattr(module, self.config_factory)
        except AttributeError as exc:
            raise RuntimeError(
                f"Config module '{self.config_module}' does not expose '{self.config_factory}'"
            ) from exc
        config = factory()
        if hasattr(config, "data") and hasattr(config.data, "image_size"):
            config.data.image_size = self.image_size
        if self.batch_size is not None and hasattr(config, "eval") and hasattr(config.eval, "batch_size"):
            config.eval.batch_size = self.batch_size
        return config

    def _restore_state(self) -> Any:
        mutils = importlib.import_module("jcm.models.utils")
        checkpoints = importlib.import_module("jcm.checkpoints")
        losses = importlib.import_module("jcm.losses")
        self._import_model_definitions()

        hk = importlib.import_module("haiku")
        rng_seq = hk.PRNGSequence(self.rng_seed)
        init_model = getattr(mutils, "init_model", None)
        if init_model is None:
            raise RuntimeError("OpenAI repo does not expose jcm.models.utils.init_model")

        init_out = init_model(next(rng_seq), self.config)
        if not isinstance(init_out, tuple) or len(init_out) != 3:
            raise RuntimeError("OpenAI repo jcm.models.utils.init_model returned an unexpected value")
        model, model_state, params = init_out
        optimizer, _ = losses.get_optimizer(self.config)
        loss_name = str(getattr(getattr(self.config, "training", object()), "loss", "")).lower()
        use_target = loss_name.endswith(("ema", "adaptive", "progressive_distillation"))
        state_cls_name = "StateWithTarget" if use_target else "State"
        state_cls = getattr(mutils, state_cls_name, None)
        if state_cls is None:
            raise RuntimeError(f"OpenAI repo does not expose {state_cls_name} in jcm.models.utils")
        state_kwargs = {
            "step": 0,
            "lr": getattr(getattr(self.config, "optim", object()), "lr", 0.0),
            "ema_rate": getattr(getattr(self.config, "model", object()), "ema_rate", 0.0),
            "params": params,
            "params_ema": params,
            "model_state": model_state,
            "opt_state": optimizer.init(params),
            "rng_state": rng_seq.internal_state,
        }
        if use_target:
            state_kwargs["target_params"] = params
        state = state_cls(**state_kwargs)

        repo_restore = getattr(checkpoints, "restore_checkpoint", None)
        if repo_restore is not None:
            try:
                restored = repo_restore(str(self.checkpoint), state, step=None)
            except TypeError:
                restored = repo_restore(str(self.checkpoint), state)
            return model, restored[0] if isinstance(restored, tuple) else restored

        return model, self.flax_checkpoints.restore_checkpoint(str(self.checkpoint), target=state)

    def _import_model_definitions(self) -> None:
        model_name = str(getattr(getattr(self.config, "model", object()), "name", ""))
        candidates = []
        if model_name:
            candidates.append(f"jcm.models.{model_name}")
        candidates.extend(
            [
                "jcm.models.ncsnpp",
                "jcm.models.wideresnet_noise_conditional",
            ]
        )
        imported_any = False
        for module_name in dict.fromkeys(candidates):
            if importlib.util.find_spec(module_name) is None:
                continue
            try:
                importlib.import_module(module_name)
                imported_any = True
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    f"OpenAI CIFAR-10 model module '{module_name}' was found, but one of its dependencies "
                    f"is missing: {exc.name}"
                ) from exc
        if not imported_any:
            raise RuntimeError(
                f"Could not find model definition module for config.model.name='{model_name}' in repo {self.repo}"
            )

    def _build_distiller_fn(self) -> Callable[..., Any]:
        mutils = importlib.import_module("jcm.models.utils")
        sde_lib = importlib.import_module("jcm.sde_lib")

        get_distiller_fn = getattr(mutils, "get_distiller_fn", None)
        if get_distiller_fn is None:
            raise RuntimeError("OpenAI repo does not expose jcm.models.utils.get_distiller_fn")

        sde = self._build_sde(sde_lib)
        params = self._state_params(self.state)
        states = getattr(self.state, "model_state", {})
        fn = get_distiller_fn(sde, self.model, params, states, train=False)

        def call(x: Any, sigma: Any) -> Any:
            return self._call_distiller(fn, x, sigma)

        return self.jax.jit(call) if self.use_jit else call

    def _build_sde(self, sde_lib: Any) -> Any:
        if hasattr(sde_lib, "get_sde"):
            return sde_lib.get_sde(self.config)
        raise RuntimeError("Could not construct an SDE from the OpenAI repo's jcm.sde_lib")

    def _call_distiller(self, fn: Callable[..., Any], x: Any, sigma: Any) -> Any:
        rng = self.jax.random.PRNGKey(self.rng_seed)
        call_attempts = (
            lambda: fn(x, sigma),
            lambda: fn(x, sigma, rng=rng),
            lambda: fn(x, sigma, train=False),
        )
        last_exc: Optional[Exception] = None
        for attempt in call_attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_exc = exc
        raise RuntimeError("Could not call OpenAI CIFAR-10 distiller function with supported signatures") from last_exc

    @staticmethod
    def _state_params(state: Any) -> Any:
        if hasattr(state, "ema_params"):
            return state.ema_params
        if hasattr(state, "params_ema"):
            return state.params_ema
        if hasattr(state, "params"):
            return state.params
        if isinstance(state, dict):
            for key in ("ema_params", "params_ema", "params", "target"):
                if key in state:
                    return state[key]
        return state

    def _to_jax_image(self, x_t: torch.Tensor) -> Any:
        x = x_t.detach().float().cpu().clamp(-1.0, 1.0)
        if self.input_range == "zero_one":
            x = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        elif self.input_range != "minus_one_one":
            raise ValueError(f"Unsupported OpenAI CIFAR JAX input_range: {self.input_range}")
        nhwc = x.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)
        return self.jax.device_put(nhwc, self.jax_device)

    def _to_jax_sigma(self, context: PredictionContext, batch_size: int) -> Any:
        sigma = context.sigma_t.detach().float().cpu().reshape(-1).numpy().astype(np.float32)
        if sigma.size == 1:
            sigma = np.repeat(sigma, batch_size)
        return self.jax.device_put(sigma, self.jax_device)

    def _from_jax_image(self, y: Any, like: torch.Tensor) -> torch.Tensor:
        if isinstance(y, (tuple, list)):
            y = y[0]
        arr = np.asarray(self.jax.device_get(y))
        if arr.ndim != 4:
            raise RuntimeError(f"OpenAI CIFAR-10 distiller returned shape {arr.shape}; expected NHWC or BCHW")
        if arr.shape[-1] in {1, 3}:
            bchw = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        elif arr.shape[1] in {1, 3}:
            bchw = torch.from_numpy(arr).contiguous()
        else:
            raise RuntimeError(f"Cannot infer channel dimension from OpenAI CIFAR-10 output shape {arr.shape}")
        if self.output_range == "zero_one":
            bchw = bchw.clamp(0.0, 1.0) * 2.0 - 1.0
        elif self.output_range != "minus_one_one":
            raise ValueError(f"Unsupported OpenAI CIFAR JAX output_range: {self.output_range}")
        return bchw.to(device=like.device, dtype=like.dtype).clamp(-1.0, 1.0)

    @torch.no_grad()
    def predict_x0(self, x_t: torch.Tensor, context: PredictionContext) -> torch.Tensor:
        if context.class_labels is not None:
            raise ValueError("OpenAI CIFAR-10 JAX CM adapter is configured as unconditional; class labels are unsupported")
        x_jax = self._to_jax_image(x_t)
        sigma_jax = self._to_jax_sigma(context, batch_size=x_t.shape[0])
        y = self.distiller_fn(x_jax, sigma_jax)
        return self._from_jax_image(y, like=x_t)
