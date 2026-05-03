from __future__ import annotations

import dataclasses
import json
import os
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class ScheduleConfig:
    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    sampling_steps: int = 4
    iN: int = 80
    gamma: float = 0.02
    eta: float = 0.0
    sigma_min: float = 0.002
    sigma_max: float = 80.0


@dataclass
class WaveletNoiseConfig:
    enabled: bool = False
    wavelet: str = "db2"
    levels: int = 2
    gains: List[float] = field(default_factory=lambda: [1.0, 1.1])
    bands: List[str] = field(default_factory=lambda: ["HL", "LH", "HH"])


@dataclass
class WaveletBPConfig:
    enabled: bool = False
    wavelet: str = "db2"
    levels: int = 2
    highfreq_scales: List[float] = field(default_factory=lambda: [1.0, 0.5])
    lowfreq_scale: float = 1.0
    bands: List[str] = field(default_factory=lambda: ["HL", "LH", "HH"])


@dataclass
class BPConfig:
    enabled: bool = False
    mu: float = 0.0
    mu_factor: float = 1.0
    zeta: float = 0.0
    sigma_y: float = 0.0
    wavelet: WaveletBPConfig = field(default_factory=WaveletBPConfig)


@dataclass
class PurificationConfig:
    backend: str = "module"
    model_type: str = "module"
    model_module: str = ""
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_input_range: str = "zero_one"
    model_output_range: str = "zero_one"
    model_time_mode: str = "rescaled_sigma"
    model_input_kind: str = "cm_scaled"
    model_prediction_type: str = "network_output"
    class_cond: bool = False
    default_class_label: int = 0
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    wavelet_noise: WaveletNoiseConfig = field(default_factory=WaveletNoiseConfig)
    bp: BPConfig = field(default_factory=BPConfig)


@dataclass
class DatasetConfig:
    name: str = "image_folder"
    root: str = "data/images"
    split: str = "val"
    image_size: Optional[int] = None
    batch_size: int = 1
    num_workers: int = 4
    max_samples: int = 0
    glob: str = "*.png"


@dataclass
class ClassifierConfig:
    name: str = "torchvision_resnet50"
    module: str = ""
    kwargs: Dict[str, Any] = field(default_factory=dict)
    input_size: int = 224
    resize_short: int = 256
    normalize: str = "imagenet"


@dataclass
class AttackConfig:
    name: str = "pgd"
    eps: float = 8.0 / 255.0
    steps: int = 10
    step_size: float = 2.0 / 255.0
    random_start: bool = True
    norm: str = "linf"


@dataclass
class EvaluationConfig:
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs/camp/baseline"
    save_images: bool = False
    lightweight: bool = True


@dataclass
class ExperimentConfig:
    experiment_name: str = "cm_purification_baseline"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    purification: PurificationConfig = field(default_factory=PurificationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


_UNRESOLVED_ENV_PATTERN = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)|%[A-Za-z_][A-Za-z0-9_]*%")


def _merge_dict(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge_dict(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _coerce_dataclass(cls: type, data: Mapping[str, Any]):
    kwargs: Dict[str, Any] = {}
    fields = getattr(cls, "__dataclass_fields__")
    for name, meta in fields.items():
        if name not in data:
            continue
        value = data[name]
        if meta.default is not dataclasses.MISSING:
            default_value = meta.default
        elif meta.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
            default_value = meta.default_factory()  # type: ignore[misc]
        else:
            default_value = None
        if is_dataclass(default_value) and isinstance(value, Mapping):
            kwargs[name] = _coerce_dataclass(type(default_value), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _find_unresolved_env_vars(text: str) -> List[str]:
    matches = []
    for match in _UNRESOLVED_ENV_PATTERN.finditer(text):
        token = match.group(0)
        name = token
        if token.startswith("${") and token.endswith("}"):
            name = token[2:-1]
        elif token.startswith("$"):
            name = token[1:]
        elif token.startswith("%") and token.endswith("%"):
            name = token[1:-1]
        if name and name not in matches:
            matches.append(name)
    return matches


def _find_unresolved_env_vars_in_obj(obj: Any) -> List[str]:
    found: List[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            for name in _find_unresolved_env_vars(value):
                if name not in found:
                    found.append(name)
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(obj)
    return found


def _read_config_file(path: Path) -> Dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8")
    text = os.path.expandvars(raw_text)
    suffix = path.suffix.lower()
    if suffix == ".json":
        loaded = json.loads(text)
        unresolved = _find_unresolved_env_vars_in_obj(loaded)
        if unresolved:
            names = ", ".join(unresolved)
            raise ValueError(
                f"Unresolved environment variable(s) in config {path}: {names}. "
                "Set them before running, or replace them with concrete paths."
            )
        return loaded
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML config files") from exc
        loaded = yaml.safe_load(text)
        loaded = loaded or {}
        unresolved = _find_unresolved_env_vars_in_obj(loaded)
        if unresolved:
            names = ", ".join(unresolved)
            raise ValueError(
                f"Unresolved environment variable(s) in config {path}: {names}. "
                "Set them before running, or replace them with concrete paths."
            )
        return loaded
    raise ValueError(f"Unsupported config suffix: {path.suffix}")


def load_experiment_config(path: Path, overrides: Optional[Mapping[str, Any]] = None) -> ExperimentConfig:
    raw = _read_config_file(path)
    if overrides:
        raw = _merge_dict(raw, overrides)
    return _coerce_dataclass(ExperimentConfig, raw)


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    return asdict(config)
