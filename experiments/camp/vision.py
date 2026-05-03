from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Subset
from torch.utils.data import Dataset

from .config import AttackConfig, ClassifierConfig


Tensor = torch.Tensor


class TinyConvClassifier(torch.nn.Module):
    """Small untrained classifier for wiring tests."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = torch.nn.Linear(32, int(num_classes))

    def forward(self, x: Tensor) -> Tensor:
        feat = self.net(x).flatten(1)
        return self.head(feat)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_to_tensor(img: Image.Image) -> Tensor:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class _TransformToTensor:
    def __call__(self, img: Image.Image) -> Tensor:
        return pil_to_tensor(img)


def tensor_to_pil(x: Tensor) -> Image.Image:
    arr = (x.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def to_minus_one_one(x: Tensor) -> Tensor:
    return x * 2.0 - 1.0


def to_zero_one(x: Tensor) -> Tensor:
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: Path,
        pattern: str = "*.png",
        image_size: Optional[int] = None,
        max_samples: int = 0,
    ) -> None:
        self.root = Path(root)
        paths = sorted(self.root.glob(pattern))
        if not paths:
            collected: List[Path] = []
            for suffix in ("*.png", "*.jpg", "*.jpeg", "*.JPEG", "*.bmp", "*.webp"):
                collected.extend(sorted(self.root.glob(suffix)))
            paths = sorted(set(collected))
        if max_samples and max_samples > 0:
            paths = paths[: int(max_samples)]
        if not paths:
            raise FileNotFoundError(f"No images found in {self.root}")
        self.paths = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if self.image_size:
            image = image.resize((int(self.image_size), int(self.image_size)), Image.BICUBIC)
        return {
            "image": pil_to_tensor(image),
            "path": str(path),
            "label": -1,
        }


class ClassFolderDataset(Dataset):
    """Small dataset layout: root/class_name/*.png.

    Labels are inferred from sorted class folder names and stored in class_to_idx.
    """

    def __init__(
        self,
        root: Path,
        pattern: str = "*",
        image_size: Optional[int] = None,
        max_samples: int = 0,
    ) -> None:
        self.root = Path(root)
        classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not classes:
            raise FileNotFoundError(f"No class folders found in {self.root}")
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        paths: List[Tuple[Path, int]] = []
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        for class_name in classes:
            class_dir = self.root / class_name
            for path in sorted(class_dir.glob(pattern)):
                if path.is_file() and path.suffix.lower() in extensions:
                    paths.append((path, self.class_to_idx[class_name]))
        if max_samples and max_samples > 0:
            paths = paths[: int(max_samples)]
        if not paths:
            raise FileNotFoundError(f"No images found under class folders in {self.root}")
        self.samples = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.image_size:
            image = image.resize((int(self.image_size), int(self.image_size)), Image.BICUBIC)
        return {
            "image": pil_to_tensor(image),
            "path": str(path),
            "label": int(label),
        }


def imagenet_normalize(x: Tensor) -> Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def cifar10_normalize(x: Tensor) -> Tensor:
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.2471, 0.2435, 0.2616], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def classifier_preprocess(x: Tensor, config: ClassifierConfig) -> Tensor:
    out = x
    if config.resize_short > 0:
        h, w = out.shape[-2:]
        short = min(h, w)
        scale = float(config.resize_short) / float(short)
        out = F.interpolate(
            out,
            size=(max(1, int(round(h * scale))), max(1, int(round(w * scale)))),
            mode="bilinear",
            align_corners=False,
        )
    if config.input_size > 0:
        h, w = out.shape[-2:]
        size = int(config.input_size)
        if h < size or w < size:
            out = F.interpolate(out, size=(max(h, size), max(w, size)), mode="bilinear", align_corners=False)
            h, w = out.shape[-2:]
        top = (h - size) // 2
        left = (w - size) // 2
        out = out[:, :, top : top + size, left : left + size]
    if config.normalize == "imagenet":
        out = imagenet_normalize(out)
    elif config.normalize == "cifar10":
        out = cifar10_normalize(out)
    elif config.normalize in {"none", ""}:
        pass
    else:
        raise ValueError(f"Unsupported classifier normalization: {config.normalize}")
    return out


def build_classifier(config: ClassifierConfig, device: torch.device) -> torch.nn.Module:
    if config.module:
        import importlib

        module_name, class_name = config.module.split(":", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        model = cls(**config.kwargs)
    else:
        if config.name == "tiny_conv_untrained":
            model = TinyConvClassifier(num_classes=int(config.kwargs.get("num_classes", 10)))
            return model.eval().to(device)
        if config.name == "cifar_resnet56":
            from .cifar_models import build_cifar_resnet56

            model = build_cifar_resnet56(**config.kwargs)
            return model.eval().to(device)

        from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

        if config.name == "torchvision_resnet18":
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif config.name == "torchvision_resnet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif config.name == "torchvision_resnet18_untrained":
            model = resnet18(weights=None, num_classes=int(config.kwargs.get("num_classes", 10)))
        elif config.name == "torchvision_resnet50_untrained":
            model = resnet50(weights=None, num_classes=int(config.kwargs.get("num_classes", 10)))
        else:
            raise ValueError(f"Unsupported classifier: {config.name}")
    return model.eval().to(device)


def classifier_logits(model: torch.nn.Module, x: Tensor, config: ClassifierConfig) -> Tensor:
    return model(classifier_preprocess(x, config))


@torch.no_grad()
def classify(model: torch.nn.Module, x: Tensor, config: ClassifierConfig) -> Tuple[Tensor, Tensor]:
    logits = classifier_logits(model, x, config)
    probs = logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf


def pgd_linf_attack(
    model: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    classifier_config: ClassifierConfig,
    attack_config: AttackConfig,
) -> Tensor:
    eps = float(attack_config.eps)
    step_size = float(attack_config.step_size)
    steps = int(attack_config.steps)
    if eps < 0.0:
        raise ValueError("attack.eps must be non-negative")
    if step_size < 0.0:
        raise ValueError("attack.step_size must be non-negative")
    if steps < 0:
        raise ValueError("attack.steps must be non-negative")
    if attack_config.random_start:
        delta = torch.empty_like(x).uniform_(-eps, eps)
    else:
        delta = torch.zeros_like(x)
    delta = (x + delta).clamp(0.0, 1.0) - x
    delta.requires_grad_(True)

    for _ in range(steps):
        logits = classifier_logits(model, (x + delta).clamp(0.0, 1.0), classifier_config)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        delta.data = (delta + step_size * grad.sign()).clamp(-eps, eps)
        delta.data = (x + delta.data).clamp(0.0, 1.0) - x
    return (x + delta.detach()).clamp(0.0, 1.0)


def run_attack(
    model: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    classifier_config: ClassifierConfig,
    attack_config: AttackConfig,
) -> Tensor:
    if attack_config.name == "none":
        return x.detach()
    if attack_config.name == "pgd" and attack_config.norm == "linf":
        return pgd_linf_attack(model, x, y, classifier_config, attack_config)
    raise ValueError(f"Unsupported attack: {attack_config.name}/{attack_config.norm}")


def build_dataset(
    name: str,
    root: Path,
    pattern: str,
    image_size: Optional[int],
    max_samples: int,
    split: str,
):
    if name == "class_folder":
        return ClassFolderDataset(root=root, pattern=pattern, image_size=image_size, max_samples=max_samples)
    if name == "image_folder":
        return ImageFolderDataset(root=root, pattern=pattern, image_size=image_size, max_samples=max_samples)
    if name == "torchvision_cifar10":
        from torchvision.datasets import CIFAR10

        train = str(split).lower() == "train"
        if not root.exists():
            raise FileNotFoundError(f"CIFAR-10 root not found: {root}")
        dataset_dir = root / "cifar-10-batches-py"
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"CIFAR-10 python batches not found: {dataset_dir}. "
                "Download/extract CIFAR-10 there or update dataset.root."
            )
        dataset = CIFAR10(root=str(root), train=train, download=False, transform=_TransformToTensor())
        if max_samples and max_samples > 0:
            dataset = Subset(dataset, list(range(min(int(max_samples), len(dataset)))))
        return dataset
    raise ValueError(f"Unsupported dataset.name: {name}")
