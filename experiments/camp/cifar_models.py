from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CIFARResNet(nn.Module):
    def __init__(self, depth: int = 56, num_classes: int = 10) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("CIFAR ResNet depth must satisfy depth=6n+2")
        n = (depth - 2) // 6
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)
        self.linear = nn.Linear(64, int(num_classes))

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, current_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view(out.size(0), -1)
        return self.linear(out)


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "net", "classifier"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise RuntimeError("Could not find a state_dict in checkpoint")


def _strip_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    prefixes = ("module.", "model.", "net.", "classifier.")
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        out[new_key] = value
    return out


def build_cifar_resnet56(
    checkpoint: str = "",
    num_classes: int = 10,
    strict_load: bool = False,
    map_location: str = "cpu",
) -> nn.Module:
    model = CIFARResNet(depth=56, num_classes=num_classes)
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if ckpt_path.is_dir():
            candidates = sorted(
                list(ckpt_path.glob("*.pth")) + list(ckpt_path.glob("*.pt")) + list(ckpt_path.glob("*.ckpt"))
            )
            if not candidates:
                raise FileNotFoundError(f"No checkpoint file found in {ckpt_path}")
            ckpt_path = candidates[0]
        state_obj = torch.load(str(ckpt_path), map_location=map_location)
        state = _strip_prefixes(_extract_state_dict(state_obj))
        load_info = model.load_state_dict(state, strict=bool(strict_load))
        if not strict_load:
            missing = getattr(load_info, "missing_keys", [])
            unexpected = getattr(load_info, "unexpected_keys", [])
            if missing or unexpected:
                print(
                    "[build_cifar_resnet56] non-strict load:",
                    {"missing_keys": len(missing), "unexpected_keys": len(unexpected), "checkpoint": str(ckpt_path)},
                )
    return model

