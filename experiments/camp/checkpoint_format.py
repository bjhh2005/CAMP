from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Optional


def guess_checkpoint_format(path: Path) -> str:
    if path.is_dir():
        if (path / "model_index.json").exists():
            return "diffusers_dir"
        candidates = []
        for pattern in ("*.pt", "*.pth", "*.ckpt"):
            candidates.extend(sorted(path.glob(pattern)))
        if candidates:
            return f"directory_with_checkpoint:{guess_checkpoint_format(candidates[0])}"
        return "directory_unknown"

    if not path.exists():
        return "missing"

    head = path.read_bytes()[:16]
    if head.startswith(b"PK\x03\x04"):
        return "pytorch_zip_or_diffusers_weight"
    if head[:1] in {b"\x80"}:
        return "pytorch_pickle_or_python_pickle"
    if head[:1] in {b"\xa4", b"\xde", b"\x82", b"\x83", b"\x84"}:
        return "flax_msgpack_probable"
    if b"version https://git-lfs.github.com/spec" in path.read_bytes()[:128]:
        return "git_lfs_pointer_not_downloaded"
    return f"unknown_first_bytes:{head.hex()}"


def recommended_backend(fmt: str) -> str:
    if "flax_msgpack" in fmt:
        return "openai_cifar_jax or another JAX/Flax adapter"
    if "pytorch" in fmt:
        return "sony_cm, openai_pytorch_cm, or another PyTorch adapter matching the source repo"
    if "diffusers" in fmt:
        return "diffusers adapter"
    if "git_lfs" in fmt:
        return "download the real checkpoint file first"
    return "unknown; inspect the checkpoint source"


def main() -> None:
    parser = argparse.ArgumentParser(description="Guess checkpoint format")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    fmt = guess_checkpoint_format(args.path)
    print(f"format: {fmt}")
    print(f"recommended_backend: {recommended_backend(fmt)}")


if __name__ == "__main__":
    main()

