from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_experiment_config


def _check_path(label: str, path_text: str, must_be_file: bool = False, must_be_dir: bool = False) -> bool:
    path = Path(path_text)
    ok = path.exists()
    if ok and must_be_file:
        ok = path.is_file()
    if ok and must_be_dir:
        ok = path.is_dir()
    status = "OK" if ok else "MISSING"
    print(f"[{status}] {label}: {path}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CAMP-CM config paths before running experiments")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)

    ok = True
    ok &= _check_path("dataset.root", cfg.dataset.root, must_be_dir=True)
    if cfg.dataset.name == "torchvision_cifar10":
        ok &= _check_path("CIFAR-10 python batches", str(Path(cfg.dataset.root) / "cifar-10-batches-py"), must_be_dir=True)

    classifier_ckpt = str(cfg.classifier.kwargs.get("checkpoint", ""))
    if classifier_ckpt:
        ok &= _check_path("classifier.checkpoint", classifier_ckpt, must_be_file=True)

    model_kwargs = cfg.purification.model_kwargs
    ctm_repo = str(model_kwargs.get("ctm_repo", ""))
    cm_ckpt = str(model_kwargs.get("checkpoint", ""))
    if ctm_repo:
        ok &= _check_path("purification.model_kwargs.ctm_repo", ctm_repo, must_be_dir=True)
        ok &= _check_path(
            "ctm_repo/code/cm/script_util.py",
            str(Path(ctm_repo) / "code" / "cm" / "script_util.py"),
            must_be_file=True,
        )
    if cm_ckpt:
        ok &= _check_path("purification.model_kwargs.checkpoint", cm_ckpt, must_be_file=True)

    if not ok:
        raise SystemExit(1)
    print("All required paths look good.")


if __name__ == "__main__":
    main()

