from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .cm_purifier import CMPurifier
    from .config import config_to_dict, load_experiment_config
    from .adapters.registry import build_backend
    from .vision import (
        build_dataset,
        build_classifier,
        classify,
        run_attack,
        set_seed,
        tensor_to_pil,
        to_minus_one_one,
        to_zero_one,
    )
except ImportError:
    from experiments.camp.cm_purifier import CMPurifier
    from experiments.camp.config import config_to_dict, load_experiment_config
    from experiments.camp.adapters.registry import build_backend
    from experiments.camp.vision import (
        build_dataset,
        build_classifier,
        classify,
        run_attack,
        set_seed,
        tensor_to_pil,
        to_minus_one_one,
        to_zero_one,
    )


def save_triplet(clean, adv, purified, path: Path) -> None:
    from PIL import Image

    images = [tensor_to_pil(clean), tensor_to_pil(adv), tensor_to_pil(purified)]
    canvas = Image.new("RGB", (sum(img.width for img in images), max(img.height for img in images)))
    x_offset = 0
    for image in images:
        canvas.paste(image, (x_offset, 0))
        x_offset += image.width
    canvas.save(path)


def _resolve_device(requested: str) -> torch.device:
    requested = str(requested or "cpu").strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] Requested device '{requested}' but CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _unpack_batch(batch, dataset_name: str, sample_offset: int, device: torch.device) -> Tuple[torch.Tensor, List[str], torch.Tensor | None]:
    if isinstance(batch, dict):
        x_clean = batch["image"].to(device).float()
        raw_paths = batch.get("path")
        if raw_paths is None:
            paths = [f"{dataset_name}:{sample_offset + i:06d}" for i in range(x_clean.shape[0])]
        else:
            paths = [str(item) for item in raw_paths]
        labels = batch.get("label")
        labels_t = labels.to(device).long() if torch.is_tensor(labels) else None
        return x_clean, paths, labels_t

    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x_clean, labels = batch[0], batch[1]
        x_clean = x_clean.to(device).float()
        labels_t = labels.to(device).long() if torch.is_tensor(labels) else None
        paths = [f"{dataset_name}:{sample_offset + i:06d}" for i in range(x_clean.shape[0])]
        return x_clean, paths, labels_t

    raise TypeError(f"Unsupported dataloader batch type: {type(batch)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAMP CM adversarial purification runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Load config and print it without model/data execution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.output_dir is not None:
        config.evaluation.output_dir = str(args.output_dir)
    if args.device:
        config.evaluation.device = args.device
    if args.max_samples > 0:
        config.dataset.max_samples = args.max_samples
    if args.save_images:
        config.evaluation.save_images = True

    if args.dry_run:
        print(json.dumps(config_to_dict(config), ensure_ascii=False, indent=2))
        return

    output_dir = Path(config.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as handle:
        json.dump(config_to_dict(config), handle, ensure_ascii=False, indent=2)

    set_seed(config.evaluation.seed)
    device = _resolve_device(config.evaluation.device)

    dataset = build_dataset(
        name=config.dataset.name,
        root=Path(config.dataset.root),
        pattern=config.dataset.glob,
        image_size=config.dataset.image_size,
        max_samples=config.dataset.max_samples,
        split=config.dataset.split,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.dataset.batch_size),
        shuffle=False,
        num_workers=int(config.dataset.num_workers),
    )
    classifier = build_classifier(config.classifier, device=device)
    gen_model = build_backend(config.purification, device=device)
    purifier = CMPurifier(config.purification, model=gen_model, device=device)

    samples: List[Dict[str, object]] = []
    clean_correct = 0
    clean_labeled = 0
    adv_same = 0
    purified_same = 0
    recovered = 0
    attacked = 0

    image_dir = output_dir / "images"
    if config.evaluation.save_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc=config.experiment_name):
        x_clean, paths, labels = _unpack_batch(
            batch,
            dataset_name=config.dataset.name,
            sample_offset=len(samples),
            device=device,
        )
        if len(paths) != x_clean.shape[0]:
            paths = [f"{config.dataset.name}:{len(samples) + i:06d}" for i in range(x_clean.shape[0])]
        pred_clean, conf_clean = classify(classifier, x_clean, config.classifier)
        labels_valid = labels is not None and labels.numel() == pred_clean.numel() and bool(torch.all(labels >= 0).item())
        y_ref = pred_clean.detach()
        x_adv = run_attack(classifier, x_clean, y_ref, config.classifier, config.attack)
        pred_adv, conf_adv = classify(classifier, x_adv, config.classifier)

        classes = y_ref if config.purification.class_cond else None
        result = purifier.purify(to_minus_one_one(x_adv), classes=classes, return_trace=True)
        x_purified = to_zero_one(result.x_purified)
        pred_purified, conf_purified = classify(classifier, x_purified, config.classifier)

        for i, path in enumerate(paths):
            attack_success = bool(pred_adv[i].item() != pred_clean[i].item())
            recover_success = bool(pred_purified[i].item() == pred_clean[i].item())
            attacked += int(attack_success)
            recovered += int(attack_success and recover_success)
            true_label = int(labels[i].item()) if labels_valid else None
            if true_label is not None:
                clean_labeled += 1
                clean_correct += int(pred_clean[i].item() == true_label)
            adv_same += int(pred_adv[i].item() == pred_clean[i].item())
            purified_same += int(recover_success)
            record = {
                "path": path,
                "true_label": true_label,
                "clean_pred": int(pred_clean[i].item()),
                "adv_pred": int(pred_adv[i].item()),
                "purified_pred": int(pred_purified[i].item()),
                "clean_conf": float(conf_clean[i].item()),
                "adv_conf": float(conf_adv[i].item()),
                "purified_conf": float(conf_purified[i].item()),
                "attack_success": attack_success,
                "recover_to_clean_pred": recover_success,
                "sigma_schedule": result.trace.get("sigma", []),
            }
            samples.append(record)
            artifacts: Dict[str, object] = {}
            if config.evaluation.save_images:
                stem = Path(path).stem
                if not stem or ":" in stem:
                    stem = f"sample_{len(samples):06d}"
                clean_path = image_dir / f"{stem}_clean.png"
                adv_path = image_dir / f"{stem}_adv.png"
                purified_path = image_dir / f"{stem}_purified.png"
                triplet_path = image_dir / f"{stem}_triplet.png"
                tensor_to_pil(x_clean[i]).save(clean_path)
                tensor_to_pil(x_adv[i]).save(adv_path)
                tensor_to_pil(x_purified[i]).save(purified_path)
                save_triplet(x_clean[i], x_adv[i], x_purified[i], triplet_path)
                artifacts = {
                    "clean": str(clean_path),
                    "adv": str(adv_path),
                    "purified": str(purified_path),
                    "triplet": str(triplet_path),
                }
            record["artifacts"] = artifacts

    aggregate = {
        "num_samples": len(samples),
        "clean_accuracy_on_labeled": clean_correct / max(1, clean_labeled),
        "attack_success_rate": attacked / max(1, len(samples)),
        "adv_same_as_clean_rate": adv_same / max(1, len(samples)),
        "recover_rate_on_attacked": recovered / max(1, attacked),
        "purified_same_as_clean_rate": purified_same / max(1, len(samples)),
    }
    summary = {
        "experiment_name": config.experiment_name,
        "aggregate": aggregate,
        "samples": samples,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Done. Summary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
