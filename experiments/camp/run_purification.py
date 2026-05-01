from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .cm_purifier import CMPurifier
    from .config import config_to_dict, load_experiment_config
    from .model_adapters import build_generative_model
    from .vision import (
        ClassFolderDataset,
        ImageFolderDataset,
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
    from experiments.camp.model_adapters import build_generative_model
    from experiments.camp.vision import (
        ClassFolderDataset,
        ImageFolderDataset,
        build_classifier,
        classify,
        run_attack,
        set_seed,
        tensor_to_pil,
        to_minus_one_one,
        to_zero_one,
    )


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
    device = torch.device(config.evaluation.device if torch.cuda.is_available() or config.evaluation.device == "cpu" else "cpu")

    if config.dataset.name == "class_folder":
        dataset = ClassFolderDataset(
            root=Path(config.dataset.root),
            pattern=config.dataset.glob,
            image_size=config.dataset.image_size,
            max_samples=config.dataset.max_samples,
        )
    elif config.dataset.name == "image_folder":
        dataset = ImageFolderDataset(
            root=Path(config.dataset.root),
            pattern=config.dataset.glob,
            image_size=config.dataset.image_size,
            max_samples=config.dataset.max_samples,
        )
    else:
        raise ValueError(f"Unsupported dataset.name: {config.dataset.name}")
    loader = DataLoader(
        dataset,
        batch_size=int(config.dataset.batch_size),
        shuffle=False,
        num_workers=int(config.dataset.num_workers),
    )
    classifier = build_classifier(config.classifier, device=device)
    gen_model = build_generative_model(config.purification, device=device)
    purifier = CMPurifier(config.purification, model=gen_model, device=device)

    samples: List[Dict[str, object]] = []
    clean_correct = 0
    adv_same = 0
    purified_same = 0
    recovered = 0
    attacked = 0

    image_dir = output_dir / "images"
    if config.evaluation.save_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc=config.experiment_name):
        x_clean = batch["image"].to(device).float()
        paths = list(batch["path"])
        pred_clean, conf_clean = classify(classifier, x_clean, config.classifier)
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
            clean_correct += 1
            adv_same += int(pred_adv[i].item() == pred_clean[i].item())
            purified_same += int(recover_success)
            record = {
                "path": path,
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
            if config.evaluation.save_images:
                stem = Path(path).stem
                tensor_to_pil(x_clean[i]).save(image_dir / f"{stem}_clean.png")
                tensor_to_pil(x_adv[i]).save(image_dir / f"{stem}_adv.png")
                tensor_to_pil(x_purified[i]).save(image_dir / f"{stem}_purified.png")

    aggregate = {
        "num_samples": len(samples),
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
