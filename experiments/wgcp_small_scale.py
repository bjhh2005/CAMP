import json
from pathlib import Path
from typing import Dict

import pywt
import torch
from tqdm import tqdm

try:
    from .wgcp_cli import parse_args, parse_steps
    from .wgcp_predictor import build_alpha_bars, build_predictor
    from .wgcp_purify import process_one_image
    from .wgcp_utils import archive_run_record, build_run_meta, ensure_dir, list_images, set_seed
except ImportError:
    from wgcp_cli import parse_args, parse_steps
    from wgcp_predictor import build_alpha_bars, build_predictor
    from wgcp_purify import process_one_image
    from wgcp_utils import archive_run_record, build_run_meta, ensure_dir, list_images, set_seed


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)

    if args.t_star <= 0 or args.t_star >= args.num_diffusion_steps:
        raise ValueError("t_star must be in (0, num_diffusion_steps)")

    device = torch.device(args.device)
    alpha_bars = build_alpha_bars(args.num_diffusion_steps).to(device)

    wavelets = pywt.wavelist(kind="discrete")
    if args.wavelet not in wavelets:
        raise ValueError(f"unknown wavelet: {args.wavelet}")

    loop_steps = parse_steps(args)
    for step in loop_steps:
        if step <= 0 or step >= args.num_diffusion_steps:
            raise ValueError("all self-correct steps must be in (0, num_diffusion_steps)")

    image_paths = list_images(args.input_dir, args.glob, args.max_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    predictor = build_predictor(args, device=device)

    results = []
    for path in tqdm(image_paths, desc="WGCP small-scale"):
        results.append(process_one_image(path, args, alpha_bars, predictor, loop_steps, device))

    run_meta = build_run_meta(args, script_path=Path(__file__).resolve())
    summary_obj: Dict[str, object] = {
        "run_meta": run_meta,
        "samples": results,
    }

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_obj, handle, ensure_ascii=False, indent=2)

    archive_path = archive_run_record(args, summary_obj, script_name="wgcp_small_scale")

    print(f"Done. Results saved to: {args.output_dir}")
    print(f"Summary: {summary_path}")
    if archive_path is not None:
        print(f"Archived summary: {archive_path}")


if __name__ == "__main__":
    main()
