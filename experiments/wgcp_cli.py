import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wavelet-Guided Consistency Purification (small-scale debug runner)"
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory of adversarial images")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/wgcp_small"))
    parser.add_argument("--clean_dir", type=Path, default=None, help="Optional clean-image directory for metrics")
    parser.add_argument("--max_images", type=int, default=4)
    parser.add_argument("--glob", type=str, default="*.png")
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--num_diffusion_steps", type=int, default=1000)
    parser.add_argument("--t_star", type=int, default=40)
    parser.add_argument("--self_correct_k", type=int, default=1)
    parser.add_argument(
        "--self_correct_steps",
        type=str,
        default="",
        help="Comma-separated time steps, e.g. 60,40,20",
    )
    parser.add_argument(
        "--t_bridge",
        type=int,
        default=-1,
        help="Bridge step t1 for interleaved projection. If <=0, auto uses round(t_star*0.25).",
    )
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--mix", type=float, default=0.45, help="x_t -> denoised blend ratio")
    parser.add_argument(
        "--predictor_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "module"],
        help="Predictor backend: built-in gaussian debug model or external module",
    )
    parser.add_argument(
        "--predictor_module",
        type=str,
        default="",
        help="When predictor_type=module, use format 'package.module:ClassName'",
    )
    parser.add_argument(
        "--predictor_kwargs_json",
        type=str,
        default="{}",
        help="JSON string for predictor constructor kwargs when predictor_type=module",
    )
    parser.add_argument(
        "--predictor_image_size",
        type=int,
        default=0,
        help="If >0, resize input to this square size before predictor, then resize back",
    )
    parser.add_argument(
        "--hf_preserve",
        type=float,
        default=0.35,
        help="How much original HF to preserve in replacement (0=full rewrite, 1=full keep)",
    )
    parser.add_argument(
        "--hf_shrink",
        type=float,
        default=0.6,
        help="Soft-threshold factor for original HF before blending (relative to robust sigma)",
    )
    parser.add_argument(
        "--replacement_mode",
        type=str,
        default="adaptive_ms",
        choices=[
            "hard",
            "fused",
            "adaptive_ms",
            "adaptive_ms_guided",
            "adaptive_ms_w2lite",
            "adaptive_ms_edge",
            "adaptive_ms_modmax",
            "adaptive_ms_prior",
        ],
        help="HF replacement strategy: hard / fused / adaptive multi-scale / guided adaptive multi-scale / W2-lite predictor-HF blend / predictor-guided shrinkage",
    )
    parser.add_argument(
        "--ablation_ll_source",
        type=str,
        default="orig",
        choices=["orig", "hat"],
        help="Ablation only: LL source for IDWT anchor. 'orig'=paper default, 'hat'=predictor LL.",
    )
    parser.add_argument(
        "--ablation_hard_hf_source",
        type=str,
        default="pred",
        choices=["pred", "orig"],
        help="Ablation only under replacement_mode=hard: HF source. 'pred'=paper default, 'orig'=keep original HF.",
    )
    parser.add_argument("--ms_levels", type=int, default=3, help="Wavelet decomposition levels for adaptive_ms mode.")
    parser.add_argument(
        "--ms_gamma_levels",
        type=str,
        default="1.6,1.2,0.9",
        help="Gamma schedule for adaptive_ms soft-shrinkage by level1->L (comma separated).",
    )
    parser.add_argument(
        "--ms_w_min",
        type=float,
        default=0.05,
        help="Minimum predictor weight used by guided adaptive multi-scale LL fusion.",
    )
    parser.add_argument(
        "--ms_w_max",
        type=float,
        default=0.95,
        help="Maximum predictor weight used by guided adaptive multi-scale LL fusion.",
    )
    parser.add_argument(
        "--ms_ll_alpha",
        type=float,
        default=0.08,
        help="Deep LL blend alpha in adaptive_ms: LL=(1-a)*LL_pred + a*LL_orig.",
    )
    parser.add_argument("--ms_eps", type=float, default=1e-6, help="Numerical epsilon used in MAD estimation for adaptive_ms.")
    parser.add_argument(
        "--ms_ll_gate_tau",
        type=float,
        default=0.75,
        help="Guided mode LL gate center; larger values make predictor LL injection more conservative.",
    )
    parser.add_argument(
        "--ms_ll_gate_gain",
        type=float,
        default=4.0,
        help="Guided mode LL gate sharpness; larger values make LL switching more selective.",
    )
    parser.add_argument(
        "--ms_hf_pred_levels",
        type=str,
        default="0.20,0.12,0.08",
        help="Guided mode predictor HF residual mix schedule by level1->L (comma separated).",
    )
    parser.add_argument(
        "--ms_w2_hf_mix_levels",
        type=str,
        default="1.0,1.0,1.0",
        help="W2-lite predictor HF mix schedule by level1->L (1=all predictor HF, 0=all C1-shrunk original HF).",
    )
    parser.add_argument(
        "--ms_hf_gate_tau",
        type=float,
        default=0.5,
        help="Guided mode HF gate center for predictor residual reinjection.",
    )
    parser.add_argument(
        "--ms_hf_gate_gain",
        type=float,
        default=4.0,
        help="Guided mode HF gate sharpness for predictor residual reinjection.",
    )
    parser.add_argument(
        "--ms_edge_eta_levels",
        type=str,
        default="0.5,0.5,0.5",
        help="Edge-aware mode threshold reduction for HL/LH by level1->L (comma separated).",
    )
    parser.add_argument(
        "--ms_edge_eta_hh_levels",
        type=str,
        default="0.3,0.3,0.3",
        help="Edge-aware mode threshold reduction for HH by level1->L (comma separated).",
    )
    parser.add_argument(
        "--ms_edge_sigma_divisor",
        type=float,
        default=30.0,
        help="Spatial divisor for local edge-energy Gaussian sigma: sigma ~= min(H,W)/divisor.",
    )
    parser.add_argument(
        "--ms_edge_alpha_min",
        type=float,
        default=0.1,
        help="Minimum local alpha used in edge-aware shrinkage.",
    )
    parser.add_argument(
        "--ms_edge_channel_agg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aggregate RGB channel energy when building wavelet edge maps.",
    )
    parser.add_argument(
        "--ms_modmax_threshold",
        type=float,
        default=0.15,
        help="Minimum normalized modulus response kept as edge maxima in adaptive_ms_modmax.",
    )
    parser.add_argument(
        "--ms_modmax_boost_levels",
        type=str,
        default="0.10,0.08,0.05",
        help="Boost schedule for HL/LH modulus-maxima coefficients by level1->L (comma separated).",
    )
    parser.add_argument("--patch_mode", action="store_true", help="Enable Patch-WGCP purification pipeline.")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size for Patch-WGCP.")
    parser.add_argument("--patch_stride", type=int, default=32, help="Patch stride for Patch-WGCP.")
    parser.add_argument("--patch_batch_size", type=int, default=64, help="Patch CTM inference batch size.")
    parser.add_argument(
        "--patch_weight_sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma (in pixels) for patch blending; <=0 means auto (patch_size/6).",
    )
    parser.add_argument(
        "--patch_lowfreq_alpha",
        type=float,
        default=0.1,
        help="Low-frequency blend weight between global-base and patch branch (0..1).",
    )
    parser.add_argument(
        "--patch_ll_source",
        type=str,
        default="hat",
        choices=["hat", "orig"],
        help="LL source used inside each patch branch before folding.",
    )
    parser.add_argument(
        "--patch_pad_mode",
        type=str,
        default="reflect",
        choices=["reflect", "replicate", "constant"],
        help="Padding mode used before unfold when shape is not divisible by patch stride.",
    )
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument(
        "--archive_dir",
        type=Path,
        default=Path.home() / ".camp_runs" / "CAMP",
        help="Directory for archived run records (default outside repo).",
    )
    parser.add_argument("--archive_tag", type=str, default="", help="Optional tag appended to archive filename.")
    parser.add_argument("--disable_archive", action="store_true", help="Disable extra archived run record.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_steps(args: argparse.Namespace) -> List[int]:
    if args.self_correct_k <= 0:
        return []

    if args.self_correct_steps.strip():
        steps = [int(x.strip()) for x in args.self_correct_steps.split(",") if x.strip()]
        steps = steps[: args.self_correct_k]
        if len(steps) < args.self_correct_k:
            raise ValueError("self_correct_steps count is smaller than self_correct_k")
    else:
        bridge = args.t_bridge if args.t_bridge > 0 else max(2, int(round(args.t_star * 0.25)))
        if args.self_correct_k == 1:
            steps = [bridge]
        else:
            tail = np.linspace(bridge, 2, args.self_correct_k, dtype=int).tolist()
            steps = []
            for step in tail:
                if not steps or steps[-1] != step:
                    steps.append(step)
    return [int(step) for step in steps]
