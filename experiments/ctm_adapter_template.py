"""
Template adapter for external CTM repository integration.

Usage example (after you implement this class):
python experiments/wgcp_attack_eval.py \
  --predictor_type module \
  --predictor_module experiments.ctm_adapter_template:CTMRepoPredictor \
  --predictor_kwargs_json '{"ctm_repo":"/path/to/third_party/ctm","checkpoint":"/data/model_cache/ctm/ctm_imagenet64_ema999.pt"}' \
  --predictor_image_size 64
"""

from __future__ import annotations

from pathlib import Path

import torch


class CTMRepoPredictor:
    """
    Bridge class for plugging an external CTM implementation into WGCP.

    Required call signature:
      __call__(x_t: Tensor[B,C,H,W], t_index: int) -> Tensor[B,C,H,W]

    Notes:
    - Most public CTM checkpoints are trained for generation datasets like ImageNet64.
    - Keep `predictor_image_size=64` when using ImageNet64 checkpoints.
    - This file is a template: fill in model loading/inference code from your chosen CTM repo.
    """

    def __init__(self, ctm_repo: str, checkpoint: str, device: str = "cuda", **kwargs):
        self.ctm_repo = Path(ctm_repo)
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)

        if not self.ctm_repo.exists():
            raise FileNotFoundError(f"CTM repo not found: {self.ctm_repo}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")

        # TODO: 1) import CTM modules from self.ctm_repo
        # TODO: 2) construct model and load checkpoint
        # TODO: 3) set eval mode and no-grad inference path
        self.model = None

    @torch.no_grad()
    def __call__(self, x_t: torch.Tensor, t_index: int) -> torch.Tensor:
        # TODO: map WGCP t_index to CTM's time embedding format if needed.
        if self.model is None:
            raise NotImplementedError(
                "CTMRepoPredictor is a template. Please implement model loading and inference first."
            )

        # Example shape contract only:
        x0_hat = self.model(x_t, t_index)
        return x0_hat.clamp(0.0, 1.0)
