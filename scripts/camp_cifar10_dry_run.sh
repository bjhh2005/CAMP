#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-experiments/camp/configs/cifar10_cm_openai_jax.yaml}"

python -m experiments.camp.run_purification \
  --config "$CONFIG" \
  --dry_run

python -m experiments.camp.check_paths \
  --config "$CONFIG"
