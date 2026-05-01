#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-${RUN_DIR:-}}"
MAX_IMAGES="${MAX_IMAGES:-12}"

if [[ -z "$RUN_DIR" ]]; then
  echo "Usage: scripts/camp_make_report.sh <run_dir>"
  echo "Example: scripts/camp_make_report.sh outputs/camp/cifar10_baseline"
  exit 2
fi

python -m experiments.camp.report \
  --run_dir "$RUN_DIR" \
  --max_images "$MAX_IMAGES"

