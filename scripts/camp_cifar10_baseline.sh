#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-experiments/camp/configs/cifar10_cm_openai_jax.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/camp/cifar10_baseline}"
DEVICE="${DEVICE:-cuda}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"

args=(
  --config "$CONFIG"
  --max_samples "$MAX_SAMPLES"
  --output_dir "$OUTPUT_DIR"
  --device "$DEVICE"
)

if [[ "$SAVE_IMAGES" == "1" ]]; then
  args+=(--save_images)
fi

python -m experiments.camp.run_purification "${args[@]}"
python -m experiments.camp.report --run_dir "$OUTPUT_DIR" --max_images 12

echo "Baseline run complete:"
echo "  Summary: $OUTPUT_DIR/summary.json"
echo "  Report:  $OUTPUT_DIR/analysis.md"
