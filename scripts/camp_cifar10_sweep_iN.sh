#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="${BASE_CONFIG:-experiments/camp/configs/cifar10_cm_baseline.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
DEVICE="${DEVICE:-cuda}"
ROOT_OUTPUT="${ROOT_OUTPUT:-outputs/camp/cifar10_sweep_iN}"
I_N_VALUES="${I_N_VALUES:-20 40 80}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"

mkdir -p "$ROOT_OUTPUT"

for iN in $I_N_VALUES; do
  LOCAL_CONFIG="$ROOT_OUTPUT/local_iN${iN}.yaml"
  RUN_DIR="$ROOT_OUTPUT/iN${iN}"
  python - "$BASE_CONFIG" "$LOCAL_CONFIG" "$iN" "$RUN_DIR" <<'PY'
from pathlib import Path
import sys
import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
iN = int(sys.argv[3])
run_dir = sys.argv[4]
data = yaml.safe_load(base_path.read_text(encoding="utf-8"))
data["experiment_name"] = f"cifar10_cm_iN{iN}"
data.setdefault("purification", {}).setdefault("schedule", {})["iN"] = iN
data.setdefault("evaluation", {})["output_dir"] = run_dir
out_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY

  args=(
    --config "$LOCAL_CONFIG"
    --max_samples "$MAX_SAMPLES"
    --output_dir "$RUN_DIR"
    --device "$DEVICE"
  )
  if [[ "$SAVE_IMAGES" == "1" ]]; then
    args+=(--save_images)
  fi

  python -m experiments.camp.run_purification "${args[@]}"
  python -m experiments.camp.report --run_dir "$RUN_DIR" --max_images 8
done

python - "$ROOT_OUTPUT" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
rows = []
for summary_path in sorted(root.glob("iN*/summary.json")):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    agg = summary.get("aggregate", {})
    rows.append((summary_path.parent.name, agg))

lines = [
    "# CIFAR-10 iN Sweep Overview",
    "",
    "| run | num_samples | attack_success_rate | recover_rate_on_attacked | purified_same_as_clean_rate |",
    "|---|---:|---:|---:|---:|",
]
for name, agg in rows:
    lines.append(
        f"| {name} | {agg.get('num_samples', '')} | "
        f"{agg.get('attack_success_rate', 0):.4f} | "
        f"{agg.get('recover_rate_on_attacked', 0):.4f} | "
        f"{agg.get('purified_same_as_clean_rate', 0):.4f} |"
    )
(root / "sweep_overview.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Sweep overview saved to: {root / 'sweep_overview.md'}")
PY

