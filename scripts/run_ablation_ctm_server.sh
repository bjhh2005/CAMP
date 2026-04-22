#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/experiments:$ROOT_DIR:${PYTHONPATH:-}"

CONFIG_PATH="${CAMP_CTM_CONFIG:-$ROOT_DIR/configs/ctm_server_config.json}"
INPUT_DIR="${1:-$ROOT_DIR/data/clean_samples}"
OUTPUT_ROOT="${2:-$ROOT_DIR/outputs/wgcp_ablation_ctm}"
shift $(( $# >= 2 ? 2 : $# ))
EXTRA_ARGS=("$@")

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Missing config: $CONFIG_PATH"
  echo "Run: bash scripts/setup_ctm_server.sh"
  exit 1
fi

CAMP_ENV="$(python -c "import json;print(json.load(open(r'$CONFIG_PATH','r',encoding='utf-8'))['camp_env'])")"
CTM_REPO="$(python -c "import json;print(json.load(open(r'$CONFIG_PATH','r',encoding='utf-8'))['ctm_repo'])")"
CTM_CKPT="$(python -c "import json;print(json.load(open(r'$CONFIG_PATH','r',encoding='utf-8'))['checkpoint'])")"
PREDICTOR_MODULE="$(python -c "import json;print(json.load(open(r'$CONFIG_PATH','r',encoding='utf-8'))['predictor_module'])")"
PREDICTOR_IMAGE_SIZE="$(python -c "import json;print(json.load(open(r'$CONFIG_PATH','r',encoding='utf-8')).get('predictor_image_size',64))")"

CLASS_COND="${CTM_CLASS_COND:-1}"
CLASS_LABEL="${CTM_CLASS_LABEL:-0}"
TORCH_CACHE="${TORCH_CACHE_DIR:-$ROOT_DIR/.cache/torch}"
MAX_IMAGES="${MAX_IMAGES:-8}"
MIN_CLEAN_CONF="${MIN_CLEAN_CONF:-0.05}"
HF_PRESERVE="${HF_PRESERVE:-0.35}"
HF_SHRINK="${HF_SHRINK:-0.6}"

PREDICTOR_KWARGS="$(python -c "import json;print(json.dumps({'ctm_repo':r'$CTM_REPO','checkpoint':r'$CTM_CKPT','class_cond':bool(int(r'$CLASS_COND')),'class_label':int(r'$CLASS_LABEL'),'predictor_image_size':int(r'$PREDICTOR_IMAGE_SIZE')}))")"

echo "Running WGCP+CTM ablation suite"
echo "  env: $CAMP_ENV"
echo "  input: $INPUT_DIR"
echo "  output root: $OUTPUT_ROOT"
echo "  ctm repo: $CTM_REPO"
echo "  ckpt: $CTM_CKPT"

mkdir -p "$OUTPUT_ROOT"

run_case() {
  local case_name="$1"
  shift
  local case_out="$OUTPUT_ROOT/$case_name"
  echo ""
  echo "==> [$case_name] output: $case_out"
  conda run -n "$CAMP_ENV" python "$ROOT_DIR/experiments/wgcp_attack_eval.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$case_out" \
    --attack pgd \
    --eps 0.0313725 \
    --pgd_steps 10 \
    --pgd_alpha 0.0078431 \
    --classifier resnet50 \
    --weights_cache_dir "$TORCH_CACHE" \
    --predictor_type module \
    --predictor_module "$PREDICTOR_MODULE" \
    --predictor_kwargs_json "$PREDICTOR_KWARGS" \
    --predictor_image_size "$PREDICTOR_IMAGE_SIZE" \
    --max_images "$MAX_IMAGES" \
    --min_clean_conf "$MIN_CLEAN_CONF" \
    --archive_tag "ablation_${case_name}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
}

# A0: baseline (paper-default implementation)
run_case "A0_baseline" \
  --t_star 40 \
  --t_bridge 10 \
  --self_correct_k 1 \
  --replacement_mode hard \
  --ablation_ll_source orig \
  --ablation_hard_hf_source pred

# A1: LL anchor from predictor output (test semantic drift sensitivity)
run_case "A1_ll_hat" \
  --t_star 40 \
  --t_bridge 10 \
  --self_correct_k 1 \
  --replacement_mode hard \
  --ablation_ll_source hat \
  --ablation_hard_hf_source pred

# A2: hard mode keeps original HF (test detail retention vs detox)
run_case "A2_hard_hf_orig" \
  --t_star 40 \
  --t_bridge 10 \
  --self_correct_k 1 \
  --replacement_mode hard \
  --ablation_ll_source orig \
  --ablation_hard_hf_source orig

# A3: both swapped (stress test)
run_case "A3_ll_hat_hf_orig" \
  --t_star 40 \
  --t_bridge 10 \
  --self_correct_k 1 \
  --replacement_mode hard \
  --ablation_ll_source hat \
  --ablation_hard_hf_source orig

# A4: fused HF replacement (soft blend)
run_case "A4_fused" \
  --t_star 40 \
  --t_bridge 10 \
  --self_correct_k 1 \
  --replacement_mode fused \
  --hf_preserve "$HF_PRESERVE" \
  --hf_shrink "$HF_SHRINK" \
  --ablation_ll_source orig \
  --ablation_hard_hf_source pred

# A5: no self-correct loop (test t*->t1 bridge contribution)
run_case "A5_no_loop" \
  --t_star 40 \
  --self_correct_k 0 \
  --replacement_mode hard \
  --ablation_ll_source orig \
  --ablation_hard_hf_source pred

conda run -n "$CAMP_ENV" python - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for summary_path in sorted(root.glob("*/summary.json")):
    case_name = summary_path.parent.name
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    agg = data.get("aggregate", {})
    run_meta = data.get("run_meta", {})
    rows.append(
        {
            "case": case_name,
            "num_valid_samples": agg.get("num_valid_samples"),
            "attack_success_rate": agg.get("attack_success_rate"),
            "recover_rate_on_attacked": agg.get("recover_rate_on_attacked"),
            "clean_pred_consistency_rate": agg.get("clean_pred_consistency_rate"),
            "timestamp_local": run_meta.get("timestamp_local"),
            "output_dir": str(summary_path.parent),
        }
    )

out_path = root / "ablation_overview.json"
out_path.write_text(json.dumps({"cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Ablation overview: {out_path}")
for r in rows:
    print(
        f"[{r['case']}] recover_rate_on_attacked={r['recover_rate_on_attacked']} "
        f"clean_pred_consistency_rate={r['clean_pred_consistency_rate']}"
    )
PY

echo ""
echo "Ablation suite finished. Root: $OUTPUT_ROOT"
