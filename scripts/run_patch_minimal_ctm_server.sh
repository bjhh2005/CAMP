#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/experiments:$ROOT_DIR:${PYTHONPATH:-}"

CONFIG_PATH="${CAMP_CTM_CONFIG:-$ROOT_DIR/configs/ctm_server_config.json}"
INPUT_DIR="${1:-$ROOT_DIR/data/imagenet_real}"
OUTPUT_ROOT="${2:-$ROOT_DIR/outputs/wgcp_patch_min}"
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
MAX_IMAGES="${MAX_IMAGES:-100}"
MIN_CLEAN_CONF="${MIN_CLEAN_CONF:-0.05}"
SAVE_DETAIL_EVERY="${SAVE_DETAIL_EVERY:-10}"
GLOB_PATTERN="${GLOB_PATTERN:-*.JPEG}"

# Patch-WGCP defaults (main case uses these unless overridden)
PATCH_SIZE="${PATCH_SIZE:-64}"
PATCH_STRIDE="${PATCH_STRIDE:-32}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-64}"
PATCH_WEIGHT_SIGMA="${PATCH_WEIGHT_SIGMA:-0}"
PATCH_LOWFREQ_ALPHA_MAIN="${PATCH_LOWFREQ_ALPHA_MAIN:-0.1}"
PATCH_LL_SOURCE="${PATCH_LL_SOURCE:-hat}"
PATCH_PAD_MODE="${PATCH_PAD_MODE:-reflect}"

PREDICTOR_KWARGS="$(python -c "import json;print(json.dumps({'ctm_repo':r'$CTM_REPO','checkpoint':r'$CTM_CKPT','class_cond':bool(int(r'$CLASS_COND')),'class_label':int(r'$CLASS_LABEL'),'predictor_image_size':int(r'$PREDICTOR_IMAGE_SIZE')}))")"

echo "Running Patch-WGCP minimal suite (3 runs)"
echo "  env: $CAMP_ENV"
echo "  input: $INPUT_DIR"
echo "  output root: $OUTPUT_ROOT"
echo "  glob: $GLOB_PATTERN"
echo "  max_images: $MAX_IMAGES"
echo "  save_detail_every: $SAVE_DETAIL_EVERY"
echo "  predictor_module: $PREDICTOR_MODULE"
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
    --max_images "$MAX_IMAGES" \
    --glob "$GLOB_PATTERN" \
    --save_detail_every "$SAVE_DETAIL_EVERY" \
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
    --t_star 40 \
    --self_correct_k 0 \
    --replacement_mode hard \
    --ablation_ll_source orig \
    --ablation_hard_hf_source pred \
    --min_clean_conf "$MIN_CLEAN_CONF" \
    --archive_tag "patch_min_${case_name}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
}

# 1) Global A5 baseline
run_case "A5_global" \
  --device cuda

# 2) Patch-A5 main
run_case "A5_patch_main" \
  --device cuda \
  --patch_mode \
  --patch_size "$PATCH_SIZE" \
  --patch_stride "$PATCH_STRIDE" \
  --patch_batch_size "$PATCH_BATCH_SIZE" \
  --patch_weight_sigma "$PATCH_WEIGHT_SIGMA" \
  --patch_lowfreq_alpha "$PATCH_LOWFREQ_ALPHA_MAIN" \
  --patch_ll_source "$PATCH_LL_SOURCE" \
  --patch_pad_mode "$PATCH_PAD_MODE"

# 3) Patch-A5 alpha=0 contrast
run_case "A5_patch_alpha0" \
  --device cuda \
  --patch_mode \
  --patch_size "$PATCH_SIZE" \
  --patch_stride "$PATCH_STRIDE" \
  --patch_batch_size "$PATCH_BATCH_SIZE" \
  --patch_weight_sigma "$PATCH_WEIGHT_SIGMA" \
  --patch_lowfreq_alpha 0.0 \
  --patch_ll_source "$PATCH_LL_SOURCE" \
  --patch_pad_mode "$PATCH_PAD_MODE"

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
    rows.append(
        {
            "case": case_name,
            "num_valid_samples": agg.get("num_valid_samples"),
            "attack_success_rate": agg.get("attack_success_rate"),
            "recover_rate_on_attacked": agg.get("recover_rate_on_attacked"),
            "clean_pred_consistency_rate": agg.get("clean_pred_consistency_rate"),
            "saved_detail_samples": agg.get("saved_detail_samples"),
            "save_detail_every": agg.get("save_detail_every"),
            "output_dir": str(summary_path.parent),
        }
    )

out_path = root / "patch_min_overview.json"
out_path.write_text(json.dumps({"cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Patch minimal overview: {out_path}")
for r in rows:
    print(
        f"[{r['case']}] recover_rate_on_attacked={r['recover_rate_on_attacked']} "
        f"clean_pred_consistency_rate={r['clean_pred_consistency_rate']} "
        f"saved_detail_samples={r['saved_detail_samples']}"
    )
PY

echo ""
echo "Patch minimal suite finished. Root: $OUTPUT_ROOT"
echo "Overview: $OUTPUT_ROOT/patch_min_overview.json"
