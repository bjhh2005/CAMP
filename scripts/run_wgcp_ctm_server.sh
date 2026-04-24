#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/experiments:$ROOT_DIR:${PYTHONPATH:-}"
CONFIG_PATH="${CAMP_CTM_CONFIG:-$ROOT_DIR/configs/ctm_server_config.json}"
INPUT_DIR="${1:-$ROOT_DIR/data/imagenet_real}"
OUTPUT_DIR="${2:-$ROOT_DIR/outputs/wgcp_eval_ctm}"
shift $(( $# >= 2 ? 2 : $# ))

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
CTM_USE_FP16="${CTM_USE_FP16:-1}"
TORCH_CACHE="${TORCH_CACHE_DIR:-$ROOT_DIR/.cache/torch}"
GLOB_PATTERN="${GLOB_PATTERN:-*.JPEG}"
MAX_IMAGES="${MAX_IMAGES:-100}"
LIGHTWEIGHT_MODE="${LIGHTWEIGHT_MODE:-1}"
SAVE_REFERENCE_EVERY="${SAVE_REFERENCE_EVERY:-10}"
REFERENCE_DIR="${REFERENCE_DIR:-}"
PATCH_MODE="${PATCH_MODE:-0}"
PATCH_SIZE="${PATCH_SIZE:-64}"
PATCH_STRIDE="${PATCH_STRIDE:-32}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-64}"
PATCH_WEIGHT_SIGMA="${PATCH_WEIGHT_SIGMA:-0}"
PATCH_LOWFREQ_ALPHA="${PATCH_LOWFREQ_ALPHA:-0.1}"
PATCH_LL_SOURCE="${PATCH_LL_SOURCE:-hat}"
PATCH_PAD_MODE="${PATCH_PAD_MODE:-reflect}"
WAVELET="${WAVELET:-db4}"
REPLACEMENT_MODE="${REPLACEMENT_MODE:-adaptive_ms}"
MS_LEVELS="${MS_LEVELS:-3}"
MS_GAMMA_LEVELS="${MS_GAMMA_LEVELS:-1.6,1.2,0.9}"
MS_W_MIN="${MS_W_MIN:-0.05}"
MS_W_MAX="${MS_W_MAX:-0.95}"
MS_LL_ALPHA="${MS_LL_ALPHA:-0.08}"
MS_EPS="${MS_EPS:-1e-6}"
MS_LL_GATE_TAU="${MS_LL_GATE_TAU:-0.75}"
MS_LL_GATE_GAIN="${MS_LL_GATE_GAIN:-4.0}"
MS_HF_PRED_LEVELS="${MS_HF_PRED_LEVELS:-0.20,0.12,0.08}"
MS_HF_GATE_TAU="${MS_HF_GATE_TAU:-0.5}"
MS_HF_GATE_GAIN="${MS_HF_GATE_GAIN:-4.0}"

PREDICTOR_KWARGS="$(python -c "import json;print(json.dumps({'ctm_repo':r'$CTM_REPO','checkpoint':r'$CTM_CKPT','class_cond':bool(int(r'$CLASS_COND')),'class_label':int(r'$CLASS_LABEL'),'predictor_image_size':int(r'$PREDICTOR_IMAGE_SIZE'),'use_fp16':bool(int(r'$CTM_USE_FP16'))}))")"

echo "Running WGCP+CTM eval"
echo "  env: $CAMP_ENV"
echo "  input: $INPUT_DIR"
echo "  output: $OUTPUT_DIR"
echo "  glob: $GLOB_PATTERN"
echo "  max_images: $MAX_IMAGES"
echo "  lightweight_mode: $LIGHTWEIGHT_MODE"
echo "  save_reference_every: $SAVE_REFERENCE_EVERY"
echo "  reference_dir: ${REFERENCE_DIR:-<default>}"
echo "  patch_mode: $PATCH_MODE"
echo "  wavelet: $WAVELET"
echo "  replacement_mode: $REPLACEMENT_MODE"
echo "  ms_w_range: [$MS_W_MIN, $MS_W_MAX]"
echo "  ms_ll_gate: tau=$MS_LL_GATE_TAU gain=$MS_LL_GATE_GAIN"
echo "  ms_hf_pred_levels: $MS_HF_PRED_LEVELS"
echo "  ms_hf_gate: tau=$MS_HF_GATE_TAU gain=$MS_HF_GATE_GAIN"
echo "  ctm_use_fp16: $CTM_USE_FP16"
echo "  ctm repo: $CTM_REPO"
echo "  ckpt: $CTM_CKPT"

echo "Preflight import check (flash_attn/xformers shims)..."
conda run -n "$CAMP_ENV" python - <<'PY'
import importlib
mods = [
    "flash_attn.flash_attn_interface",
    "flash_attn.bert_padding",
    "xformers.ops",
    "xformers.components.attention",
]
for m in mods:
    importlib.import_module(m)
print("shim import check: OK")
PY

EVAL_ARGS=(
  --input_dir "$INPUT_DIR"
  --glob "$GLOB_PATTERN"
  --output_dir "$OUTPUT_DIR"
  --max_images "$MAX_IMAGES"
  --attack pgd
  --eps 0.0313725
  --pgd_steps 10
  --pgd_alpha 0.0078431
  --classifier resnet50
  --weights_cache_dir "$TORCH_CACHE"
  --predictor_type module
  --predictor_module "$PREDICTOR_MODULE"
  --predictor_kwargs_json "$PREDICTOR_KWARGS"
  --predictor_image_size "$PREDICTOR_IMAGE_SIZE"
  --wavelet "$WAVELET"
  --t_star 40
  --self_correct_k 0
  --replacement_mode "$REPLACEMENT_MODE"
  --ms_levels "$MS_LEVELS"
  --ms_gamma_levels "$MS_GAMMA_LEVELS"
  --ms_w_min "$MS_W_MIN"
  --ms_w_max "$MS_W_MAX"
  --ms_ll_alpha "$MS_LL_ALPHA"
  --ms_eps "$MS_EPS"
  --ms_ll_gate_tau "$MS_LL_GATE_TAU"
  --ms_ll_gate_gain "$MS_LL_GATE_GAIN"
  --ms_hf_pred_levels "$MS_HF_PRED_LEVELS"
  --ms_hf_gate_tau "$MS_HF_GATE_TAU"
  --ms_hf_gate_gain "$MS_HF_GATE_GAIN"
  --min_clean_conf 0.05
)

if [ "$LIGHTWEIGHT_MODE" = "1" ]; then
  EVAL_ARGS+=(--lightweight_mode --save_reference_every "$SAVE_REFERENCE_EVERY")
fi
if [ -n "$REFERENCE_DIR" ]; then
  EVAL_ARGS+=(--reference_dir "$REFERENCE_DIR")
fi
if [ "$PATCH_MODE" = "1" ]; then
  EVAL_ARGS+=(
    --patch_mode
    --patch_size "$PATCH_SIZE"
    --patch_stride "$PATCH_STRIDE"
    --patch_batch_size "$PATCH_BATCH_SIZE"
    --patch_weight_sigma "$PATCH_WEIGHT_SIGMA"
    --patch_lowfreq_alpha "$PATCH_LOWFREQ_ALPHA"
    --patch_ll_source "$PATCH_LL_SOURCE"
    --patch_pad_mode "$PATCH_PAD_MODE"
  )
fi

conda run -n "$CAMP_ENV" python "$ROOT_DIR/experiments/wgcp_attack_eval.py" \
  "${EVAL_ARGS[@]}" \
  "$@"
