#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/experiments:$ROOT_DIR:${PYTHONPATH:-}"

CONFIG_PATH="${CAMP_CTM_CONFIG:-$ROOT_DIR/configs/ctm_server_config.json}"
INPUT_DIR="${1:-$ROOT_DIR/data/imagenet_real}"
OUTPUT_DIR="${2:-$ROOT_DIR/outputs/ctm_feature_probe}"
shift $(( $# >= 2 ? 2 : $# ))

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Missing config: $CONFIG_PATH"
  echo "Run: bash scripts/setup_ctm_server.sh"
  exit 1
fi

CAMP_ENV="$(python -c "import json;print(json.load(open(r'$CONFIG_PATH','r',encoding='utf-8'))['camp_env'])")"

TORCH_CACHE="${TORCH_CACHE_DIR:-$ROOT_DIR/.cache/torch}"
GLOB_PATTERN="${GLOB_PATTERN:-*.JPEG}"
MAX_IMAGES="${MAX_IMAGES:-10}"
ATTACK="${ATTACK:-pgd}"
EPS="${EPS:-0.0313725}"
PGD_STEPS="${PGD_STEPS:-10}"
PGD_ALPHA="${PGD_ALPHA:-0.0078431}"
CLASSIFIER="${CLASSIFIER:-resnet50}"
MIN_CLEAN_CONF="${MIN_CLEAN_CONF:-0.05}"
ONLY_ATTACK_SUCCESS="${ONLY_ATTACK_SUCCESS:-1}"

T_STAR="${T_STAR:-40}"
SELF_CORRECT_K="${SELF_CORRECT_K:-0}"
WAVELET="${WAVELET:-db4}"
REPLACEMENT_MODE="${REPLACEMENT_MODE:-adaptive_ms}"
MS_LEVELS="${MS_LEVELS:-3}"
MS_GAMMA_LEVELS="${MS_GAMMA_LEVELS:-1.6,1.2,0.9}"
MS_W_MIN="${MS_W_MIN:-0.05}"
MS_W_MAX="${MS_W_MAX:-0.95}"
MS_LL_ALPHA="${MS_LL_ALPHA:-0.08}"

CTM_FEATURE_LAYER="${CTM_FEATURE_LAYER:-}"
CTM_FEATURE_SOURCES="${CTM_FEATURE_SOURCES:-adv,xt,hat}"
CTM_FEATURE_REDUCE="${CTM_FEATURE_REDUCE:-mean_abs}"
CTM_FEATURE_T_INDEX="${CTM_FEATURE_T_INDEX:--1}"
CTM_FEATURE_OVERLAY_ALPHA="${CTM_FEATURE_OVERLAY_ALPHA:-0.45}"
CTM_FEATURE_LEAF_ONLY="${CTM_FEATURE_LEAF_ONLY:-1}"
LIST_FEATURE_MODULES="${LIST_FEATURE_MODULES:-0}"

echo "Running CTM feature probe (E2)"
echo "  env: $CAMP_ENV"
echo "  input: $INPUT_DIR"
echo "  output: $OUTPUT_DIR"
echo "  glob: $GLOB_PATTERN"
echo "  max_images: $MAX_IMAGES"
echo "  attack: $ATTACK"
echo "  replacement_mode: $REPLACEMENT_MODE"
echo "  t_star: $T_STAR"
echo "  feature_sources: $CTM_FEATURE_SOURCES"
echo "  feature_reduce: $CTM_FEATURE_REDUCE"
echo "  feature_layer: ${CTM_FEATURE_LAYER:-<auto>}"

ARGS=(
  --input_dir "$INPUT_DIR"
  --output_dir "$OUTPUT_DIR"
  --glob "$GLOB_PATTERN"
  --max_images "$MAX_IMAGES"
  --attack "$ATTACK"
  --eps "$EPS"
  --pgd_steps "$PGD_STEPS"
  --pgd_alpha "$PGD_ALPHA"
  --classifier "$CLASSIFIER"
  --weights_cache_dir "$TORCH_CACHE"
  --min_clean_conf "$MIN_CLEAN_CONF"
  --predictor_type module
  --predictor_config_path "$CONFIG_PATH"
  --wavelet "$WAVELET"
  --t_star "$T_STAR"
  --self_correct_k "$SELF_CORRECT_K"
  --replacement_mode "$REPLACEMENT_MODE"
  --ms_levels "$MS_LEVELS"
  --ms_gamma_levels "$MS_GAMMA_LEVELS"
  --ms_w_min "$MS_W_MIN"
  --ms_w_max "$MS_W_MAX"
  --ms_ll_alpha "$MS_LL_ALPHA"
  --ctm_feature_sources "$CTM_FEATURE_SOURCES"
  --ctm_feature_reduce "$CTM_FEATURE_REDUCE"
  --ctm_feature_t_index "$CTM_FEATURE_T_INDEX"
  --ctm_feature_overlay_alpha "$CTM_FEATURE_OVERLAY_ALPHA"
)

if [ -n "$CTM_FEATURE_LAYER" ]; then
  ARGS+=(--ctm_feature_layer "$CTM_FEATURE_LAYER")
fi
if [ "$ONLY_ATTACK_SUCCESS" = "1" ]; then
  ARGS+=(--only_attack_success)
fi
if [ "$CTM_FEATURE_LEAF_ONLY" = "1" ]; then
  ARGS+=(--ctm_feature_leaf_only)
else
  ARGS+=(--no-ctm_feature_leaf_only)
fi
if [ "$LIST_FEATURE_MODULES" = "1" ]; then
  ARGS+=(--list_feature_modules)
fi

conda run -n "$CAMP_ENV" python "$ROOT_DIR/experiments/ctm_feature_probe.py" \
  "${ARGS[@]}" \
  "$@"
