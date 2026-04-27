#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/experiments:$ROOT_DIR:${PYTHONPATH:-}"
CONFIG_PATH="${CAMP_CM_CONFIG:-$ROOT_DIR/configs/cm_server_config.json}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Missing config: $CONFIG_PATH"
  echo "Run: bash scripts/setup_cm_server.sh"
  exit 1
fi

CAMP_ENV="$(python -c "import json;print(json.load(open(r'$CONFIG_PATH','r',encoding='utf-8'))['camp_env'])")"
CM_REPO="$(python -c "import json;print(json.load(open(r'$CONFIG_PATH','r',encoding='utf-8'))['ctm_repo'])")"
CM_CKPT="$(python -c "import json;print(json.load(open(r'$CONFIG_PATH','r',encoding='utf-8'))['checkpoint'])")"
PREDICTOR_MODULE="$(python -c "import json;print(json.load(open(r'$CONFIG_PATH','r',encoding='utf-8'))['predictor_module'])")"
PREDICTOR_IMAGE_SIZE="$(python -c "import json;print(json.load(open(r'$CONFIG_PATH','r',encoding='utf-8')).get('predictor_image_size',64))")"
TRAINING_MODE="$(python -c "import json;print(json.load(open(r'$CONFIG_PATH','r',encoding='utf-8')).get('training_mode','consistency_distillation'))")"
CTM_INFERENCE="$(python -c "import json;print(int(bool(json.load(open(r'$CONFIG_PATH','r',encoding='utf-8')).get('ctm_inference', False))))")"
OUTPUT_HEAD="$(python -c "import json;print(json.load(open(r'$CONFIG_PATH','r',encoding='utf-8')).get('output_head','g_theta'))")"

CLASS_COND="${CM_CLASS_COND:-1}"
CLASS_LABEL="${CM_CLASS_LABEL:-0}"
CM_USE_FP16="${CM_USE_FP16:-1}"
TEST_SIZE="${CM_TEST_SIZE:-64}"
TEST_T_INDEX="${CM_TEST_T_INDEX:-40}"

PREDICTOR_KWARGS="$(python -c "import json;print(json.dumps({'ctm_repo':r'$CM_REPO','checkpoint':r'$CM_CKPT','class_cond':bool(int(r'$CLASS_COND')),'class_label':int(r'$CLASS_LABEL'),'predictor_image_size':int(r'$PREDICTOR_IMAGE_SIZE'),'use_fp16':bool(int(r'$CM_USE_FP16')),'training_mode':r'$TRAINING_MODE','ctm_inference':bool(int(r'$CTM_INFERENCE')),'output_head':r'$OUTPUT_HEAD'}))")"

echo "Testing CM predictor"
echo "  env: $CAMP_ENV"
echo "  repo: $CM_REPO"
echo "  ckpt: $CM_CKPT"
echo "  predictor_module: $PREDICTOR_MODULE"
echo "  training_mode: $TRAINING_MODE"
echo "  ctm_inference: $CTM_INFERENCE"
echo "  output_head: $OUTPUT_HEAD"

conda run -n "$CAMP_ENV" python - <<PY
import argparse
import json
import torch

from experiments.wgcp_predictor import build_predictor

args = argparse.Namespace(
    predictor_type="module",
    predictor_module=r"$PREDICTOR_MODULE",
    predictor_kwargs_json=r'''$PREDICTOR_KWARGS''',
    kernel_size=5,
    sigma=1.0,
    mix=0.45,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = build_predictor(args, device=device)
x = torch.rand(1, 3, int(r"$TEST_SIZE"), int(r"$TEST_SIZE"), device=device)
with torch.no_grad():
    y = predictor(x, int(r"$TEST_T_INDEX"))
print("device =", device)
print("input_shape =", tuple(x.shape))
print("output_shape =", tuple(y.shape))
print("output_min =", float(y.min().item()))
print("output_max =", float(y.max().item()))
assert y.shape == x.shape, "shape mismatch"
assert torch.isfinite(y).all(), "non-finite output"
assert float(y.min().item()) >= -1e-5 and float(y.max().item()) <= 1.00001, "output not clamped to [0,1]"
print("CM smoke test: OK")
PY
