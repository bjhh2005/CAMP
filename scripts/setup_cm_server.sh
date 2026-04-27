#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMP_ENV="${CAMP_ENV:-camp}"
CM_REPO_URL="${CM_REPO_URL:-https://github.com/sony/ctm.git}"
CM_REPO_DIR="${CM_REPO_DIR:-/home/HHY/CAMP/.cache/ctm}"
CM_CACHE_DIR="${CM_CACHE_DIR:-/home/HHY/CAMP/.cache/cm}"
CM_CHECKPOINT_PATH="${CM_CHECKPOINT_PATH:-}"
CM_TRAINING_MODE="${CM_TRAINING_MODE:-consistency_distillation}"
CM_CTM_INFERENCE="${CM_CTM_INFERENCE:-0}"
CM_OUTPUT_HEAD="${CM_OUTPUT_HEAD:-g_theta}"
INSTALL_CM_REQS="${INSTALL_CM_REQS:-1}"
INSTALL_CM_RUNTIME_REQS="${INSTALL_CM_RUNTIME_REQS:-1}"
CM_RUNTIME_PIP_PKGS="${CM_RUNTIME_PIP_PKGS:-blobfile einops mpi4py}"

mkdir -p "$CM_CACHE_DIR"
mkdir -p "$(dirname "$CM_REPO_DIR")"

echo "[1/6] Update CAMP conda env: $CAMP_ENV"
conda env update -n "$CAMP_ENV" -f "$ROOT_DIR/environment.yml" --prune

echo "[2/6] Prepare CM repo"
if [ -d "$CM_REPO_DIR/.git" ]; then
  git -C "$CM_REPO_DIR" fetch --all --tags || true
  git -C "$CM_REPO_DIR" pull --ff-only || true
elif [ -d "$CM_REPO_DIR" ]; then
  echo "Using existing non-git repo dir: $CM_REPO_DIR"
else
  git clone "$CM_REPO_URL" "$CM_REPO_DIR"
fi

echo "[3/6] Install CM runtime Python deps"
if [ "$INSTALL_CM_RUNTIME_REQS" = "1" ]; then
  conda run -n "$CAMP_ENV" python -m pip install $CM_RUNTIME_PIP_PKGS
else
  echo "Skip CM runtime deps install."
fi

echo "[4/6] Resolve CM checkpoint"
CKPT_PATH=""
if [ -n "$CM_CHECKPOINT_PATH" ]; then
  if [ ! -f "$CM_CHECKPOINT_PATH" ]; then
    echo "ERROR: CM_CHECKPOINT_PATH not found: $CM_CHECKPOINT_PATH"
    exit 1
  fi
  CKPT_PATH="$CM_CHECKPOINT_PATH"
  echo "Using manual checkpoint: $CKPT_PATH"
else
  CANDIDATES=(
    "$CM_CACHE_DIR/cm_imagenet64.pt"
    "$CM_CACHE_DIR/cm_imagenet64_ema.pt"
    "$CM_CACHE_DIR/consistency_imagenet64.pt"
    "$CM_CACHE_DIR/ema_0.999_049000.pt"
    "$CM_REPO_DIR/cm_imagenet64.pt"
    "$CM_REPO_DIR/cm_imagenet64_ema.pt"
    "$CM_REPO_DIR/ema_0.999_049000.pt"
  )
  for cand in "${CANDIDATES[@]}"; do
    if [ -f "$cand" ]; then
      CKPT_PATH="$cand"
      break
    fi
  done
  if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: could not auto-find a CM checkpoint."
    echo "Please set CM_CHECKPOINT_PATH=/abs/path/to/your_cm_checkpoint.pt"
    echo "Searched:"
    printf '  %s\n' "${CANDIDATES[@]}"
    exit 1
  fi
  echo "Using discovered checkpoint: $CKPT_PATH"
fi

echo "[5/6] Optionally install repo requirements"
if [ "$INSTALL_CM_REQS" = "1" ]; then
  if [ -f "$CM_REPO_DIR/requirements.txt" ]; then
    conda run -n "$CAMP_ENV" python -m pip install -r "$CM_REPO_DIR/requirements.txt"
  else
    echo "No requirements.txt found in $CM_REPO_DIR, skipped"
  fi
else
  echo "Skip CM repo requirements install."
fi

echo "[6/6] Write local CM config"
mkdir -p "$ROOT_DIR/configs"
cat > "$ROOT_DIR/configs/cm_server_config.json" <<EOF
{
  "camp_env": "$CAMP_ENV",
  "ctm_repo": "$CM_REPO_DIR",
  "checkpoint": "$CKPT_PATH",
  "predictor_image_size": 64,
  "predictor_module": "experiments.cm_adapter_sony:CMRepoPredictor",
  "training_mode": "$CM_TRAINING_MODE",
  "ctm_inference": $( [ "$CM_CTM_INFERENCE" = "1" ] && echo "true" || echo "false" ),
  "output_head": "$CM_OUTPUT_HEAD"
}
EOF

echo "Done."
echo "CM repo: $CM_REPO_DIR"
echo "CM ckpt: $CKPT_PATH"
echo "Config: $ROOT_DIR/configs/cm_server_config.json"
