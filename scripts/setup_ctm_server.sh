#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMP_ENV="${CAMP_ENV:-camp}"
CTM_REPO_URL="${CTM_REPO_URL:-https://github.com/sony/ctm.git}"
CTM_REPO_DIR="${CTM_REPO_DIR:-$ROOT_DIR/third_party/ctm}"
CTM_CACHE_DIR="${CTM_CACHE_DIR:-$ROOT_DIR/.cache/ctm}"
CTM_IMAGENET64_CKPT_ID="${CTM_IMAGENET64_CKPT_ID:-17XHwI5-IDpATRnBsxjOi6YCg1oD3MGC6}"
CTM_CHECKPOINTS_FOLDER_ID="${CTM_CHECKPOINTS_FOLDER_ID:-1KPF3tWLRad3n18XJ1TD7J04XtoMIQ8QV}"
CTM_CHECKPOINT_PATH="${CTM_CHECKPOINT_PATH:-}"
DOWNLOAD_CKPT="${DOWNLOAD_CKPT:-0}"
DOWNLOAD_FOLDER="${DOWNLOAD_FOLDER:-0}"
INSTALL_CTM_REQS="${INSTALL_CTM_REQS:-1}"

mkdir -p "$CTM_CACHE_DIR"
mkdir -p "$(dirname "$CTM_REPO_DIR")"

echo "[1/6] Update CAMP conda env: $CAMP_ENV"
conda env update -n "$CAMP_ENV" -f "$ROOT_DIR/environment.yml" --prune

NEED_GDOWN=0
if [ "$DOWNLOAD_CKPT" = "1" ] || [ "$DOWNLOAD_FOLDER" = "1" ]; then
  NEED_GDOWN=1
fi

echo "[2/6] Prepare download helper (optional)"
if [ "$NEED_GDOWN" = "1" ]; then
  echo "Installing gdown in $CAMP_ENV ..."
  conda run -n "$CAMP_ENV" python -m pip install --upgrade gdown
else
  echo "Skip gdown install (manual checkpoint mode)."
fi

echo "[3/6] Clone or update CTM repo"
if [ -d "$CTM_REPO_DIR/.git" ]; then
  git -C "$CTM_REPO_DIR" fetch --all --tags
  git -C "$CTM_REPO_DIR" pull --ff-only
else
  git clone "$CTM_REPO_URL" "$CTM_REPO_DIR"
fi

echo "[4/6] Resolve CTM checkpoint"
CKPT_PATH=""
if [ -n "$CTM_CHECKPOINT_PATH" ]; then
  if [ ! -f "$CTM_CHECKPOINT_PATH" ]; then
    echo "ERROR: CTM_CHECKPOINT_PATH not found: $CTM_CHECKPOINT_PATH"
    exit 1
  fi
  CKPT_PATH="$CTM_CHECKPOINT_PATH"
  echo "Using manual checkpoint: $CKPT_PATH"
elif [ -f "$CTM_CACHE_DIR/ctm_imagenet64_ema999.pt" ]; then
  CKPT_PATH="$CTM_CACHE_DIR/ctm_imagenet64_ema999.pt"
  echo "Using existing checkpoint: $CKPT_PATH"
elif [ -f "$CTM_CACHE_DIR/ema_0.999_049000.pt" ]; then
  CKPT_PATH="$CTM_CACHE_DIR/ema_0.999_049000.pt"
  echo "Using existing checkpoint: $CKPT_PATH"
elif [ "$DOWNLOAD_CKPT" = "1" ]; then
  CKPT_PATH="$CTM_CACHE_DIR/ctm_imagenet64_ema999.pt"
  FILE_URL="https://drive.google.com/uc?id=$CTM_IMAGENET64_CKPT_ID"
  if ! conda run -n "$CAMP_ENV" python -m gdown "$FILE_URL" -O "$CKPT_PATH"; then
    echo "gdown URL mode failed, retry with raw ID mode..."
    conda run -n "$CAMP_ENV" python -m gdown "$CTM_IMAGENET64_CKPT_ID" -O "$CKPT_PATH"
  fi
else
  echo "ERROR: checkpoint not found and DOWNLOAD_CKPT=0."
  echo "Please do one of the following:"
  echo "  1) Set CTM_CHECKPOINT_PATH=/abs/path/ema_0.999_049000.pt"
  echo "  2) Put checkpoint under $CTM_CACHE_DIR as:"
  echo "     - ema_0.999_049000.pt  OR  ctm_imagenet64_ema999.pt"
  echo "  3) Enable auto-download: DOWNLOAD_CKPT=1"
  exit 1
fi

if [ "$DOWNLOAD_FOLDER" = "1" ]; then
  echo "[optional] Download author shared checkpoint folder"
  conda run -n "$CAMP_ENV" python -m gdown --folder "https://drive.google.com/drive/folders/$CTM_CHECKPOINTS_FOLDER_ID" -O "$CTM_CACHE_DIR/author_folder"
fi

echo "[5/6] Optionally install CTM repo requirements"
if [ "$INSTALL_CTM_REQS" = "1" ]; then
  if [ -f "$CTM_REPO_DIR/requirements.txt" ]; then
    conda run -n "$CAMP_ENV" python -m pip install -r "$CTM_REPO_DIR/requirements.txt"
  else
    echo "No requirements.txt found in $CTM_REPO_DIR, skipped"
  fi
fi

echo "[6/6] Write local CTM config"
mkdir -p "$ROOT_DIR/configs"
cat > "$ROOT_DIR/configs/ctm_server_config.json" <<EOF
{
  "camp_env": "$CAMP_ENV",
  "ctm_repo": "$CTM_REPO_DIR",
  "checkpoint": "$CKPT_PATH",
  "predictor_image_size": 64,
  "predictor_module": "experiments.ctm_adapter_sony:CTMRepoPredictor"
}
EOF

echo "Done."
echo "CTM repo: $CTM_REPO_DIR"
echo "CTM ckpt: $CKPT_PATH"
echo "Config: $ROOT_DIR/configs/ctm_server_config.json"
