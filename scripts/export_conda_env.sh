#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-camp}"
OUT_FILE="${2:-environment.yml}"

conda export -n "$ENV_NAME" --from-history --file "$OUT_FILE"
echo "Exported $ENV_NAME to $OUT_FILE"
