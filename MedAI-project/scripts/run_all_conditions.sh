#!/usr/bin/env bash
set -euo pipefail

SLAKE_ROOT="${1:-/workspace/datasets/SLAKE_raw/Slake1.0}"
SPLIT="${2:-test}"
MAX_SAMPLES="${3:-20}"
BASE_OUT="${4:-outputs}"

CONDITIONS=(
  original
  black
  lpf
  hpf
  patch_shuffle
)

echo "SLAKE_ROOT: ${SLAKE_ROOT}"
echo "SPLIT: ${SPLIT}"
echo "MAX_SAMPLES: ${MAX_SAMPLES}"
echo "BASE_OUT: ${BASE_OUT}"

for CONDITION in "${CONDITIONS[@]}"; do
  echo "========================================"
  echo "Running condition: ${CONDITION}"
  echo "========================================"

  python -m src.run_eval \
    --use_hf \
    --slake_root "${SLAKE_ROOT}" \
    --split "${SPLIT}" \
    --condition "${CONDITION}" \
    --output_dir "${BASE_OUT}/${CONDITION}" \
    --max_samples "${MAX_SAMPLES}"
done

echo "All conditions finished."