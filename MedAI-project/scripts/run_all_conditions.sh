set -e

SLAKE_ROOT=${1:-/path/to/SLAKE}
BASE_OUT=${2:-outputs}

for CONDITION in original lpf hpf black patch_shuffle; do
  python -m src.run_eval \
    --slake_root "$SLAKE_ROOT" \
    --split test \
    --condition "$CONDITION" \
    --output_dir "$BASE_OUT/$CONDITION"
done