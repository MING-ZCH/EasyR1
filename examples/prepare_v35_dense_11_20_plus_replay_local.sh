#!/bin/bash
# Build the local v35 11-20 + 0-10 replay dataset expected by v35_dense_11_20.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/local_path_env.sh"

OUT_DIR="${STEPCOUNT_DENSE_11_20_PLUS_REPLAY_DATA}/data"
mkdir -p "$(dirname "${OUT_DIR}")"

python3 "${REPO_ROOT}/tools/v34_prepare_dense_subset.py" \
  --combined "${STEPCOUNT_DENSE_COMBINED_DATA}/data" \
  --replay "${STEPCOUNT_REPLAY_DATA}/data" \
  --count_min 11 \
  --count_max 20 \
  --replay_ratio "${V35_REPLAY_RATIO:-0.25}" \
  --out_dir "${OUT_DIR}" \
  "$@"

echo "[prepare-v35-local] wrote ${OUT_DIR}"
