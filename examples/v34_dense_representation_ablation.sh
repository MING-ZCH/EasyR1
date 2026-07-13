#!/bin/bash
# V34 Phase2: Dense representation ablation — DESIGN STUB (NOT runnable as-is).
#
# Batch-K pointing and grounding-token modes require changes to the interleaved
# rollout engine (verl/workers/rollout/vllm_rollout_spmd.py) and the red-dot
# annotator — these knobs are NOT wired today. Running this as a training job would
# silently fall back to single-point behaviour and produce a misleading "ablation".
#
# This script therefore only PRINTS the intended config and exits. Implement the
# rollout changes first, then convert to a real launcher.
#
# Modes: single (==v34_dense_curriculum_11_20) | batch3 | batch5 | grounding_token
set -euo pipefail

V34_POINT_MODE=${V34_POINT_MODE:-batch3}

cat <<EOF
[V34] Representation ablation DESIGN STUB — mode=${V34_POINT_MODE}
  single          : max_turns=21, 1 point/turn  -> already runnable via v34_dense_curriculum_11_20.sh
  batch3          : max_turns~12, 3 points/turn  -> REQUIRES rollout multi-point parsing + annotator
  batch5          : max_turns~8,  5 points/turn  -> REQUIRES rollout multi-point parsing + annotator
  grounding_token : single grounding token/turn  -> REQUIRES tokenizer + rollout changes

Wiring TODO before this can run:
  1. rollout: parse >1 <point> per assistant turn; render all before next obs image
  2. reward : _parse_pred_points already handles multiple points/turn (verify ordering)
  3. annotator: batch-render K dots with correct accumulated indices
  4. add INTERLEAVED_POINTS_PER_TURN / INTERLEAVED_BATCH_POINTING reads in rollout config
EOF

# Hard stop: do not pretend to launch an unimplemented ablation.
echo "[V34] Not launching (design stub). See header." >&2
exit 2
