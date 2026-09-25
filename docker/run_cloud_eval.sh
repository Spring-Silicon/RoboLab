#!/usr/bin/env bash
set -euo pipefail

: "${POLICY_HOST:?Set POLICY_HOST to the compiled-policy server or tunnel host}"
POLICY_VARIANT="${POLICY_VARIANT:-pi05_compiled_regular}"
POLICY_PORT="${POLICY_PORT:-8000}"
NUM_ENVS="${NUM_ENVS:-1}"
VIDEO_MODE="${VIDEO_MODE:-all}"
TASKS_TEXT="${TASKS:-BananaInBowlTask}"
read -r -a TASK_ARGS <<<"$TASKS_TEXT"

exec /workspace/isaaclab/_isaac_sim/python.sh \
  policies/pi0_family/run.py \
  --policy "$POLICY_VARIANT" \
  --remote-host "$POLICY_HOST" \
  --remote-port "$POLICY_PORT" \
  --num-envs "$NUM_ENVS" \
  --video-mode "$VIDEO_MODE" \
  --task "${TASK_ARGS[@]}" \
  --headless \
  "$@"
