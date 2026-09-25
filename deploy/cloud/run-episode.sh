#!/usr/bin/env bash
set -euo pipefail

HOST="${1:?usage: run-episode.sh <gpu-host> <model> <TaskName> [output-name] [ssh-user]}"
MODEL="${2:?missing model}"
TASK="${3:?missing task name}"
OUTPUT="${4:-$(date -u +%Y%m%dT%H%M%SZ)_${MODEL}_${TASK}}"
USER_NAME="${5:-ubuntu}"
IMAGE="${ROBOLAB_IMAGE:-public.ecr.aws/m4l3e1i0/spring-silicon/robolab:2026-09-25}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY="$HOME/.ssh/robolab-cloud-tunnel"
DASHBOARD_IP="${ROBOLAB_DASHBOARD_IP:-$(ip -4 route get 1.1.1.1 | sed -n 's/.* src \([^ ]*\).*/\1/p')}"
[[ "$HOST" =~ ^[A-Za-z0-9.:-]+$ && "$TASK" =~ ^[A-Za-z0-9_]+$ && "$OUTPUT" =~ ^[A-Za-z0-9_.-]+$ ]] \
  || { echo "host, task, or output contains unsupported characters" >&2; exit 2; }

case "$MODEL" in
  pi05_spring_regular) VARIANT=pi05_compiled_regular ;;
  pi05_spring_optimized) VARIANT=pi05_compiled_optimized ;;
  pi0|pi0_fast|pi05|paligemma|paligemma_fast)
    echo "$MODEL is not available on the B580 yet; refusing to run policy inference on the simulator GPU." >&2
    exit 3
    ;;
  *) echo "model: pi05_spring_regular or pi05_spring_optimized" >&2; exit 2 ;;
esac

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/$OUTPUT.log"
exec > >(tee -a "$LOG") 2>&1
set -x
"$ROOT/setup-ec2.sh" "$HOST" "$USER_NAME"

if systemctl --user cat spring-openpi-compiled@.service >/dev/null 2>&1; then
  POLICYCTL=(systemctl --user)
else
  POLICYCTL=(sudo systemctl)
fi
restore_regular() {
  "${POLICYCTL[@]}" stop spring-openpi-compiled@pi05_compiled_optimized.service || true
  "${POLICYCTL[@]}" start spring-openpi-compiled@pi05_compiled_regular.service || true
}
trap restore_regular EXIT
"${POLICYCTL[@]}" stop spring-openpi-compiled@pi05_compiled_regular.service \
  spring-openpi-compiled@pi05_compiled_optimized.service || true
"${POLICYCTL[@]}" start "spring-openpi-compiled@${VARIANT}.service"
for _ in $(seq 1 180); do
  curl -fsS http://127.0.0.1:8000/healthz >/dev/null && break
  sleep 5
done
"$HOME/.venv-compiled-policy/bin/python" "$HOME/.local/share/robolab/openpi/deploy/smoke_compiled_policy.py" \
  --expected-policy "$VARIANT"

ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$USER_NAME@$HOST" \
  bash -s -- "$IMAGE" "$VARIANT" "$TASK" "$OUTPUT" <<'REMOTE'
set -euo pipefail
IMAGE="$1"; VARIANT="$2"; TASK="$3"; OUTPUT="$4"
POLICY_PORT=18000
sudo docker run --rm --gpus all --network host --ipc host --shm-size 16g \
  -e POLICY_HOST=127.0.0.1 -e POLICY_PORT="$POLICY_PORT" -e POLICY_VARIANT="$VARIANT" \
  -e TASKS="$TASK" -e NUM_ENVS=1 \
  -v "$HOME/robolab-output:/workspace/robolab/output" \
  --entrypoint /workspace/robolab/docker/run_cloud_eval.sh "$IMAGE" \
  --output-folder-name "$OUTPUT"
REMOTE

printf 'DONE output=%s UI=http://%s:8080 log=%s\n' "$OUTPUT" "$DASHBOARD_IP" "$LOG"
