#!/usr/bin/env bash
set -euo pipefail

HOST="${1:?usage: run-episode.sh <gpu-host> <model> <TaskName> [output-name] [ssh-user]}"
MODEL="${2:?missing model}"
TASK="${3:?missing task name}"
OUTPUT="${4:-$(date -u +%Y%m%dT%H%M%SZ)_${MODEL}_${TASK}}"
USER_NAME="${5:-ubuntu}"
IMAGE="${ROBOLAB_IMAGE:-public.ecr.aws/m4l3e1i0/spring-silicon/robolab:2026-09-25}"
OPENPI_IMAGE="${OPENPI_IMAGE:-public.ecr.aws/m4l3e1i0/spring-silicon/openpi-server:2026-09-25}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY="$HOME/.ssh/robolab-cloud-tunnel"
DASHBOARD_IP="${ROBOLAB_DASHBOARD_IP:-$(ip -4 route get 1.1.1.1 | sed -n 's/.* src \([^ ]*\).*/\1/p')}"
[[ "$HOST" =~ ^[A-Za-z0-9.:-]+$ && "$TASK" =~ ^[A-Za-z0-9_]+$ && "$OUTPUT" =~ ^[A-Za-z0-9_.-]+$ ]] \
  || { echo "host, task, or output contains unsupported characters" >&2; exit 2; }

COMPILED=0; CONFIG=""; CHECKPOINT=""
case "$MODEL" in
  regular|pi05_compiled_regular) VARIANT=pi05_compiled_regular; COMPILED=1 ;;
  optimized|pi05_compiled_optimized) VARIANT=pi05_compiled_optimized; COMPILED=1 ;;
  pi0) VARIANT=pi0; CONFIG=pi0_droid_jointpos; CHECKPOINT=gs://openpi-assets-simeval/pi0_droid_jointpos ;;
  pi0_fast) VARIANT=pi0_fast; CONFIG=pi0_fast_droid_jointpos; CHECKPOINT=gs://openpi-assets-simeval/pi0_fast_droid_jointpos ;;
  pi05) VARIANT=pi05; CONFIG=pi05_droid_jointpos; CHECKPOINT=gs://openpi-assets-simeval/pi05_droid_jointpos ;;
  paligemma) VARIANT=paligemma; CONFIG=paligemma_binning_droid_jointpos; CHECKPOINT=gs://openpi-assets-simeval/paligemma_binning_droid_jointpos ;;
  paligemma_fast) VARIANT=paligemma_fast; CONFIG=paligemma_fast_droid_jointpos; CHECKPOINT=gs://openpi-assets-simeval/paligemma_fast_droid_jointpos ;;
  *) echo "model: pi0, pi0_fast, pi05, paligemma, paligemma_fast, regular, or optimized" >&2; exit 2 ;;
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
if (( COMPILED )); then
  "${POLICYCTL[@]}" start "spring-openpi-compiled@${VARIANT}.service"
  for _ in $(seq 1 180); do
    curl -fsS http://127.0.0.1:8000/healthz >/dev/null && break
    sleep 5
  done
  "$HOME/.venv-compiled-policy/bin/python" "$HOME/Desktop/openpi/deploy/smoke_compiled_policy.py" \
    --expected-policy "$VARIANT"
fi

ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$USER_NAME@$HOST" \
  bash -s -- "$IMAGE" "$OPENPI_IMAGE" "$VARIANT" "$TASK" "$OUTPUT" "$CONFIG" "$CHECKPOINT" <<'REMOTE'
set -euo pipefail
IMAGE="$1"; OPENPI_IMAGE="$2"; VARIANT="$3"; TASK="$4"; OUTPUT="$5"; CONFIG="$6"; CHECKPOINT="$7"
POLICY_PORT=18000
if [[ -n "$CONFIG" ]]; then
  mkdir -p "$HOME/openpi-cache"
  sudo docker pull "$OPENPI_IMAGE"
  sudo docker rm -f openpi-policy >/dev/null 2>&1 || true
  sudo docker run -d --name openpi-policy --gpus all --network host \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 -e OPENPI_DATA_HOME=/openpi_assets \
    -v "$HOME/openpi-cache:/openpi_assets" "$OPENPI_IMAGE" \
    policy:checkpoint --policy.config="$CONFIG" --policy.dir="$CHECKPOINT" --port 8000 >/dev/null
  trap 'sudo docker rm -f openpi-policy >/dev/null 2>&1 || true' EXIT
  for _ in $(seq 1 720); do
    curl -fsS http://127.0.0.1:8000/healthz >/dev/null && break
    sleep 5
  done
  curl -fsS http://127.0.0.1:8000/healthz >/dev/null
  POLICY_PORT=8000
fi
sudo docker run --rm --gpus all --network host --ipc host --shm-size 16g \
  -e POLICY_HOST=127.0.0.1 -e POLICY_PORT="$POLICY_PORT" -e POLICY_VARIANT="$VARIANT" \
  -e TASKS="$TASK" -e NUM_ENVS=1 \
  -v "$HOME/robolab-output:/workspace/robolab/output" \
  --entrypoint /workspace/robolab/docker/run_cloud_eval.sh "$IMAGE" \
  --output-folder-name "$OUTPUT"
REMOTE

printf 'DONE output=%s UI=http://%s:8080 log=%s\n' "$OUTPUT" "$DASHBOARD_IP" "$LOG"
