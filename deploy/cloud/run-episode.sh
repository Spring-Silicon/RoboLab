#!/usr/bin/env bash
set -euo pipefail

HOST="${1:?usage: run-episode.sh <ec2-address> <regular|optimized> <TaskName> [output-name] [ssh-user]}"
MODEL="${2:?missing model}"
TASK="${3:?missing task name}"
OUTPUT="${4:-$(date -u +%Y%m%dT%H%M%SZ)_${MODEL}_${TASK}}"
USER_NAME="${5:-ubuntu}"
IMAGE="${ROBOLAB_IMAGE:-public.ecr.aws/m4l3e1i0/spring-silicon/robolab:2026-09-25}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY="$HOME/.ssh/robolab-cloud-tunnel"
[[ "$HOST" =~ ^[A-Za-z0-9.:-]+$ && "$TASK" =~ ^[A-Za-z0-9_]+$ && "$OUTPUT" =~ ^[A-Za-z0-9_.-]+$ ]] \
  || { echo "host, task, or output contains unsupported characters" >&2; exit 2; }
case "$MODEL" in
  regular|pi05_compiled_regular) VARIANT=pi05_compiled_regular ;;
  optimized|pi05_compiled_optimized) VARIANT=pi05_compiled_optimized ;;
  *) echo "model must be regular or optimized" >&2; exit 2 ;;
esac

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/$OUTPUT.log"
exec > >(tee -a "$LOG") 2>&1
set -x
"$ROOT/setup-ec2.sh" "$HOST" "$USER_NAME"

restore_regular() {
  sudo systemctl stop spring-openpi-compiled@pi05_compiled_optimized.service || true
  sudo systemctl start spring-openpi-compiled@pi05_compiled_regular.service || true
}
trap restore_regular EXIT
sudo systemctl stop spring-openpi-compiled@pi05_compiled_regular.service \
  spring-openpi-compiled@pi05_compiled_optimized.service || true
sudo systemctl start "spring-openpi-compiled@${VARIANT}.service"
for _ in $(seq 1 180); do
  curl -fsS http://127.0.0.1:8000/healthz >/dev/null && break
  sleep 5
done
/home/sentradel/.venv-compiled-policy/bin/python \
  /home/sentradel/Spring-openpi/deploy/smoke_compiled_policy.py --expected-policy "$VARIANT"

ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$USER_NAME@$HOST" \
  bash -s -- "$IMAGE" "$VARIANT" "$TASK" "$OUTPUT" <<'REMOTE'
set -euo pipefail
IMAGE="$1"; VARIANT="$2"; TASK="$3"; OUTPUT="$4"
sudo docker run --rm --gpus all --network host --ipc host --shm-size 16g \
  -e POLICY_HOST=127.0.0.1 -e POLICY_PORT=18000 -e POLICY_VARIANT="$VARIANT" \
  -e TASKS="$TASK" -e NUM_ENVS=1 \
  -v "$HOME/robolab-output:/workspace/robolab/output" \
  --entrypoint /workspace/robolab/docker/run_cloud_eval.sh "$IMAGE" \
  --output-folder-name "$OUTPUT"
REMOTE

printf 'DONE output=%s UI=http://100.123.6.81:8080 log=%s\n' "$OUTPUT" "$LOG"
