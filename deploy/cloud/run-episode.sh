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
LOCAL_POLICY_PORT="${ROBOLAB_POLICY_PORT:-8100}"
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
USE_COLOR=0
[[ -t 1 ]] && USE_COLOR=1
exec > >(tee >(sed -u $'s/\033\\[[0-9;]*m//g' >> "$LOG")) 2>&1
format_simulator_output() {
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      *"[Warning]"*|*"Warning:"*|Warp\ CUDA\ error*|cat:\ *cpufreq*)
        if [[ "$USE_COLOR" == 1 ]]; then
          printf '\033[1;33m[WARNING] %s\033[0m\n' "$line"
        else
          printf '[WARNING] %s\n' "$line"
        fi
        ;;
      *) printf '%s\n' "$line" ;;
    esac
  done
}
exec 9>"$ROOT/run.lock"
if ! flock -n 9; then
  echo "Another RoboLab episode is already running on Cleveland. Wait for it to finish, then try again." >&2
  exit 4
fi
if systemctl --user is-active --quiet franka-droid-policy.service; then
  echo "Cleveland is currently running the real-robot Franka/DROID workload." >&2
  echo "RoboLab left it untouched. Run this episode after that workload is intentionally released." >&2
  exit 5
fi
printf '[1/4] Connecting to the cloud simulator...\n'
"$ROOT/setup-ec2.sh" "$HOST" "$USER_NAME"

if systemctl --user cat spring-openpi-compiled@.service >/dev/null 2>&1; then
  POLICYCTL=(systemctl --user)
else
  POLICYCTL=(sudo systemctl)
fi
stop_spring_policies() {
  "${POLICYCTL[@]}" stop spring-openpi-compiled@pi05_compiled_regular.service \
    spring-openpi-compiled@pi05_compiled_optimized.service || true
}
trap stop_spring_policies EXIT
printf '[2/4] Starting %s on the Intel B580 (cold start takes about 45 seconds)' "$MODEL"
"${POLICYCTL[@]}" stop spring-openpi-compiled@pi05_compiled_regular.service \
  spring-openpi-compiled@pi05_compiled_optimized.service >/dev/null 2>&1 || true
"${POLICYCTL[@]}" start "spring-openpi-compiled@${VARIANT}.service"
POLICY_READY=0
for _ in $(seq 1 180); do
  if curl -fs "http://127.0.0.1:$LOCAL_POLICY_PORT/healthz" >/dev/null 2>&1; then
    POLICY_READY=1
    break
  fi
  if ! "${POLICYCTL[@]}" is-active --quiet "spring-openpi-compiled@${VARIANT}.service"; then
    printf '\nPolicy service stopped during startup. Recent service log:\n' >&2
    journalctl --user -u "spring-openpi-compiled@${VARIANT}.service" -n 40 --no-pager >&2 || true
    exit 1
  fi
  printf '.'
  sleep 5
done
printf '\n'
if [[ "$POLICY_READY" != 1 ]]; then
  echo "Policy did not become ready within 15 minutes." >&2
  exit 1
fi
printf '[3/4] Checking policy inference...\n'
"$HOME/.venv-compiled-policy/bin/python" "$HOME/.local/share/robolab/openpi/deploy/smoke_compiled_policy.py" \
  --port "$LOCAL_POLICY_PORT" --expected-policy "$VARIANT"
printf '[4/4] Running %s in Isaac Sim...\n' "$TASK"

ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$USER_NAME@$HOST" \
  bash -s -- "$IMAGE" "$VARIANT" "$TASK" "$OUTPUT" 2>&1 <<'REMOTE' | format_simulator_output
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
